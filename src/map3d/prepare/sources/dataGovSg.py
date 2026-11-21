"""Singapore official open data (data.gov.sg) fetchers and adapters.

Datasets were inventoried on 2026-11-20 (see ``docs/research.md``). All are
under the Singapore Open Data Licence v1.0.

- **HDB footprints + storeys** (``sgHdbBuildings``): footprints carry
  ``BLK_NO`` + ``ST_COD`` (a street *code*). HDB Property Information carries
  ``blk_no`` + ``street`` (a street *name*) + ``max_floor_lvl``. There is no
  shared key. :func:`joinStreetCodes` maps each code to the name whose
  block-number set best matches, breaking ties by the code's two-letter prefix
  (``YIA06H`` -> YIshun Ave). That joins ~98.6% of footprints, fully offline.
- **URA Master Plan building outlines** (``sgUraBuildings``): no heights.
- **URA Master Plan rail** (``sgMasterPlanRail``): ``GRND_LEVEL ==
  ABOVEGROUND`` segments are kept as bridges; underground ones are dropped.
- **LTA bridges/flyovers** (``sgLtaElevatedStructures``): footprints only;
  tunnels and underpasses are excluded.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import shapely

from map3d.pack.geoparquet import writeGeoParquet
from map3d.pack.manifest import Bounds
from map3d.prepare.http import HttpSession

__all__ = [
    "DATASETS",
    "MIN_INTERVAL_s",
    "SINGAPORE_BOUNDS",
    "adaptHdbBuildings",
    "adaptLtaStructures",
    "adaptMasterPlanRail",
    "adaptUraBuildings",
    "downloadDataset",
    "intersectsSingapore",
    "joinStreetCodes",
]

_API = "https://api-open.data.gov.sg/v1/public/api/datasets/{}/{}"
MIN_INTERVAL_s = {"api-open.data.gov.sg": 5.0}  # 2 requests / 10 s without an API key
SINGAPORE_BOUNDS: Bounds = (103.55, 1.13, 104.15, 1.48)

DATASETS: dict[str, tuple[str, bool]] = {
    "hdbBuildings": ("d_16b157c52ed637edd6ba1232e026258d", False),
    "hdbPropertyInfo": ("d_17f5382f26140b1fdae0ba2ef6239d2f", True),
    "uraBuildings": ("d_255551acbbce32df735e14d62e067e4f", False),
    "mpRail": ("d_1c6365f0cad13a77bd79bdbb499131bf", False),
    "ltaStructures": ("d_97268dd70d5634bbed12c7845852ec4b", False),
}

_EXCLUDED_STRUCTURES = {"Vehicular Tunnel", "Vehicular Underpass"}


def intersectsSingapore(bounds: Bounds) -> bool:
    """Whether a lon/lat box overlaps Singapore."""
    return bool(shapely.intersects(shapely.box(*bounds), shapely.box(*SINGAPORE_BOUNDS)))


def downloadDataset(
    session: HttpSession, datasetId: str, target: Path, isCsv: bool, maxPolls: int = 12
) -> Path:
    """Download a data.gov.sg dataset file via the poll-download API (cached).

    Args:
        session: HTTP session (should throttle ``api-open.data.gov.sg``).
        datasetId: Dataset id, e.g. ``d_16b1...``.
        target: Cache path for the file.
        isCsv: CSV datasets must be initiated before polling.
        maxPolls: Poll attempts before giving up.

    Returns:
        The downloaded file path.

    Raises:
        RuntimeError: If the download URL never becomes available.
    """
    if target.is_file():
        return target
    if isCsv:
        session.get(_API.format(datasetId, "initiate-download"))
    for _ in range(maxPolls):
        data = session.get(_API.format(datasetId, "poll-download")).json().get("data") or {}
        if data.get("url"):
            return session.download(data["url"], target)
    raise RuntimeError(f"data.gov.sg dataset {datasetId} was not ready after {maxPolls} polls")


def joinStreetCodes(
    codeBlocks: dict[str, set[str]], nameBlocks: dict[str, set[str]]
) -> dict[str, str]:
    """Map HDB street codes to street names by matching block-number sets.

    Args:
        codeBlocks: Street code -> block numbers seen in the footprints.
        nameBlocks: Street name -> block numbers in Property Information.

    Returns:
        Resolved code -> name mapping. A code is resolved when its best name
        covers at least half its blocks and is unique after the prefix
        tie-break.
    """
    byBlock: dict[str, set[str]] = defaultdict(set)
    for name, blocks in nameBlocks.items():
        for block in blocks:
            byBlock[block].add(name)
    mapping: dict[str, str] = {}
    for code, blocks in codeBlocks.items():
        hits: Counter[str] = Counter(n for b in blocks for n in byBlock.get(b, ()))
        if not hits:
            continue
        ranked = sorted(
            hits.items(), key=lambda kv: (kv[1], _prefix(kv[0]) == code[:2]), reverse=True
        )
        best, count = ranked[0]
        key = (count, _prefix(best) == code[:2])
        if count / len(blocks) < 0.5:
            continue
        if len(ranked) > 1 and (ranked[1][1], _prefix(ranked[1][0]) == code[:2]) == key:
            continue
        mapping[code] = best
    return mapping


def adaptHdbBuildings(
    footprintsGeojson: Path, propertyCsv: Path, bounds: Bounds, outPath: Path
) -> Path:
    """HDB footprints joined to storey counts -> canonical buildings GeoParquet.

    Args:
        footprintsGeojson: "HDB Existing Building" GeoJSON.
        propertyCsv: "HDB Property Information" CSV.
        bounds: Clip box.
        outPath: Output path.

    Returns:
        ``outPath``.
    """
    allGeoms, allProps = _readGeoJson(footprintsGeojson, None)
    floors: dict[tuple[str, str], int] = {}
    nameBlocks: dict[str, set[str]] = defaultdict(set)
    with propertyCsv.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            street, block = row["street"].strip().upper(), row["blk_no"].strip().upper()
            nameBlocks[street].add(block)
            if row.get("max_floor_lvl", "").strip().isdigit():
                floors[(street, block)] = int(row["max_floor_lvl"])
    # Street codes are matched island-wide; only the output is clipped.
    codeBlocks: dict[str, set[str]] = defaultdict(set)
    for p in allProps:
        codeBlocks[str(p["ST_COD"]).strip()].add(str(p["BLK_NO"]).strip().upper())
    mapping = joinStreetCodes(codeBlocks, nameBlocks)
    keep = np.flatnonzero(shapely.intersects(allGeoms, shapely.box(*bounds)))
    geoms, props = allGeoms[keep], [allProps[i] for i in keep]
    numFloors: list[int | None] = []
    for p in props:
        street = mapping.get(str(p["ST_COD"]).strip())
        numFloors.append(floors.get((street, str(p["BLK_NO"]).strip().upper())) if street else None)
    writeGeoParquet(
        outPath,
        geoms,
        {
            "sourceFeatureId": [str(p.get("POSTAL_COD") or p.get("OBJECTID")) for p in props],
            "height_m": [None] * len(props),
            "numFloors": numFloors,
        },
    )
    return outPath


def adaptUraBuildings(geojson: Path, bounds: Bounds, outPath: Path) -> Path:
    """URA Master Plan building outlines -> canonical buildings (no heights)."""
    geoms, props = _readGeoJson(geojson, bounds)
    n = len(props)
    writeGeoParquet(
        outPath,
        geoms,
        {
            "sourceFeatureId": [str(p["OBJECTID"]) for p in props],
            "height_m": [None] * n,
            "numFloors": [None] * n,
        },
    )
    return outPath


def adaptMasterPlanRail(geojson: Path, bounds: Bounds, outPath: Path) -> Path:
    """URA Master Plan rail -> canonical transport (aboveground segments as bridges)."""
    geoms, props = _readGeoJson(geojson, bounds)
    keep = [i for i, p in enumerate(props) if p.get("GRND_LEVEL") == "ABOVEGROUND"]
    n = len(keep)
    writeGeoParquet(
        outPath,
        geoms[keep],
        {
            "sourceFeatureId": [str(props[i]["OBJECTID"]) for i in keep],
            "kind": ["rail"] * n,
            "isBridge": [True] * n,
            "level": [None] * n,
            "width_m": [None] * n,
        },
    )
    return outPath


def adaptLtaStructures(geojson: Path, bounds: Bounds, outPath: Path) -> Path:
    """LTA bridges/flyovers -> canonical elevated-structure footprints."""
    geoms, props = _readGeoJson(geojson, bounds)
    keep = [i for i, p in enumerate(props) if p.get("TYP_CD") not in _EXCLUDED_STRUCTURES]
    writeGeoParquet(
        outPath,
        geoms[keep],
        {
            "sourceFeatureId": [
                str(props[i].get("OBJECTID_1") or props[i].get("OBJECTID")) for i in keep
            ],
            "structureType": [str(props[i]["TYP_CD"]) for i in keep],
        },
    )
    return outPath


def _prefix(name: str) -> str:
    return "".join(ch for ch in name if ch.isalpha())[:2]


def _readGeoJson(path: Path, bounds: Bounds | None) -> tuple[Any, list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    geoms = np.array(
        [shapely.geometry.shape(f["geometry"]) for f in data["features"]], dtype=object
    )
    props = [f.get("properties") or {} for f in data["features"]]
    if bounds is None:
        return geoms, props
    keep = np.flatnonzero(shapely.intersects(geoms, shapely.box(*bounds)))
    return geoms[keep], [props[i] for i in keep]
