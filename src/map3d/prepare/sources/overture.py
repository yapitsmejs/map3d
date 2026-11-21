"""Overture Maps fetchers: buildings and transportation segments.

Reads the public GeoParquet release on S3 (``overturemaps-us-west-2``) with
Arrow's own S3 client, pushing a filter on the ``bbox`` struct column down to
row-group statistics, and writes the canonical pack schemas:

- **buildings:** ``sourceFeatureId``, ``height_m``, ``numFloors``, plus the
  optional 3D columns ``minHeight_m``, ``minFloor``, ``roofShape``,
  ``roofHeight_m``, ``roofDirection_deg``, ``roofOrientation``, ``facadeColor``
  and ``roofColor`` (OSM Simple 3D Buildings tags). Where a building is mapped as **building parts**
  (``building_part`` rows), the parts replace its outline, as Simple 3D
  Buildings renderers do. Underground buildings and parts are excluded.
- **water:** polygonal water (ocean, reservoirs, rivers, ponds, pools) clipped
  to the box, as land-cover polygons of class water (``landClass`` 80).
- **land use:** surface-like ``land_use`` (parks, grass, plazas, construction
  sites, farmland) and natural ``land`` (woods, scrub, grass, wetland, sand)
  areas as land-cover polygons with WorldCover classes.
- **transport:** ``sourceFeatureId``, ``kind`` (road/rail/footway), ``isBridge``,
  ``level``, ``width_m``. Overture's linear-referenced rules (``between``
  fractions along the segment) are honoured by splitting segments at rule
  boundaries. Each piece takes the flags, level and width at its midpoint, and
  tunnel pieces are dropped.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import shapely
from shapely.ops import substring

from map3d.pack.geoparquet import writeGeoParquet
from map3d.pack.manifest import Bounds

__all__ = [
    "BUCKET",
    "fetchOvertureBuildings",
    "fetchOvertureLandUse",
    "fetchOvertureTransport",
    "fetchOvertureWater",
    "latestRelease",
    "s3FileSystem",
]

BUCKET = "overturemaps-us-west-2"
_EPS = 1e-9
# Overture road classes that are paths for people, not vehicles.
FOOTWAY_CLASSES = frozenset(
    {"footway", "pedestrian", "path", "cycleway", "steps", "bridleway", "sidewalk", "crosswalk"}
)


def s3FileSystem(caBundle: str | os.PathLike[str] | None) -> pafs.S3FileSystem:
    """Anonymous S3 filesystem for the Overture bucket.

    Args:
        caBundle: PEM CA bundle (for TLS-intercepting networks), or ``None``.

    Returns:
        The filesystem.
    """
    kwargs: dict[str, Any] = {"anonymous": True, "region": "us-west-2"}
    if caBundle is not None:
        kwargs["tls_ca_file_path"] = str(caBundle)
    return pafs.S3FileSystem(**kwargs)


def latestRelease(fs: pafs.FileSystem, bucket: str = BUCKET) -> str:
    """Name of the newest release directory, e.g. ``2026-11-18.0``."""
    infos = fs.get_file_info(pafs.FileSelector(f"{bucket}/release/"))
    return str(max(info.base_name for info in infos if info.type == pafs.FileType.Directory))


def fetchOvertureBuildings(
    fs: pafs.FileSystem, releaseBase: str, bounds: Bounds, outPath: Path
) -> Path:
    """Extract buildings (and building parts) intersecting ``bounds`` into canonical GeoParquet.

    Args:
        fs: Filesystem holding the release.
        releaseBase: ``<bucket>/release/<version>``.
        bounds: ``(minLon, minLat, maxLon, maxLat)``.
        outPath: Output GeoParquet path.

    Returns:
        ``outPath``.
    """
    buildings = _readBuildingType(fs, f"{releaseBase}/theme=buildings/type=building", bounds, [])
    parts = _readBuildingType(
        fs, f"{releaseBase}/theme=buildings/type=building_part", bounds, ["building_id"]
    )
    replaced = {str(v) for v in parts.get("building_id", [])}
    keepBuilding = [str(i) not in replaced for i in buildings["id"]]
    geoms: list[Any] = []
    cols: dict[str, list[Any]] = {name: [] for name in _CANONICAL.values()}
    for table, keep in ((buildings, keepBuilding), (parts, [True] * len(parts["id"]))):
        for k in np.flatnonzero(np.asarray(keep, dtype=bool)):
            geoms.append(table["geometry"][k])
            for raw, name in _CANONICAL.items():
                cols[name].append(table[raw][k])
    writeGeoParquet(outPath, np.array(geoms, dtype=object), cols)
    return outPath


# Overture attribute -> canonical pack column.
_CANONICAL = {
    "id": "sourceFeatureId",
    "height": "height_m",
    "num_floors": "numFloors",
    "min_height": "minHeight_m",
    "min_floor": "minFloor",
    "roof_shape": "roofShape",
    "roof_height": "roofHeight_m",
    "roof_direction": "roofDirection_deg",
    "roof_orientation": "roofOrientation",
    "facade_color": "facadeColor",
    "roof_color": "roofColor",
}
_FLOAT_ATTRS = ("height", "min_height", "roof_height", "roof_direction")


def _readBuildingType(
    fs: pafs.FileSystem, path: str, bounds: Bounds, extra: list[str]
) -> dict[str, list[Any]]:
    """Rows of one building type intersecting ``bounds``, as columns (missing columns are null)."""
    names = [*_CANONICAL, *extra]
    empty: dict[str, list[Any]] = {name: [] for name in [*names, "geometry"]}
    if fs.get_file_info(path).type == pafs.FileType.NotFound:
        return empty
    dataset = ds.dataset(path, filesystem=fs, format="parquet")
    present = [n for n in names if n in dataset.schema.names]
    underground = pc.field("is_underground")
    table = dataset.to_table(
        columns=[*present, "geometry"],
        filter=_bboxFilter(bounds) & (underground.is_null() | (underground == False)),  # noqa: E712
    )
    geoms = shapely.from_wkb(table.column("geometry").to_numpy(zero_copy_only=False))
    keep = shapely.intersects(geoms, shapely.box(*bounds))
    table = table.filter(keep)
    out: dict[str, list[Any]] = {"geometry": list(geoms[keep])}
    for name in names:
        if name not in present:
            out[name] = [None] * table.num_rows
        elif name in _FLOAT_ATTRS:
            out[name] = table.column(name).cast("float64").to_pylist()
        else:
            out[name] = table.column(name).to_pylist()
    return out


def fetchOvertureWater(
    fs: pafs.FileSystem, releaseBase: str, bounds: Bounds, outPath: Path
) -> Path:
    """Extract polygonal water intersecting ``bounds`` as land-cover polygons.

    Args:
        fs: Filesystem holding the release.
        releaseBase: ``<bucket>/release/<version>``.
        bounds: ``(minLon, minLat, maxLon, maxLat)``.
        outPath: Output GeoParquet path.

    Returns:
        ``outPath``.
    """
    polygons = _coverPolygons(
        fs, f"{releaseBase}/theme=base/type=water", bounds, lambda subtype, cls: WATER_CLASS
    )
    return _writeCoverPolygons(outPath, [polygons])


def fetchOvertureLandUse(
    fs: pafs.FileSystem, releaseBase: str, bounds: Bounds, outPath: Path
) -> Path:
    """Extract surface-like land use and natural land cover as land-cover polygons.

    Overture ``land_use`` areas that describe a surface (parks, managed grass,
    pitches, cemeteries: grass; pedestrian plazas: built-up; construction
    sites: bare; farmland: cropland) and ``land`` areas (woods, scrub, grass,
    wetland, sand, rock), clipped to the box. Zoning-like land use
    (residential, commercial, ...) is skipped: it says nothing about the
    surface.

    Args:
        fs: Filesystem holding the release.
        releaseBase: ``<bucket>/release/<version>``.
        bounds: ``(minLon, minLat, maxLon, maxLat)``.
        outPath: Output GeoParquet path.

    Returns:
        ``outPath``.
    """
    landUse = _coverPolygons(fs, f"{releaseBase}/theme=base/type=land_use", bounds, _landUseClass)
    land = _coverPolygons(fs, f"{releaseBase}/theme=base/type=land", bounds, _landClass)
    return _writeCoverPolygons(outPath, [landUse, land])


WATER_CLASS = 80  # WorldCover "permanent water bodies"
# Overture land_use class (else subtype) -> WorldCover code.
_LAND_USE_CLASSES = {
    **dict.fromkeys(
        (
            "park",
            "garden",
            "grass",
            "village_green",
            "meadow",
            "pitch",
            "golf_course",
            "recreation_ground",
            "cemetery",
            "grave_yard",
            "dog_park",
        ),
        30,
    ),
    **dict.fromkeys(("pedestrian", "plaza"), 50),
    **dict.fromkeys(("construction", "brownfield", "landfill", "quarry"), 60),
    **dict.fromkeys(("farmland", "allotments", "orchard", "plant_nursery"), 40),
}
# Overture land subtype -> WorldCover code ("land" itself, islands etc. are skipped).
_LAND_SUBTYPES = {
    "forest": 10,
    "shrub": 20,
    "grass": 30,
    "wetland": 90,
    "sand": 60,
    "rock": 60,
    "glacier": 70,
}


def _landUseClass(subtype: str | None, cls: str | None) -> int | None:
    return _LAND_USE_CLASSES.get(cls or "", _LAND_USE_CLASSES.get(subtype or ""))


def _landClass(subtype: str | None, cls: str | None) -> int | None:
    return _LAND_SUBTYPES.get(subtype or "")


_CoverPolygons = tuple[npt.NDArray[np.object_], list[str], list[int]]


def _coverPolygons(
    fs: pafs.FileSystem,
    path: str,
    bounds: Bounds,
    classify: Callable[[str | None, str | None], int | None],
) -> _CoverPolygons:
    """Polygons of an Overture base type, clipped to ``bounds``, with a land class each.

    ``classify(subtype, class)`` returns a WorldCover code, or ``None`` to skip.

    Returns:
        ``(polygons, source feature ids, class codes)``.
    """
    dataset = ds.dataset(path, filesystem=fs, format="parquet")
    table = dataset.to_table(
        columns=["id", "subtype", "class", "geometry"], filter=_bboxFilter(bounds)
    )
    geoms = shapely.from_wkb(table.column("geometry").to_numpy(zero_copy_only=False))
    box = shapely.box(*bounds)
    codes = [
        classify(s, c)
        for s, c in zip(
            table.column("subtype").to_pylist(), table.column("class").to_pylist(), strict=True
        )
    ]
    polygonal = np.isin(shapely.get_type_id(geoms), (3, 6, 7))  # (multi)polygons, collections
    keep = polygonal & np.array([c is not None for c in codes], dtype=bool)
    keep &= shapely.intersects(geoms, box)
    idx = np.flatnonzero(keep)
    clipped = shapely.intersection(shapely.make_valid(geoms[idx]), box)
    # Clipping can leave lines or collections: keep the polygon parts.
    parts, owner = shapely.get_parts(clipped, return_index=True)
    poly = (shapely.get_type_id(parts) == 3) & (shapely.area(parts) > 0)
    parts, src = parts[poly], idx[owner[poly]]
    ids = table.column("id").to_pylist()
    return (
        np.asarray(parts, dtype=object),
        [ids[i] for i in src],
        [int(codes[i] or 0) for i in src],
    )


def _writeCoverPolygons(outPath: Path, groups: list[_CoverPolygons]) -> Path:
    writeGeoParquet(
        outPath,
        np.concatenate([g[0] for g in groups]),
        {
            "sourceFeatureId": [i for g in groups for i in g[1]],
            "landClass": [c for g in groups for c in g[2]],
        },
    )
    return outPath


def fetchOvertureTransport(
    fs: pafs.FileSystem, releaseBase: str, bounds: Bounds, outPath: Path
) -> Path:
    """Extract road/rail segments intersecting ``bounds`` into canonical GeoParquet.

    Args:
        fs: Filesystem holding the release.
        releaseBase: ``<bucket>/release/<version>``.
        bounds: ``(minLon, minLat, maxLon, maxLat)``.
        outPath: Output GeoParquet path.

    Returns:
        ``outPath``.
    """
    dataset = ds.dataset(
        f"{releaseBase}/theme=transportation/type=segment", filesystem=fs, format="parquet"
    )
    table = dataset.to_table(
        columns=[
            "id",
            "subtype",
            "class",
            "road_flags",
            "rail_flags",
            "level_rules",
            "width_rules",
            "geometry",
        ],
        filter=_bboxFilter(bounds) & pc.field("subtype").isin(["road", "rail"]),
    )
    box = shapely.box(*bounds)
    geoms: list[Any] = []
    cols: dict[str, list[Any]] = {
        k: [] for k in ("sourceFeatureId", "kind", "isBridge", "level", "width_m")
    }
    for row in table.to_pylist():
        line = shapely.from_wkb(row["geometry"])
        if not shapely.intersects(line, box):
            continue
        for pieceId, piece, isBridge, level, width in _splitSegment(row, line):
            geoms.append(piece)
            cols["sourceFeatureId"].append(pieceId)
            cols["kind"].append(_kind(row))
            cols["isBridge"].append(isBridge)
            cols["level"].append(level)
            cols["width_m"].append(width)
    writeGeoParquet(outPath, np.array(geoms, dtype=object), cols)
    return outPath


def _kind(row: dict[str, Any]) -> str:
    if row["subtype"] == "road" and row.get("class") in FOOTWAY_CLASSES:
        return "footway"
    return str(row["subtype"])


def _bboxFilter(bounds: Bounds) -> pc.Expression:
    minLon, minLat, maxLon, maxLat = bounds
    return (
        (pc.field("bbox", "xmin") <= maxLon)
        & (pc.field("bbox", "xmax") >= minLon)
        & (pc.field("bbox", "ymin") <= maxLat)
        & (pc.field("bbox", "ymax") >= minLat)
    )


def _covers(between: list[float] | None, m: float) -> bool:
    return between is None or between[0] - _EPS <= m <= between[1] + _EPS


def _splitSegment(row: dict[str, Any], line: Any) -> list[tuple[str, Any, bool, int, float | None]]:
    flagRules = (row.get("road_flags") or []) + (row.get("rail_flags") or [])
    levelRules = row.get("level_rules") or []
    widthRules = row.get("width_rules") or []
    breaks = {0.0, 1.0}
    for rule in flagRules + levelRules:
        if rule.get("between"):
            breaks.update(min(1.0, max(0.0, b)) for b in rule["between"])
    edges = sorted(breaks)
    spans = [(a, b) for a, b in zip(edges[:-1], edges[1:], strict=True) if b - a > _EPS]
    pieces: list[tuple[str, Any, bool, int, float | None]] = []
    for k, (a, b) in enumerate(spans):
        m = 0.5 * (a + b)
        flags = {
            v
            for rule in flagRules
            if _covers(rule.get("between"), m)
            for v in (rule.get("values") or [])
        }
        if "is_tunnel" in flags:
            continue
        level = next((r["value"] for r in levelRules if _covers(r.get("between"), m)), 0)
        width = next((r["value"] for r in widthRules if _covers(r.get("between"), m)), None)
        geom = line if len(spans) == 1 else substring(line, a, b, normalized=True)
        pieceId = row["id"] if len(spans) == 1 else f"{row['id']}#{k}"
        pieces.append((pieceId, geom, "is_bridge" in flags, int(level), width))
    return pieces
