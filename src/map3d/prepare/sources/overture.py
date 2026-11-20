"""Overture Maps fetchers: buildings and transportation segments.

Reads the public GeoParquet release on S3 (``overturemaps-us-west-2``) with
Arrow's own S3 client, pushing a filter on the ``bbox`` struct column down to
row-group statistics, and writes the canonical pack schemas:

- **buildings:** ``sourceFeatureId``, ``height_m``, ``numFloors``. Underground
  buildings are excluded.
- **transport:** ``sourceFeatureId``, ``kind`` (road/rail/footway), ``isBridge``,
  ``level``, ``width_m``. Overture's linear-referenced rules (``between``
  fractions along the segment) are honoured by splitting segments at rule
  boundaries. Each piece takes the flags, level and width at its midpoint, and
  tunnel pieces are dropped.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import shapely
from shapely.ops import substring

from map3d.pack.geoparquet import writeGeoParquet

__all__ = [
    "BUCKET",
    "fetchOvertureBuildings",
    "fetchOvertureTransport",
    "latestRelease",
    "s3FileSystem",
]

Bounds = tuple[float, float, float, float]

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
    """Extract buildings intersecting ``bounds`` into canonical GeoParquet.

    Args:
        fs: Filesystem holding the release.
        releaseBase: ``<bucket>/release/<version>``.
        bounds: ``(minLon, minLat, maxLon, maxLat)``.
        outPath: Output GeoParquet path.

    Returns:
        ``outPath``.
    """
    dataset = ds.dataset(
        f"{releaseBase}/theme=buildings/type=building", filesystem=fs, format="parquet"
    )
    underground = pc.field("is_underground")
    table = dataset.to_table(
        columns=["id", "height", "num_floors", "geometry"],
        filter=_bboxFilter(bounds) & (underground.is_null() | (underground == False)),  # noqa: E712
    )
    geoms = shapely.from_wkb(table.column("geometry").to_numpy(zero_copy_only=False))
    keep = shapely.intersects(geoms, shapely.box(*bounds))
    table = table.filter(keep)
    writeGeoParquet(
        outPath,
        geoms[keep],
        {
            "sourceFeatureId": table.column("id").to_pylist(),
            "height_m": table.column("height").cast("float64").to_pylist(),
            "numFloors": table.column("num_floors").to_pylist(),
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
