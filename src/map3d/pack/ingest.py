"""``map3d ingest``: validate files and add them to a pack (SPEC section 9).

Files must already be in pack-ready form: GeoTIFF for raster roles, and
GeoParquet in the role's canonical schema for vector roles (see
:mod:`map3d.pack.registry`). Converting raw downloads into that form is the job
of the per-source adapters used by ``map3d prepare``.

All files are validated before any is copied, so a failed ingest leaves the
pack unchanged.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds

from map3d.geoid import GeoidGrid
from map3d.pack.geoparquet import inspectGeoParquet
from map3d.pack.manifest import (
    MANIFEST_NAME,
    Bounds,
    Manifest,
    PackFile,
    SourceEntry,
    fileSha256,
    validateBounds,
)
from map3d.pack.registry import ROLES, SourceSpec, getSource

__all__ = ["ingestFiles"]

_RASTER_SUFFIXES = (".tif", ".tiff")
_VECTOR_SUFFIXES = (".parquet",)


@dataclass(frozen=True)
class _Checked:
    path: Path
    bounds_deg: Bounds | None
    crsWkt: str | None


def ingestFiles(
    packDir: str | os.PathLike[str],
    sourceId: str,
    files: Sequence[str | os.PathLike[str]],
    bounds_deg: Sequence[float] | None = None,
    accessedDate: str | None = None,
) -> Manifest:
    """Validate ``files`` as data for ``sourceId`` and copy them into a pack.

    Args:
        packDir: Pack root. Created if it does not exist (then ``bounds_deg`` is required).
        sourceId: Registry source identifier.
        files: GeoTIFF (raster sources) or canonical GeoParquet (vector sources).
        bounds_deg: Pack coverage ``(minLon, minLat, maxLon, maxLat)``; required for a
            new pack, and must match if given for an existing one.
        accessedDate: ISO date (``YYYY-MM-DD``) the data was obtained; defaults to
            today (UTC). Used in attribution notices.

    Returns:
        The updated manifest (also saved to the pack).

    Raises:
        ValueError: If any input is invalid; the pack is then left unchanged.
    """
    root = Path(packDir)
    accessed = _checkDate(accessedDate)
    manifest = _openOrCreate(root, bounds_deg)
    spec = getSource(sourceId)
    if not files:
        raise ValueError("no files given to ingest")

    existing = manifest.sources.get(sourceId)
    expectedCrs = existing.crs if existing is not None and existing.files else None
    checked: list[_Checked] = []
    for file in files:
        item = _check(Path(file), spec, expectedCrs, manifest.bounds_deg)
        expectedCrs = expectedCrs or item.crsWkt
        checked.append(item)

    entry = existing or SourceEntry.fromSpec(spec)
    entry.crs = expectedCrs if spec.kind == "raster" else None
    entry.accessedDate = accessed
    targetDir = root / spec.kind / sourceId
    targetDir.mkdir(parents=True, exist_ok=True)
    for item in checked:
        target = targetDir / item.path.name
        shutil.copyfile(item.path, target)
        relPath = target.relative_to(root).as_posix()
        entry.files = [f for f in entry.files if f.path != relPath]
        entry.files.append(
            PackFile(relPath, fileSha256(target), target.stat().st_size, item.bounds_deg)
        )
    manifest.sources[sourceId] = entry
    manifest.save(root)
    return manifest


def _checkDate(accessedDate: str | None) -> str:
    if accessedDate is None:
        return datetime.now(UTC).date().isoformat()
    try:
        return date.fromisoformat(accessedDate).isoformat()
    except ValueError:
        raise ValueError(f"accessedDate must be YYYY-MM-DD, got {accessedDate!r}") from None


def _openOrCreate(root: Path, bounds_deg: Sequence[float] | None) -> Manifest:
    if (root / MANIFEST_NAME).is_file():
        manifest = Manifest.load(root)
        if bounds_deg is not None and validateBounds(bounds_deg) != manifest.bounds_deg:
            raise ValueError(
                f"pack already covers {manifest.bounds_deg}; cannot change bounds to "
                f"{tuple(bounds_deg)}"
            )
        return manifest
    if bounds_deg is None:
        raise ValueError(f"bounds are required to create a new pack at {root}")
    manifest = Manifest.new(bounds_deg)
    root.mkdir(parents=True, exist_ok=True)
    return manifest


def _check(path: Path, spec: SourceSpec, expectedCrs: str | None, packBounds: Bounds) -> _Checked:
    if not path.is_file():
        raise ValueError(f"file not found: {path}")
    if spec.kind == "raster":
        item = _checkRaster(path, spec, expectedCrs)
    else:
        item = _checkVector(path, spec)
    if item.bounds_deg is not None and not _intersects(item.bounds_deg, packBounds):
        raise ValueError(
            f"{path} (extent {item.bounds_deg}) does not intersect the pack {packBounds}"
        )
    return item


def _checkRaster(path: Path, spec: SourceSpec, expectedCrs: str | None) -> _Checked:
    if path.suffix.lower() not in _RASTER_SUFFIXES:
        raise ValueError(f"{path}: raster source {spec.sourceId!r} files must be GeoTIFF (.tif)")
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"{path} has no CRS")
        crsWkt = str(src.crs.to_wkt())
        if expectedCrs is not None and rasterio.crs.CRS.from_wkt(expectedCrs) != src.crs:
            raise ValueError(f"{path}: CRS differs from the other files of {spec.sourceId!r}")
        bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    if spec.role == "geoid":
        GeoidGrid.fromGeoTiff(path)  # raises unless a north-up geographic grid
    return _Checked(path, validateBounds(bounds), crsWkt)


def _checkVector(path: Path, spec: SourceSpec) -> _Checked:
    if path.suffix.lower() not in _VECTOR_SUFFIXES:
        raise ValueError(f"{path}: vector source {spec.sourceId!r} files must be GeoParquet")
    info = inspectGeoParquet(path)
    missing = [c for c in ROLES[spec.role].requiredColumns if c not in info.columns]
    if missing:
        raise ValueError(f"{path}: missing canonical columns for role {spec.role!r}: {missing}")
    return _Checked(path, info.bounds_deg, None)


def _intersects(a: Bounds, b: Bounds) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]
