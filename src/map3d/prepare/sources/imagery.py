"""Sentinel-2 cloudless imagery (``sentinel2Composite``) from EOX's public WMTS.

Tiles are Web Mercator 256 px JPEGs; zoom 14 is ~9.6 m/px at the equator,
matching Sentinel-2's 10 m. The tiles covering the box are downloaded
(cached) and mosaicked into one RGB GeoTIFF in EPSG:3857. Licence: CC BY-NC-SA
4.0 (2018+ releases): fine for internal use, attributed in every output.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import httpx
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from map3d.pack.manifest import Bounds
from map3d.prepare.http import HttpSession

__all__ = ["DEFAULT_YEAR", "DEFAULT_ZOOM", "fetchSentinel2Cloudless", "tileRange", "tileUrl"]

DEFAULT_YEAR = 2024
DEFAULT_ZOOM = 14
_TILE_PX = 256
_HALF_WORLD_m = math.pi * 6378137.0
_WMTS = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-{year}_3857/default/g/{z}/{y}/{x}.jpg"


def tileRange(bounds: Bounds, zoom: int) -> tuple[int, int, int, int]:
    """Inclusive Web Mercator tile range ``(x0, y0, x1, y1)`` covering a lon/lat box."""
    minLon, minLat, maxLon, maxLat = bounds
    n = 1 << zoom

    def tx(lon: float) -> int:
        return min(n - 1, int((lon + 180.0) / 360.0 * n))

    def ty(lat: float) -> int:
        return min(n - 1, int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n))

    return tx(minLon), ty(maxLat), tx(maxLon), ty(minLat)


def tileUrl(year: int, zoom: int, x: int, y: int) -> str:
    """WMTS URL of one tile."""
    return _WMTS.format(year=year, z=zoom, x=x, y=y)


def fetchSentinel2Cloudless(
    session: HttpSession,
    bounds: Bounds,
    workDir: Path,
    year: int = DEFAULT_YEAR,
    zoom: int = DEFAULT_ZOOM,
) -> list[Path]:
    """Download and mosaic the imagery tiles covering ``bounds``.

    Args:
        session: HTTP session.
        bounds: ``(minLon, minLat, maxLon, maxLat)``.
        workDir: Working directory (tiles cached in ``raw/``).
        year: s2cloudless release year.
        zoom: WMTS zoom level.

    Returns:
        One RGB GeoTIFF (EPSG:3857). Missing tiles are left black.
    """
    x0, y0, x1, y1 = tileRange(bounds, zoom)
    mosaic = np.zeros((3, (y1 - y0 + 1) * _TILE_PX, (x1 - x0 + 1) * _TILE_PX), dtype=np.uint8)
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            cached = workDir / "raw" / f"s2cloudless{year}_{zoom}_{x}_{y}.jpg"
            try:
                session.download(tileUrl(year, zoom, x, y), cached)
            except httpx.HTTPStatusError:
                continue
            with warnings.catch_warnings():  # WMTS JPEGs carry no georeferencing by design
                warnings.simplefilter("ignore", NotGeoreferencedWarning)
                with MemoryFile(cached.read_bytes()) as mem, mem.open() as tile:
                    r, c = (y - y0) * _TILE_PX, (x - x0) * _TILE_PX
                    mosaic[:, r : r + _TILE_PX, c : c + _TILE_PX] = tile.read(indexes=[1, 2, 3])
    size_m = 2 * _HALF_WORLD_m / 2**zoom
    left = x0 * size_m - _HALF_WORLD_m
    top = _HALF_WORLD_m - y0 * size_m
    out = workDir / f"sentinel2Composite_{year}_z{zoom}.tif"
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out,
        "w",
        driver="GTiff",
        width=mosaic.shape[2],
        height=mosaic.shape[1],
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(left, top, size_m / _TILE_PX, size_m / _TILE_PX),
        compress="deflate",
        photometric="RGB",
    ) as dst:
        dst.write(mosaic)
    return [out]
