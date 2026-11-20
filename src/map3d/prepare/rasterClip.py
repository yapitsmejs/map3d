"""Windowed raster clipping to a lon/lat box, in the source's own CRS.

Works on local paths and on http(s) URLs, the latter through an
:class:`~map3d.prepare.http.HttpSession` range-read opener. Only the needed
blocks are fetched, so multi-GB remote tiles are fine. No reprojection: pack
rasters keep their native grid, and the reader samples them in any CRS.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

from map3d.prepare.http import HttpSession

__all__ = ["clipRaster"]

_TILE = 256
_SNAP = 1e-6  # pixels: absorb float noise in bounds -> window conversion


def clipRaster(
    source: str,
    bounds_deg: tuple[float, float, float, float],
    outPath: str | os.PathLike[str],
    session: HttpSession | None = None,
    band: int = 1,
    decimation: int = 1,
    marginPixels: int = 2,
    nodata: float | None = None,
    resampling: Resampling = Resampling.nearest,
) -> Path | None:
    """Clip one band of a raster to a lon/lat box and write a tiled GeoTIFF.

    Args:
        source: Local path or http(s) URL.
        bounds_deg: ``(minLon, minLat, maxLon, maxLat)``.
        outPath: Output GeoTIFF path.
        session: HTTP session (required for URLs).
        band: 1-based source band.
        decimation: Integer downsampling factor (GDAL uses overviews when present).
        marginPixels: Source pixels of padding around the box.
        nodata: Nodata value to declare (default: the source's).
        resampling: Resampling used when decimating.

    Returns:
        The output path, or ``None`` if the box does not overlap the raster.

    Raises:
        ValueError: If ``source`` is a URL and no session is given.
    """
    isRemote = source.startswith(("http://", "https://"))
    if isRemote and session is None:
        raise ValueError("a session (HttpSession) is required to clip a remote raster")
    openKw: dict[str, Any] = {"opener": session.rasterOpener} if isRemote and session else {}
    with rasterio.open(source, **openKw) as src:
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *bounds_deg)
        window = _outwardWindow(from_bounds(left, bottom, right, top, src.transform), marginPixels)
        window = _intersect(window, Window(0, 0, src.width, src.height))
        if window is None:
            return None
        if decimation > 1:
            window = _alignToDecimation(window, decimation, src.width, src.height)
        outW = math.ceil(window.width / decimation)
        outH = math.ceil(window.height / decimation)
        data = src.read(band, window=window, out_shape=(outH, outW), resampling=resampling)
        transform = src.window_transform(window) @ rasterio.Affine.scale(
            window.width / outW, window.height / outH
        )
        profile: dict[str, Any] = {
            "driver": "GTiff",
            "width": outW,
            "height": outH,
            "count": 1,
            "dtype": data.dtype.name,
            "crs": src.crs,
            "transform": transform,
            "nodata": nodata if nodata is not None else src.nodatavals[band - 1],
            "compress": "deflate",
        }
    if outW >= _TILE and outH >= _TILE:
        profile.update(tiled=True, blockxsize=_TILE, blockysize=_TILE)
    out = Path(outPath)
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(data, 1)
    return out


def _outwardWindow(window: Window, margin: int) -> Window:
    col0 = math.floor(window.col_off + _SNAP) - margin
    row0 = math.floor(window.row_off + _SNAP) - margin
    col1 = math.ceil(window.col_off + window.width - _SNAP) + margin
    row1 = math.ceil(window.row_off + window.height - _SNAP) + margin
    return Window(col0, row0, col1 - col0, row1 - row0)


def _alignToDecimation(window: Window, d: int, width: int, height: int) -> Window:
    """Snap a window to whole ``d x d`` blocks from the raster origin, staying inside it."""
    col0 = (int(window.col_off) // d) * d
    row0 = (int(window.row_off) // d) * d
    col1 = min(-(-int(window.col_off + window.width) // d) * d, (width // d) * d)
    row1 = min(-(-int(window.row_off + window.height) // d) * d, (height // d) * d)
    if col1 <= col0 or row1 <= row0:
        return window  # raster smaller than one block: keep the unaligned window
    return Window(col0, row0, col1 - col0, row1 - row0)


def _intersect(a: Window, b: Window) -> Window | None:
    col0 = max(a.col_off, b.col_off)
    row0 = max(a.row_off, b.row_off)
    col1 = min(a.col_off + a.width, b.col_off + b.width)
    row1 = min(a.row_off + a.height, b.row_off + b.height)
    if col1 <= col0 or row1 <= row0:
        return None
    return Window(col0, row0, col1 - col0, row1 - row0)
