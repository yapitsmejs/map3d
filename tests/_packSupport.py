"""Helpers shared by the pack/prepare tests (``from _packSupport import ...``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import numpy as np
import rasterio

from map3d.prepare.http import HttpSession


def mockSession(handler, **kw: Any) -> HttpSession:
    """An ``HttpSession`` over an ``httpx.MockTransport`` handler, with no retry sleeps."""
    kw.setdefault("sleep", lambda s: None)
    return HttpSession(transport=httpx.MockTransport(handler), **kw)


def writeBands(path: Path, data: np.ndarray, crs: str, transform, nodata=None) -> Path:
    """Write a ``(bands, rows, cols)`` float32 GeoTIFF."""
    count, height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data.astype(np.float32))
    return path
