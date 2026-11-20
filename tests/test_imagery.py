"""Tests for ``map3d.prepare.sources.imagery`` (Sentinel-2 cloudless WMTS mosaics)."""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import httpx
import numpy as np
import pytest
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile

from map3d.prepare.http import HttpSession
from map3d.prepare.sources import imagery

BOUNDS = (103.85, 1.275, 103.87, 1.29)


def _jpeg(value: int) -> bytes:
    data = np.full((3, 256, 256), value, dtype=np.uint8)
    data[1] = 255 - value
    with warnings.catch_warnings(), MemoryFile() as mem:
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with mem.open(driver="JPEG", width=256, height=256, count=3, dtype="uint8") as dst:
            dst.write(data)
        return mem.read()


def testTileRange_coversBoundsAtZoom():
    x0, y0, x1, y1 = imagery.tileRange(BOUNDS, 14)
    n = 2**14
    assert x0 == int((103.85 + 180) / 360 * n) and x1 == int((103.87 + 180) / 360 * n)
    lat = math.radians(1.29)
    assert y0 == int((1 - math.asinh(math.tan(lat)) / math.pi) / 2 * n)
    assert y1 >= y0


def testTileUrl():
    assert imagery.tileUrl(2024, 14, 12920, 8133).endswith(
        "/s2cloudless-2024_3857/default/g/14/8133/12920.jpg"
    )


def testFetch_mosaicsTilesIntoGeoreferencedRgb(tmp_path: Path):
    x0, y0, x1, y1 = imagery.tileRange(BOUNDS, 14)
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        x = int(request.url.path.split("/")[-1].removesuffix(".jpg"))
        return httpx.Response(200, content=_jpeg(40 + 20 * (x - x0)))

    with HttpSession(transport=httpx.MockTransport(handler), sleep=lambda s: None) as session:
        out = imagery.fetchSentinel2Cloudless(
            session, BOUNDS, tmp_path / "work", year=2024, zoom=14
        )
        imagery.fetchSentinel2Cloudless(session, BOUNDS, tmp_path / "work", year=2024, zoom=14)
    assert len(requested) == (x1 - x0 + 1) * (y1 - y0 + 1)  # second call served from cache
    (path,) = out
    with rasterio.open(path) as src:
        assert src.count == 3 and src.dtypes[0] == "uint8"
        assert src.crs.to_epsg() == 3857
        assert src.width == 256 * (x1 - x0 + 1) and src.height == 256 * (y1 - y0 + 1)
        r = src.read(1)
        assert abs(int(r[10, 10]) - 40) <= 3  # JPEG is lossy
        left, _, _, top = src.bounds
        half = math.pi * 6378137.0
        assert left == pytest.approx(x0 / 2**14 * 2 * half - half)
        assert top == pytest.approx(half - y0 / 2**14 * 2 * half)


def testFetch_missingTilesAreLeftBlack(tmp_path: Path):
    with HttpSession(
        transport=httpx.MockTransport(lambda r: httpx.Response(404)), sleep=lambda s: None
    ) as session:
        (path,) = imagery.fetchSentinel2Cloudless(session, BOUNDS, tmp_path / "work", zoom=14)
    with rasterio.open(path) as src:
        assert src.read().max() == 0
