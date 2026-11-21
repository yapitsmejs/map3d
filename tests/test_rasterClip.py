"""Tests for ``map3d.prepare.rasterClip``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from _packSupport import mockSession, writeBands
from _support import rangeServer
from rasterio.transform import from_bounds

from map3d.prepare.rasterClip import clipRaster

BOUNDS = (103.0, 1.0, 104.0, 2.0)


def _grid(shape=(100, 100)) -> np.ndarray:
    return np.arange(shape[0] * shape[1], dtype=np.float32).reshape(shape)


def testClip_extractsWindowWithMargin(tmp_path: Path, writeRaster):
    src = writeRaster(tmp_path / "s.tif", BOUNDS, values=_grid(), shape=(100, 100))
    out = clipRaster(str(src), (103.2, 1.2, 103.4, 1.3), tmp_path / "o.tif", marginPixels=1)
    assert out == tmp_path / "o.tif"
    with rasterio.open(out) as clipped:
        # Pixel size 0.01 deg: lon 103.2..103.4 -> cols 20..40, lat 1.3..1.2 -> rows 70..80; +1 margin.
        np.testing.assert_array_equal(clipped.read(1), _grid()[69:81, 19:41])
        assert clipped.transform.c == pytest.approx(103.19)
        assert clipped.transform.f == pytest.approx(1.31)
        assert clipped.crs.to_epsg() == 4326


def testClip_clampsToRasterExtent(tmp_path: Path, writeRaster):
    src = writeRaster(tmp_path / "s.tif", BOUNDS, values=_grid(), shape=(100, 100))
    out = clipRaster(str(src), (102.5, 1.95, 103.05, 2.5), tmp_path / "o.tif")
    with rasterio.open(out) as clipped:
        np.testing.assert_array_equal(clipped.read(1), _grid()[0:7, 0:7])  # 5 px + 2 margin


def testClip_disjointReturnsNone(tmp_path: Path, writeRaster):
    src = writeRaster(tmp_path / "s.tif", BOUNDS, values=_grid(), shape=(100, 100))
    assert clipRaster(str(src), (110.0, 1.0, 111.0, 2.0), tmp_path / "o.tif") is None
    assert not (tmp_path / "o.tif").exists()


def testClip_decimatesAndScalesTransform(tmp_path: Path, writeRaster):
    src = writeRaster(tmp_path / "s.tif", BOUNDS, values=_grid(), shape=(100, 100))
    out = clipRaster(
        str(src), (103.0, 1.0, 104.0, 2.0), tmp_path / "o.tif", decimation=4, marginPixels=0
    )
    with rasterio.open(out) as clipped:
        assert clipped.shape == (25, 25)
        assert clipped.res == pytest.approx((0.04, 0.04))
        assert clipped.read(1)[0, 0] == _grid()[2, 2]  # nearest = centre of each 4x4 block


def testClip_selectsBandAndOverridesNodata(tmp_path: Path):
    data = np.stack([np.full((10, 10), v, np.float32) for v in (1.0, 2.0, 3.0)])
    path = writeBands(tmp_path / "rgb.tif", data, "EPSG:4326", from_bounds(*BOUNDS, 10, 10))
    out = clipRaster(str(path), BOUNDS, tmp_path / "o.tif", band=2, nodata=-99.0)
    with rasterio.open(out) as clipped:
        assert clipped.count == 1
        assert clipped.nodata == -99.0
        assert float(clipped.read(1).mean()) == 2.0


def testClip_projectedSource(tmp_path: Path, writeRaster):
    src = writeRaster(
        tmp_path / "u.tif",
        (360000.0, 140000.0, 380000.0, 160000.0),
        values=_grid((20, 20)),
        crs="EPSG:32648",
        shape=(20, 20),
    )
    out = clipRaster(str(src), (103.8, 1.3, 103.85, 1.35), tmp_path / "o.tif", marginPixels=0)
    with rasterio.open(out) as clipped:
        assert clipped.crs.to_epsg() == 32648
        assert 0 < clipped.width < 20 and 0 < clipped.height < 20


def testClip_remoteThroughSessionOpener(tmp_path: Path, writeRaster):
    src = writeRaster(tmp_path / "s.tif", BOUNDS, values=_grid(), shape=(100, 100))
    with mockSession(rangeServer({"https://x/s.tif": src.read_bytes()})) as session:
        out = clipRaster(
            "https://x/s.tif",
            (103.2, 1.2, 103.4, 1.3),
            tmp_path / "o.tif",
            session=session,
            marginPixels=0,
        )
    with rasterio.open(out) as clipped:
        np.testing.assert_array_equal(clipped.read(1), _grid()[70:80, 20:40])


def testClip_remoteRequiresSession(tmp_path: Path):
    with pytest.raises(ValueError, match="session"):
        clipRaster("https://x/s.tif", BOUNDS, tmp_path / "o.tif")


def testClip_writesTiledCompressedForLargeOutputs(tmp_path: Path, writeRaster):
    src = writeRaster(tmp_path / "s.tif", BOUNDS, values=_grid((600, 600)), shape=(600, 600))
    out = clipRaster(str(src), BOUNDS, tmp_path / "o.tif", marginPixels=0)
    with rasterio.open(out) as clipped:
        assert clipped.profile["tiled"]
        assert clipped.compression.value.lower() == "deflate"


def testClip_decimationAlignsToWholeBlocks(tmp_path: Path, writeRaster):
    src = writeRaster(tmp_path / "s.tif", BOUNDS, values=_grid(), shape=(100, 100))
    # Cols 21..43 (+0 margin) -> snapped to whole 4-px blocks 20..44.
    out = clipRaster(
        str(src), (103.21, 1.21, 103.43, 1.33), tmp_path / "o.tif", decimation=4, marginPixels=0
    )
    with rasterio.open(out) as clipped:
        assert clipped.res == pytest.approx((0.04, 0.04))
        assert clipped.transform.c == pytest.approx(103.20)
        assert clipped.width == 6


def testClip_decimationOnTinyRasterKeepsWindow(tmp_path: Path, writeRaster):
    src = writeRaster(tmp_path / "s.tif", BOUNDS, values=_grid((3, 3)), shape=(3, 3))
    out = clipRaster(str(src), BOUNDS, tmp_path / "o.tif", decimation=8, marginPixels=0)
    with rasterio.open(out) as clipped:
        assert clipped.shape == (1, 1)
