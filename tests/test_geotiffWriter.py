"""Tests for ``map3d.io.geotiffWriter``."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pyproj
import pytest
import rasterio

from map3d.io.geotiffWriter import writeGeoTiff
from map3d.pack.synthetic import makeSyntheticPack
from map3d.request import SceneRequest
from map3d.scene import Scene, buildScene


@pytest.fixture(scope="module")
def scene(tmp_path_factory: pytest.TempPathFactory) -> Scene:
    syn = makeSyntheticPack(tmp_path_factory.mktemp("g") / "p.pack")
    h = math.radians(30.0)
    axes = [[math.cos(h), -math.sin(h), 0.0], [math.sin(h), math.cos(h), 0.0], [0.0, 0.0, 1.0]]
    req = SceneRequest((syn.frame.lat0_deg, syn.frame.lon0_deg), 5.0, axes, 1280.0, 1300.0)
    return buildScene(req, syn.packDir)


def testWriteGeoTiff_oneFilePerLayerWithSameValues(tmp_path: Path, scene: Scene):
    paths = writeGeoTiff(scene, tmp_path / "tif")
    assert {p.stem for p in paths} == set(scene.layers)
    for path in paths:
        with rasterio.open(path) as src:
            expected = scene.layers[path.stem]
            data = src.read() if expected.ndim == 3 else src.read(1)
            assert src.count == (expected.shape[0] if expected.ndim == 3 else 1)
            assert data.dtype == expected.dtype
            np.testing.assert_array_equal(data, expected)


def testWriteGeoTiff_georeferencingIsExact(tmp_path: Path, scene: Scene):
    path = next(p for p in writeGeoTiff(scene, tmp_path / "tif") if p.stem == "surfaceHeight_ENU_m")
    with rasterio.open(path) as src:
        assert tuple(src.transform)[:6] == pytest.approx(tuple(scene.geometry.affine())[:6])
        crs = pyproj.CRS.from_wkt(src.crs.to_wkt())
        assert crs.equals(scene.geometry.orthoCrs())
        tags = src.tags()
        assert tags["units"] == "m"
        assert "Up coordinate" in tags["description"]
        assert "map3d synthetic" in tags["attribution"]
        # Pixel (row 3, col 5) center maps back to that node's ENU position.
        x, y = src.transform @ (5 + 0.5, 3 + 0.5)
    toGeo = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = toGeo.transform(x, y)
    e, n, _ = scene.geometry.frame.geodeticToEnu(lat, lon, 0.0)
    eNode, nNode = scene.geometry.nodesEnu(slice(3, 4), slice(5, 6))
    assert float(e) == pytest.approx(float(eNode[0, 0]), abs=1e-6)
    assert float(n) == pytest.approx(float(nNode[0, 0]), abs=1e-6)


def testWriteGeoTiff_floatNodataIsNan(tmp_path: Path, scene: Scene):
    path = next(p for p in writeGeoTiff(scene, tmp_path / "tif") if p.stem == "buildingHeight_m")
    with rasterio.open(path) as src:
        assert math.isnan(src.nodata)


def testWriteGeoTiff_stacksThreeDimensionalLayersAsBands(tmp_path: Path, scene: Scene):
    stacked = np.stack([scene.layers["groundHeight_datum_m"]] * 3)
    scene.layers["testStack_m"] = stacked
    try:
        path = next(p for p in writeGeoTiff(scene, tmp_path / "tif") if p.stem == "testStack_m")
        with rasterio.open(path) as src:
            assert src.count == 3
            np.testing.assert_array_equal(src.read(), stacked)
    finally:
        del scene.layers["testStack_m"]


def testWriteGeoTiff_skipsZeroBandLayers(tmp_path: Path):
    syn = makeSyntheticPack(tmp_path / "p.pack")
    req = SceneRequest(
        (syn.frame.lat0_deg, syn.frame.lon0_deg), 20.0, np.eye(3), 400.0, 400.0, maxDeckLevels=0
    )
    paths = writeGeoTiff(buildScene(req, syn.packDir), tmp_path / "tif")
    assert not any(p.stem.startswith("deck") for p in paths)


def testWriteGeoTiff_bandedWritesMatchForTallRasters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from map3d.io import geotiffWriter

    monkeypatch.setattr(geotiffWriter, "_BAND_ROWS", 7)  # force several bands, last one partial
    syn = makeSyntheticPack(tmp_path / "p.pack")
    req = SceneRequest((syn.frame.lat0_deg, syn.frame.lon0_deg), 20.0, np.eye(3), 400.0, 600.0)
    scene = buildScene(req, syn.packDir)
    for path in writeGeoTiff(scene, tmp_path / "tif"):
        with rasterio.open(path) as src:
            expected = scene.layers[path.stem]
            data = src.read() if expected.ndim == 3 else src.read(1)
            np.testing.assert_array_equal(data, expected)
