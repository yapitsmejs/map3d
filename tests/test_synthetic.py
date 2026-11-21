"""Tests for ``map3d.pack.synthetic`` (analytic test packs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import shapely

from map3d.pack.reader import Pack
from map3d.pack.synthetic import makeSyntheticPack


@pytest.fixture(scope="module")
def scene(tmp_path_factory: pytest.TempPathFactory):
    return makeSyntheticPack(tmp_path_factory.mktemp("syn") / "syn.pack", terrain="hill")


def testSyntheticPack_opensAndCoversCenter(scene):
    pack = Pack.open(scene.packDir)
    assert pack.verify() == []
    warnings = pack.checkCoverage(scene.sceneBounds_deg(halfSize_m=500.0))
    roles = {w.split("'")[1] for w in warnings}
    assert roles == {"buildingHeightRaster", "dsmHeight", "imagery"}


def testSyntheticPack_groundMatchesAnalyticTerrain(scene):
    pack = Pack.open(scene.packDir)
    window = pack.readRaster("syntheticGround", scene.sceneBounds_deg(halfSize_m=600.0))
    rng = np.random.default_rng(3)
    e_m = rng.uniform(-500.0, 500.0, 200)
    n_m = rng.uniform(-500.0, 500.0, 200)
    lat_deg, lon_deg, _ = scene.frame.columnPointAtHeight(e_m, n_m, 0.0)
    # Bilinear on a 5 m grid of a smooth hill: small interpolation error only.
    np.testing.assert_allclose(
        window.sample(lat_deg, lon_deg), scene.terrainHeight_m(e_m, n_m), atol=0.05
    )


def testSyntheticPack_planeTerrainIsLinear(tmp_path: Path):
    scene = makeSyntheticPack(tmp_path / "plane.pack", terrain="plane", halfSize_m=300.0)
    window = Pack.open(scene.packDir).readRaster("syntheticGround", scene.sceneBounds_deg(200.0))
    e_m = np.array([-150.0, 0.0, 123.4])
    n_m = np.array([90.0, 0.0, -170.0])
    lat_deg, lon_deg, _ = scene.frame.columnPointAtHeight(e_m, n_m, 0.0)
    np.testing.assert_allclose(
        window.sample(lat_deg, lon_deg), scene.terrainHeight_m(e_m, n_m), atol=1e-3
    )


def testSyntheticPack_vectorsMatchDefinitions(scene):
    pack = Pack.open(scene.packDir)
    bounds = scene.sceneBounds_deg(halfSize_m=900.0)
    buildings = pack.readVectors("syntheticBuildings", bounds)
    assert len(buildings) == len(scene.buildings)
    assert buildings.table.column("height_m").to_pylist() == [b.height_m for b in scene.buildings]
    transport = pack.readVectors("syntheticTransport", bounds)
    assert transport.table.column("isBridge").to_pylist() == [True, False]
    assert all(isinstance(g, shapely.LineString) for g in transport.geometries)
    trees = pack.readVectors("syntheticTrees", bounds)
    assert trees.table.column("height_m").to_pylist() == [8.0]


def testSyntheticPack_buildingFootprintIsAtEnuBox(scene):
    pack = Pack.open(scene.packDir)
    building = scene.buildings[0]
    geom = pack.readVectors("syntheticBuildings", scene.sceneBounds_deg(900.0)).geometries[0]
    lon, lat = np.asarray(geom.exterior.coords).T
    e_m, n_m, _ = scene.frame.geodeticToEnu(lat, lon, 0.0)
    np.testing.assert_allclose(
        [e_m.min(), n_m.min(), e_m.max(), n_m.max()],
        [building.e0_m, building.n0_m, building.e1_m, building.n1_m],
        atol=1e-6,
    )


def testSyntheticPack_rastersHaveExpectedValues(scene):
    pack = Pack.open(scene.packDir)
    bounds = scene.sceneBounds_deg(halfSize_m=900.0)
    assert float(
        pack.geoid().undulation_m(*scene.frame.enuToGeodetic(0.0, 0.0, 0.0)[:2])
    ) == pytest.approx(scene.geoidUndulation_m)
    lat, lon, _ = scene.frame.columnPointAtHeight(
        np.array([0.0, 0.0, -350.0]), np.array([0.0, 450.0, -350.0]), 0.0
    )
    landCover = pack.readRaster("syntheticLandCover", bounds).sample(lat, lon, method="nearest")
    np.testing.assert_array_equal(landCover, [30, 80, 10])
    canopy = pack.readRaster("syntheticCanopy", bounds).sample(lat, lon, method="nearest")
    np.testing.assert_array_equal(canopy, [0.0, 0.0, 15.0])


def testSyntheticPack_rejectsUnknownTerrain(tmp_path: Path):
    with pytest.raises(ValueError, match="terrain"):
        makeSyntheticPack(tmp_path / "x.pack", terrain="volcano")  # type: ignore[arg-type]


def testSyntheticPack_deckFootprintUnderBridge(scene):
    pack = Pack.open(scene.packDir)
    elevated = pack.readVectors("syntheticElevatedStructures", scene.sceneBounds_deg(900.0))
    assert elevated.table.column("structureType").to_pylist() == ["Flyover"]
    lon, lat = np.asarray(elevated.geometries[0].exterior.coords).T
    e_m, n_m, _ = scene.frame.geodeticToEnu(lat, lon, 0.0)
    np.testing.assert_allclose(
        [e_m.min(), n_m.min(), e_m.max(), n_m.max()], [-400.0, 194.0, 400.0, 206.0], atol=1e-6
    )
