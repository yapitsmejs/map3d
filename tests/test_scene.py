"""Tests for ``map3d.scene`` (buildScene / Scene)."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from map3d.layers.provenance import HeightSource
from map3d.pack.manifest import Manifest
from map3d.pack.reader import Pack
from map3d.pack.synthetic import SyntheticScene, makeSyntheticPack
from map3d.request import SceneRequest
from map3d.scene import LAYER_INFO, Scene, buildScene


@pytest.fixture(scope="module")
def plane(tmp_path_factory: pytest.TempPathFactory) -> SyntheticScene:
    return makeSyntheticPack(tmp_path_factory.mktemp("s") / "plane.pack", terrain="plane")


def _request(
    scene: SyntheticScene, heading_deg: float = 0.0, swath_m: float = 800.0, s: float = 4.0, **kw
) -> SceneRequest:
    h = math.radians(heading_deg)
    axes = [[math.cos(h), -math.sin(h), 0.0], [math.sin(h), math.cos(h), 0.0], [0.0, 0.0, 1.0]]
    return SceneRequest(
        (scene.frame.lat0_deg, scene.frame.lon0_deg), s, axes, swath_m, swath_m, **kw
    )


@pytest.fixture(scope="module")
def built(plane: SyntheticScene) -> Scene:
    return buildScene(_request(plane, heading_deg=20.0), plane.packDir)


def _rc(scene: Scene, e_m: float, n_m: float) -> tuple[int, int]:
    e, n = scene.geometry.nodesEnu()
    flat = int(np.argmin((e - e_m) ** 2 + (n - n_m) ** 2))
    return int(np.unravel_index(flat, e.shape)[0]), int(np.unravel_index(flat, e.shape)[1])


def testBuildScene_producesAllM3Layers(built: Scene):
    assert set(built.layers) == {
        "groundHeight_datum_m",
        "groundHeight_ENU_m",
        "surfaceHeight_datum_m",
        "surfaceHeight_ENU_m",
        "buildingHeight_m",
        "featureId",
        "heightSource",
        "nativeResolution_m",
    }
    for name, arr in built.layers.items():
        assert arr.shape == (200, 200)
        assert name in LAYER_INFO
    assert built.layers["groundHeight_datum_m"].dtype == np.float32
    assert built.layers["featureId"].dtype == np.int64
    assert built.layers["heightSource"].dtype == np.uint8


def testGroundDatum_msl_matchesTerrain(built: Scene, plane: SyntheticScene):
    r, c = _rc(built, 0.0, 0.0)
    e, n = built.geometry.nodesEnu()
    # Cell-max of a plane is within half a cell diagonal * slope of the center value.
    center = float(plane.terrainHeight_m(e[r, c], n[r, c]))
    assert built.layers["groundHeight_datum_m"][r, c] == pytest.approx(center, abs=0.07)


def testEnuLayer_equalsExactColumnSolve(built: Scene):
    h_msl = built.layers["groundHeight_datum_m"].astype(np.float64)
    e, n = built.geometry.nodesEnu()
    _, _, uRef = built.geometry.frame.columnPointAtHeight(e, n, h_msl + 8.0)  # synthetic N = 8 m
    np.testing.assert_allclose(
        built.layers["groundHeight_ENU_m"], uRef, atol=1e-4
    )  # float32 storage


def testEllipsoidDatum_addsUndulation(plane: SyntheticScene, built: Scene):
    ell = buildScene(_request(plane, heading_deg=20.0, verticalDatum="ellipsoid"), plane.packDir)
    np.testing.assert_allclose(
        ell.layers["groundHeight_datum_m"], built.layers["groundHeight_datum_m"] + 8.0, atol=1e-4
    )
    np.testing.assert_array_equal(
        ell.layers["groundHeight_ENU_m"], built.layers["groundHeight_ENU_m"]
    )


def testBuildings_raiseSurfaceAndRecordProvenance(built: Scene):
    r, c = _rc(built, -160.0, -100.0)  # inside B0 (30 m, explicit height)
    assert built.layers["featureId"][r, c] == 1
    assert built.layers["surfaceHeight_datum_m"][r, c] == pytest.approx(36.5, abs=1e-3)
    ground = built.layers["groundHeight_datum_m"][r, c]
    assert built.layers["buildingHeight_m"][r, c] == pytest.approx(36.5 - ground, abs=1e-3)
    assert built.layers["heightSource"][r, c] == HeightSource.EXPLICIT_HEIGHT
    assert built.layers["nativeResolution_m"][r, c] == 0.0


def testOpenGround_surfaceEqualsGround(built: Scene):
    r, c = _rc(built, 0.0, 0.0)
    assert built.layers["featureId"][r, c] == 0
    assert np.isnan(built.layers["buildingHeight_m"][r, c])
    assert built.layers["surfaceHeight_datum_m"][r, c] == built.layers["groundHeight_datum_m"][r, c]
    assert built.layers["heightSource"][r, c] == HeightSource.GROUND
    assert built.layers["nativeResolution_m"][r, c] == 5.0


def testFeatures_tableMatchesIds(built: Scene):
    assert list(built.features["sourceFeatureId"]) == ["synB0", "synB1", "synB2"]
    present = set(np.unique(built.layers["featureId"])) - {0}
    assert present == {1, 2, 3}


def testChunkSize_doesNotChangeResult(plane: SyntheticScene, built: Scene):
    other = buildScene(_request(plane, heading_deg=20.0), plane.packDir, chunkSize=37)
    for name, arr in built.layers.items():
        np.testing.assert_array_equal(other.layers[name], arr, err_msg=name)


def testMetadata_describesFrameDatumSourcesAndWarnings(built: Scene):
    meta = built.metadata
    assert meta["frame"]["nRows"] == 200 and meta["frame"]["nCols"] == 200
    assert meta["frame"]["arrayOrder"].startswith("[row, col]")
    assert meta["frame"]["nodeConvention"] == "cell-centered"
    assert len(meta["frame"]["affine"]) == 6
    assert meta["verticalDatum"]["name"] == "EGM2008"
    assert meta["verticalDatum"]["geoidUndulationAtCenter_m"] == pytest.approx(8.0)
    assert {s["sourceId"] for s in meta["sources"]} >= {
        "syntheticGround",
        "syntheticGeoid",
        "syntheticBuildings",
    }
    assert meta["heightSourceCodes"]["GROUND"] == 1
    assert any("finer than" in w and "syntheticGround" in w for w in meta["warnings"])
    assert any("imagery" in w for w in meta["warnings"])
    assert meta["request"]["gridSpacing_m"] == 4.0
    assert built.warnings is meta["warnings"]


def testBuildScene_acceptsOpenedPack(plane: SyntheticScene):
    scene = buildScene(_request(plane, swath_m=40.0), Pack.open(plane.packDir))
    assert scene.layers["featureId"].shape == (10, 10)


def testBuildScene_outsidePackRaises(plane: SyntheticScene):
    with pytest.raises(ValueError, match="outside the pack"):
        buildScene(_request(plane, swath_m=4000.0), plane.packDir)


def testBuildScene_withoutBuildingsSource(plane: SyntheticScene, tmp_path: Path):
    pack = tmp_path / "nob.pack"
    shutil.copytree(plane.packDir, pack)
    manifest = Manifest.load(pack)
    del manifest.sources["syntheticBuildings"]
    manifest.save(pack)
    scene = buildScene(_request(plane, swath_m=400.0), pack)
    assert scene.layers["featureId"].max() == 0
    assert np.isnan(scene.layers["buildingHeight_m"]).all()
    assert any("buildings" in w for w in scene.warnings)
