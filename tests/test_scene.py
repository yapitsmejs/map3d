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


def testBuildScene_producesAllLayers(built: Scene):
    deckLayers = {
        "deckTopHeight_datum_m",
        "deckTopHeight_ENU_m",
        "deckBottomHeight_datum_m",
        "deckBottomHeight_ENU_m",
    }
    assert set(built.layers) == set(LAYER_INFO)
    assert deckLayers <= set(built.layers)
    for name, arr in built.layers.items():
        assert arr.shape == ((5, 200, 200) if name in deckLayers else (200, 200)), name
    assert built.layers["groundHeight_datum_m"].dtype == np.float32
    assert built.layers["featureId"].dtype == np.int64
    assert built.layers["heightSource"].dtype == np.uint8
    assert built.layers["landClass"].dtype == np.uint8


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
    assert list(built.features["sourceFeatureId"]) == ["synB0", "synB1", "synB2", "synR0"]
    assert list(built.features["kind"]) == ["building"] * 3 + ["deck"]
    np.testing.assert_array_equal(built.features["featureId"], [1, 2, 3, 4])
    present = set(np.unique(built.layers["featureId"])) - {0}
    assert present == {1, 2, 3, 4}


def testCanopy_raisesSurfaceWithCanopyProvenance(built: Scene):
    r, c = _rc(built, -350.0, -350.0)
    ground = built.layers["groundHeight_datum_m"][r, c]
    assert built.layers["canopyHeight_m"][r, c] == pytest.approx(15.0)
    assert built.layers["surfaceHeight_datum_m"][r, c] == pytest.approx(ground + 15.0, abs=1e-4)
    assert built.layers["heightSource"][r, c] == HeightSource.CANOPY
    assert built.layers["landClass"][r, c] == 10


def testDeck_setsSurfaceClassFeatureAndLevels(built: Scene, plane: SyntheticScene):
    r, c = _rc(built, 0.0, 200.0)
    top = built.layers["deckTopHeight_datum_m"][0, r, c]
    e, _ = built.geometry.nodesEnu()
    assert top == pytest.approx(float(plane.terrainHeight_m(e[r, c], 200.0)) + 7.0, abs=0.05)
    assert built.layers["deckBottomHeight_datum_m"][0, r, c] == pytest.approx(top - 1.5, abs=1e-4)
    assert np.isnan(built.layers["deckTopHeight_datum_m"][1, r, c])
    assert built.layers["surfaceHeight_datum_m"][r, c] == pytest.approx(top, abs=1e-4)
    assert built.layers["heightSource"][r, c] == HeightSource.DECK_ESTIMATED
    assert built.layers["featureId"][r, c] == 4
    assert built.layers["landClass"][r, c] == 220
    assert built.layers["nativeResolution_m"][r, c] == 0.0
    _, _, uRef = built.geometry.frame.columnPointAtHeight(
        built.geometry.nodesEnu()[0][r, c], built.geometry.nodesEnu()[1][r, c], float(top) + 8.0
    )
    assert built.layers["deckTopHeight_ENU_m"][0, r, c] == pytest.approx(float(uRef), abs=1e-3)


def testLandClass_waterRoadAndBuildingPrecedence(built: Scene):
    assert built.layers["landClass"][_rc(built, 0.0, 380.0)] == 80
    assert built.layers["landClass"][_rc(built, 100.0, -390.0)] == 30
    assert built.layers["landClass"][_rc(built, -160.0, -100.0)] == 200


def testZeroDeckLevels_keepsDeckInSurfaceAndWarns(plane: SyntheticScene):
    scene = buildScene(_request(plane, maxDeckLevels=0), plane.packDir)
    assert scene.layers["deckTopHeight_datum_m"].shape == (0, 200, 200)
    r, c = _rc(scene, 0.0, 200.0)
    assert scene.layers["heightSource"][r, c] == HeightSource.DECK_ESTIMATED
    assert any("maxDeckLevels=0" in w for w in scene.warnings)


def testMetadata_hasClassTableAndDeckModel(built: Scene):
    assert built.metadata["landClassCodes"]["220"] == "deck"
    assert built.metadata["deckModel"]["roadLevelHeight_m"] == 7.0
    sources = {s["sourceId"] for s in built.metadata["sources"]}
    assert {
        "syntheticCanopy",
        "syntheticTrees",
        "syntheticLandCover",
        "syntheticTransport",
    } <= sources


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


def testParallelWorkers_identicalToSequential(plane: SyntheticScene, built: Scene):
    import warnings

    before = list(warnings.filters)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no leaked warnings from worker threads
        parallel = buildScene(
            _request(plane, heading_deg=20.0), plane.packDir, chunkSize=64, workers=8
        )
    assert warnings.filters == before  # global warning filters left untouched
    for name, arr in built.layers.items():
        np.testing.assert_array_equal(parallel.layers[name], arr, err_msg=name)


def testMemmapStorage_identicalAndCleanedUp(plane: SyntheticScene, built: Scene, tmp_path: Path):
    work = tmp_path / "work"
    with buildScene(_request(plane, heading_deg=20.0), plane.packDir, workDir=work) as scene:
        assert scene.metadata["storage"] == "memmap"
        assert isinstance(scene.layers["surfaceHeight_ENU_m"], np.memmap)
        assert (work / "surfaceHeight_ENU_m.npy").is_file()
        for name, arr in built.layers.items():
            np.testing.assert_array_equal(scene.layers[name], arr, err_msg=name)
    assert (work / "surfaceHeight_ENU_m.npy").is_file()  # caller-owned workDir is kept


def testMemmapStorage_autoWhenOverBudgetUsesOwnedTempDir(plane: SyntheticScene):
    scene = buildScene(_request(plane, swath_m=200.0), plane.packDir, memoryBudget_GB=1e-6)
    assert scene.metadata["storage"] == "memmap"
    storage = scene.storageDir
    assert storage is not None and storage.is_dir()
    scene.close()
    assert not storage.exists()
    scene.close()  # idempotent


def testMemoryStorage_byDefaultForSmallScenes(built: Scene):
    assert built.metadata["storage"] == "memory"
    assert built.storageDir is None
    built.close()  # no-op for in-memory scenes
