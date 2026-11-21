"""Tests for ``map3d.scene`` (buildScene / Scene)."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import shapely
from _support import headingAxes

from map3d.layers.provenance import HeightSource
from map3d.pack.geoparquet import writeGeoParquet
from map3d.pack.ingest import ingestFiles
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
    axes = headingAxes(heading_deg)
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
    # 30 m above the median ground under B0: the plane at its centre, 7.8 m.
    top = built.layers["surfaceHeight_datum_m"][r, c]
    assert top == pytest.approx(37.8, abs=0.05)
    ground = built.layers["groundHeight_datum_m"][r, c]
    assert built.layers["buildingHeight_m"][r, c] == pytest.approx(top - ground, abs=1e-3)
    assert built.layers["heightSource"][r, c] == HeightSource.EXPLICIT_HEIGHT
    assert built.layers["nativeResolution_m"][r, c] == 0.0


def testOpenGround_surfaceEqualsGround(built: Scene):
    r, c = _rc(built, 0.0, 0.0)
    assert built.layers["featureId"][r, c] == 0
    assert np.isnan(built.layers["buildingHeight_m"][r, c])
    assert built.layers["surfaceHeight_datum_m"][r, c] == built.layers["groundHeight_datum_m"][r, c]
    assert built.layers["heightSource"][r, c] == HeightSource.GROUND
    # effective resolution: the 5 m source smoothed at the default 90 m
    assert built.layers["nativeResolution_m"][r, c] == 90.0


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


def testGround_corridorNodesUsePointSample(tmp_path: Path):
    """Ground at road/rail nodes is the corridor grade (point sample), not the cell max.

    Cell-max pooling biases toward embankment and roadside vegetation; along a
    transport corridor the ground *is* the graded road, so corridor nodes take
    the ground sampled at their own footprint.
    """
    import shapely

    from map3d.pack.geoparquet import writeGeoParquet
    from map3d.pack.ingest import ingestFiles

    hill = makeSyntheticPack(tmp_path / "h1.pack", terrain="hill")

    def lineAt(n0: float) -> None:
        t = np.linspace(0.0, 1.0, 9)
        lat, lon, _ = hill.frame.columnPointAtHeight(-400.0 + t * 800.0, np.full(9, n0), 0.0)
        writeGeoParquet(
            tmp_path / "t.parquet",
            np.array([shapely.LineString(np.column_stack([lon, lat]))], dtype=object),
            {
                "sourceFeatureId": ["r0"],
                "kind": ["road"],
                "isBridge": [False],
                "level": [0],
                "width_m": [20.0],
            },
        )

    withRoad = tmp_path / "road.pack"
    shutil.copytree(hill.packDir, withRoad)
    manifest = Manifest.load(withRoad)
    del manifest.sources["syntheticTransport"], manifest.sources["syntheticElevatedStructures"]
    manifest.save(withRoad)
    lineAt(-200.0)  # an at-grade road across the hill's flank (sloping ground)
    ingestFiles(withRoad, "overtureTransportation", [tmp_path / "t.parquet"])

    bare = tmp_path / "bare.pack"
    shutil.copytree(hill.packDir, bare)
    manifest = Manifest.load(bare)
    del manifest.sources["syntheticTransport"], manifest.sources["syntheticElevatedStructures"]
    manifest.save(bare)

    sceneRoad = buildScene(_request(hill, s=10.0), withRoad)
    sceneBare = buildScene(_request(hill, s=10.0), bare)
    r, c = _rc(sceneRoad, 0.0, -200.0)
    gRoad = sceneRoad.layers["groundHeight_datum_m"][r, c]
    gBare = sceneBare.layers["groundHeight_datum_m"][r, c]
    # the road node's ground is its own footprint's terrain, below the cell max
    lat, lon = _nodeLatLon(sceneRoad, r, c)
    pointGround = float(
        np.asarray(sceneRoad.models.ground.orthometricAt(np.array([[lat]]), np.array([[lon]])))[
            0, 0
        ]
    )
    assert gRoad == pytest.approx(pointGround, abs=1e-6)
    assert gBare - gRoad > 0.1
    # off the corridor, the cell max is unchanged
    r2, c2 = _rc(sceneRoad, 0.0, -150.0)
    assert sceneRoad.layers["groundHeight_datum_m"][r2, c2] == pytest.approx(
        sceneBare.layers["groundHeight_datum_m"][r2, c2], abs=1e-6
    )


def _nodeLatLon(scene: Scene, r: int, c: int) -> tuple[float, float]:
    from map3d.layers.columns import ChunkColumns

    lat, lon = ChunkColumns(scene.geometry, slice(r, r + 1), slice(c, c + 1)).nodeLatLon()
    return float(lat[0, 0]), float(lon[0, 0])


def testCanopy_doesNotWinRoadCorridors(plane: SyntheticScene, tmp_path: Path):
    """Canopy over a ground-level road never tops the surface: the road stays at grade.

    The canopy raster layer itself is untouched (trees still draw beside the
    road as objects); only the surface composition suppresses it.
    """

    from map3d.pack.ingest import ingestFiles
    from map3d.pack.synthetic import _METERS_PER_DEG_LAT, _pixelCentersEnu, _writeGeoTiff

    pack = tmp_path / "c.pack"
    shutil.copytree(plane.packDir, pack)
    bounds = plane.sceneBounds_deg(plane.halfSize_m)
    width = math.ceil((bounds[2] - bounds[0]) * _METERS_PER_DEG_LAT / 5.0)
    height = math.ceil((bounds[3] - bounds[1]) * _METERS_PER_DEG_LAT / 5.0)
    e, n = _pixelCentersEnu(plane.frame, bounds, (height, width))
    # 15 m canopy over the at-grade road at n = -450 and the ground beside it.
    patch = (e >= -350) & (e <= -250) & (n >= -460) & (n <= -425)
    _writeGeoTiff(
        tmp_path / "canopy.tif",
        np.where(patch, 15.0, 0.0).astype(np.float32),
        bounds,
    )
    ingestFiles(pack, "syntheticCanopy", [tmp_path / "canopy.tif"])

    scene = buildScene(_request(plane, swath_m=1000.0), pack)
    r, c = _rc(scene, -300.0, -450.0)  # on the road, under the canopy patch
    assert scene.layers["canopyHeight_m"][r, c] == pytest.approx(15.0)
    assert scene.layers["landClass"][r, c] == 210  # road
    assert scene.layers["surfaceHeight_datum_m"][r, c] == pytest.approx(
        scene.layers["groundHeight_datum_m"][r, c], abs=1e-4
    )
    assert scene.layers["heightSource"][r, c] == HeightSource.GROUND
    r2, c2 = _rc(scene, -300.0, -430.0)  # in the patch, off the road: canopy still wins
    assert scene.layers["heightSource"][r2, c2] == HeightSource.CANOPY
    assert scene.layers["surfaceHeight_datum_m"][r2, c2] == pytest.approx(
        scene.layers["groundHeight_datum_m"][r2, c2] + 15.0, abs=1e-4
    )


def testHeightUncertainty_followsTheWinningSource(built: Scene):
    """The uncertainty layer mirrors each node's height source."""
    from map3d.layers.provenance import HEIGHT_UNCERTAINTY_m, HeightSource

    known = built.layers["heightSource"] != int(HeightSource.NONE)
    uncertainty = built.layers["heightUncertainty_m"]
    src = built.layers["heightSource"]
    assert uncertainty.shape == src.shape
    known = src != int(HeightSource.NONE)
    assert np.isnan(uncertainty[~known]).all()
    r, c = _rc(built, 0.0, 0.0)  # open ground
    assert uncertainty[r, c] == pytest.approx(HEIGHT_UNCERTAINTY_m[HeightSource.GROUND])
    r2, c2 = _rc(built, -350.0, -350.0)  # canopy
    assert uncertainty[r2, c2] == pytest.approx(HEIGHT_UNCERTAINTY_m[HeightSource.CANOPY])
    r3, c3 = _rc(built, 0.0, 200.0)  # deck
    assert uncertainty[r3, c3] == pytest.approx(HEIGHT_UNCERTAINTY_m[HeightSource.DECK_ESTIMATED])
    assert built.metadata["heightUncertainty_m"]["DECK_ESTIMATED"] == pytest.approx(4.0)


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
    # bridges burn as the roads they carry: the corridor wins over the deck
    assert built.layers["landClass"][r, c] == 210
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


def testChunkedParallelBuild_identicalToWholeSequential(plane: SyntheticScene, built: Scene):
    import warnings

    before = list(warnings.filters)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no leaked warnings from worker threads
        parallel = buildScene(
            _request(plane, heading_deg=20.0), plane.packDir, chunkSize=37, workers=8
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


def testGroundSmoothing_defaultRecordedAndOverridable(plane: SyntheticScene, built: Scene):
    from map3d.layers.terrain import DEFAULT_GROUND_SMOOTHING_m

    assert built.metadata["groundSmoothing_m"] == DEFAULT_GROUND_SMOOTHING_m == 90.0
    assert built.models.ground.nativeResolution_m == 90.0  # source 5 m, smoothed to 90 m
    raw = buildScene(_request(plane, swath_m=200.0, s=10.0), plane.packDir, groundSmoothing_m=0.0)
    assert raw.metadata["groundSmoothing_m"] == 0.0
    assert raw.models.ground.nativeResolution_m == 5.0  # the unsmoothed source


def testDeck_classDefersToRoadCorridors(plane: SyntheticScene, tmp_path: Path):
    """A deck over a ground road paints the corridor, not the deck class.

    The terrain under a viaduct shows the road beneath (class road/rail);
    the viaduct's own slab top samples the same class, so a road bridge
    reads as a road on both layers.
    """
    pack = tmp_path / "cross.pack"
    shutil.copytree(plane.packDir, pack)
    manifest = Manifest.load(pack)
    del manifest.sources["syntheticTransport"], manifest.sources["syntheticElevatedStructures"]
    manifest.save(pack)
    t = np.linspace(0.0, 1.0, 9)
    latR, lonR, _ = plane.frame.columnPointAtHeight(-300.0 + t * 600.0, np.zeros(9), 0.0)
    latB, lonB, _ = plane.frame.columnPointAtHeight(np.zeros(9), -300.0 + t * 600.0, 0.0)
    writeGeoParquet(
        tmp_path / "t.parquet",
        np.array(
            [
                shapely.LineString(np.column_stack([lonR, latR])),
                shapely.LineString(np.column_stack([lonB, latB])),
            ],
            dtype=object,
        ),
        {
            "sourceFeatureId": ["ground", "bridge"],
            "kind": ["road", "road"],
            "isBridge": [False, True],
            "level": [0, 1],
            "width_m": [8.0, 8.0],
        },
    )
    ingestFiles(pack, "overtureTransportation", [tmp_path / "t.parquet"])
    scene = buildScene(_request(plane), pack)
    r, c = _rc(scene, 0.0, 0.0)  # the crossing: ground road under the bridge
    assert scene.layers["landClass"][r, c] == 210  # the corridor keeps its class
    assert scene.layers["heightSource"][r, c] == HeightSource.DECK_ESTIMATED  # surface = deck
    r2, c2 = _rc(scene, 0.0, 200.0)  # on the bridge, off the ground road
    assert scene.layers["landClass"][r2, c2] == 210  # the bridge's own corridor


def testRaisedBuildingPart_isAnElevatedLevelOverOpenGround(plane: SyntheticScene, tmp_path: Path):
    pack = tmp_path / "park.pack"
    shutil.copytree(plane.packDir, pack)

    def box(e0: float, n0: float, e1: float, n1: float) -> shapely.Polygon:
        lat, lon, _ = plane.frame.columnPointAtHeight(
            np.array([e0, e1, e1, e0]), np.array([n0, n0, n1, n1]), 0.0
        )
        return shapely.Polygon(np.column_stack([lon, lat]))

    writeGeoParquet(
        tmp_path / "o.parquet",
        np.array([box(-360, 300, -340, 320), box(-370, 300, -230, 320)], dtype=object),
        {
            "sourceFeatureId": ["tower", "park"],
            "height_m": [100.0, 110.0],
            "numFloors": [None, None],
            "minHeight_m": [None, 100.0],
        },
    )
    ingestFiles(pack, "overtureBuildings", [tmp_path / "o.parquet"])
    scene = buildScene(_request(plane), pack)
    free, onTower = _rc(scene, -280.0, 310.0), _rc(scene, -350.0, 310.0)
    surface = scene.layers["surfaceHeight_datum_m"]
    ground = scene.layers["groundHeight_datum_m"]
    bottom, top = scene.layers["deckBottomHeight_datum_m"], scene.layers["deckTopHeight_datum_m"]
    assert surface[free] == pytest.approx(top[(0, *free)])
    assert bottom[(0, *free)] - ground[free] == pytest.approx(100.0, abs=2.5)  # sloped ground
    assert scene.layers["landClass"][free] == 200
    assert scene.layers["buildingHeight_m"][free] == pytest.approx(surface[free] - ground[free])
    assert np.isnan(top[(0, *onTower)])  # the park rests on the tower: one solid column
    assert surface[onTower] == pytest.approx(surface[free], abs=1.5)
