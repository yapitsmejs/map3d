"""Tests for ``map3d.layers.landCover``."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from map3d.gridGeometry import GridGeometry
from map3d.layers import landCover as lc
from map3d.layers.columns import ChunkColumns
from map3d.pack.ingest import ingestFiles
from map3d.pack.manifest import Manifest
from map3d.pack.reader import Pack
from map3d.pack.synthetic import SyntheticScene, makeSyntheticPack
from map3d.request import SceneRequest


@pytest.fixture(scope="module")
def plane(tmp_path_factory: pytest.TempPathFactory) -> SyntheticScene:
    return makeSyntheticPack(tmp_path_factory.mktemp("l") / "plane.pack")


def _geometry(scene: SyntheticScene, s: float = 2.0, heading: float = 0.0) -> GridGeometry:
    h = math.radians(heading)
    axes = [[math.cos(h), -math.sin(h), 0.0], [math.sin(h), math.cos(h), 0.0], [0.0, 0.0, 1.0]]
    return GridGeometry.fromRequest(
        SceneRequest((scene.frame.lat0_deg, scene.frame.lon0_deg), s, axes, 1000.0, 1000.0)
    )


def _classes(pack: Path, geom: GridGeometry) -> np.ndarray:
    model = lc.LandCoverModel.load(Pack.open(pack), geom)
    return model.classes(ChunkColumns(geom, slice(None), slice(None)))


def _at(geom: GridGeometry, values: np.ndarray, e: float, n: float) -> int:
    ee, nn = geom.nodesEnu()
    return int(values[np.unravel_index(np.argmin((ee - e) ** 2 + (nn - n) ** 2), ee.shape)])


def testClasses_landCoverAndGroundRoads(plane: SyntheticScene):
    geom = _geometry(plane)
    classes = _classes(plane.packDir, geom)
    assert classes.dtype == np.uint8
    assert _at(geom, classes, 0.0, 0.0) == 30  # grass
    assert _at(geom, classes, 0.0, 450.0) == 80  # water
    assert _at(geom, classes, -350.0, -350.0) == 10  # tree patch
    assert _at(geom, classes, 100.0, -450.0) == lc.CLASS_ROAD  # 8 m wide ground road
    assert _at(geom, classes, 100.0, -460.0) == 30  # 10 m off its centerline
    assert _at(geom, classes, 0.0, 200.0) == 30  # the bridge is not a ground road


def testClasses_rotatedGridChunkedEqualsWhole(plane: SyntheticScene):
    geom = _geometry(plane, s=4.0, heading=40.0)
    model = lc.LandCoverModel.load(Pack.open(plane.packDir), geom)
    whole = model.classes(ChunkColumns(geom, slice(None), slice(None)))
    parts = np.zeros_like(whole)
    for rows, cols in geom.iterChunks(33, 71):
        parts[rows, cols] = model.classes(ChunkColumns(geom, rows, cols))
    np.testing.assert_array_equal(parts, whole)


def testImpactObservatory_isRemappedToWorldCover(
    plane: SyntheticScene, tmp_path: Path, writeRaster
):
    pack = tmp_path / "io.pack"
    shutil.copytree(plane.packDir, pack)
    bounds = Manifest.load(pack).bounds_deg
    ingestFiles(
        pack,
        "impactObservatoryLulc",
        [writeRaster(tmp_path / "io.tif", bounds, values=np.full((10, 10), 1, np.uint8))],
    )
    geom = _geometry(plane, s=20.0)
    classes = _classes(pack, geom)
    assert _at(geom, classes, 0.0, 0.0) == 80  # IO 1 (water) -> WorldCover 80
    assert lc.remapClasses(np.array([2, 7, 11, 3]), "impactObservatoryLulc").tolist() == [
        10,
        50,
        30,
        0,
    ]
    assert lc.remapClasses(np.array([10, 80]), "esaWorldCover").tolist() == [10, 80]


def testNoLandCoverSource_givesZeroButRoadsStillDrawn(plane: SyntheticScene, tmp_path: Path):
    pack = tmp_path / "nolc.pack"
    shutil.copytree(plane.packDir, pack)
    manifest = Manifest.load(pack)
    del manifest.sources["syntheticLandCover"]
    manifest.save(pack)
    geom = _geometry(plane)
    classes = _classes(pack, geom)
    assert _at(geom, classes, 0.0, 0.0) == 0
    assert _at(geom, classes, 100.0, -450.0) == lc.CLASS_ROAD


def testClassTable_namesEveryCode():
    table = lc.classTable()
    assert table["200"] == "building" and table["220"] == "deck" and table["80"] == "water"


def testFootways_doNotOverrideLandCover(plane: SyntheticScene, tmp_path: Path):
    import shapely

    from map3d.pack.geoparquet import writeGeoParquet

    pack = tmp_path / "fw.pack"
    shutil.copytree(plane.packDir, pack)
    lat, lon, _ = plane.frame.columnPointAtHeight(
        np.array([-300.0, 300.0]), np.array([0.0, 0.0]), 0.0
    )
    writeGeoParquet(
        tmp_path / "fw.parquet",
        np.array([shapely.LineString(np.column_stack([lon, lat]))], dtype=object),
        {
            "sourceFeatureId": ["fw"],
            "kind": ["footway"],
            "isBridge": [False],
            "level": [0],
            "width_m": [None],
        },
    )
    ingestFiles(pack, "overtureTransportation", [tmp_path / "fw.parquet"])
    geom = _geometry(plane)
    assert _at(geom, _classes(pack, geom), 0.0, 0.0) == 30


def testClasses_segmentEndsBurnWithRoundCaps(plane: SyntheticScene, tmp_path: Path):
    """A node just past a road segment's end still burns (round cap, not flat).

    With flat caps, the last node of a dead-end road projects past the
    endpoint and falls outside the burn, painting raw land cover.
    """
    import shapely

    from map3d.pack.geoparquet import writeGeoParquet

    pack = tmp_path / "ends.pack"
    shutil.copytree(plane.packDir, pack)
    manifest = Manifest.load(pack)
    del manifest.sources["syntheticTransport"], manifest.sources["syntheticElevatedStructures"]
    manifest.save(pack)
    t = np.linspace(0.0, 1.0, 9)
    lat, lon, _ = plane.frame.columnPointAtHeight(
        -100.0 + t * 200.0, np.zeros(9), 0.0
    )  # E-W road, e = -100..100
    writeGeoParquet(
        tmp_path / "t.parquet",
        np.array([shapely.LineString(np.column_stack([lon, lat]))], dtype=object),
        {
            "sourceFeatureId": ["r0"],
            "kind": ["road"],
            "isBridge": [False],
            "level": [0],
            "width_m": [8.0],
        },
    )
    ingestFiles(pack, "overtureTransportation", [tmp_path / "t.parquet"])
    geom = _geometry(plane)
    classes = _classes(pack, geom)
    # the node whose centre projects ~1 m past the segment's east end
    assert _at(geom, classes, 101.0, -1.0) == lc.CLASS_ROAD
