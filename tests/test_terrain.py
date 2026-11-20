"""Tests for ``map3d.layers.terrain``."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from map3d.gridGeometry import GridGeometry
from map3d.layers.columns import ChunkColumns
from map3d.layers.terrain import GroundModel
from map3d.pack.reader import Pack
from map3d.pack.synthetic import SyntheticScene, makeSyntheticPack
from map3d.request import SceneRequest


@pytest.fixture(scope="module")
def plane(tmp_path_factory: pytest.TempPathFactory) -> SyntheticScene:
    return makeSyntheticPack(tmp_path_factory.mktemp("t") / "plane.pack", terrain="plane")


@pytest.fixture(scope="module")
def hill(tmp_path_factory: pytest.TempPathFactory) -> SyntheticScene:
    return makeSyntheticPack(tmp_path_factory.mktemp("t") / "hill.pack", terrain="hill")


def _geometry(scene: SyntheticScene, heading_deg: float, swath_m: float, s: float) -> GridGeometry:
    h = math.radians(heading_deg)
    axes = [[math.cos(h), -math.sin(h), 0.0], [math.sin(h), math.cos(h), 0.0], [0.0, 0.0, 1.0]]
    lat, lon = scene.frame.lat0_deg, scene.frame.lon0_deg
    return GridGeometry.fromRequest(SceneRequest((lat, lon), s, axes, swath_m, swath_m))


def _cornerMax(scene: SyntheticScene, geom: GridGeometry, f) -> np.ndarray:
    s = geom.gridSpacing_m
    gx = np.arange(geom.nCols + 1) * s - geom.nCols * s / 2
    gy = np.arange(geom.nRows + 1) * s - geom.nRows * s / 2
    a = geom.gridAxes_ENU
    e = gy[:, None] * a[1, 0] + gx[None, :] * a[0, 0]
    n = gy[:, None] * a[1, 1] + gx[None, :] * a[0, 1]
    v = f(e, n)
    return np.maximum.reduce([v[:-1, :-1], v[1:, :-1], v[:-1, 1:], v[1:, 1:]])


def testGroundModel_usesPrimarySourceAndGeoid(plane: SyntheticScene):
    geom = _geometry(plane, 0.0, 800.0, 20.0)
    model = GroundModel.load(Pack.open(plane.packDir), geom)
    assert model.sourceId == "syntheticGround"
    assert model.nativeResolution_m == 5.0
    lat, lon = ChunkColumns(geom, slice(None), slice(None)).nodeLatLon()
    np.testing.assert_allclose(model.undulation_m(lat, lon), plane.geoidUndulation_m)


@pytest.mark.parametrize("heading_deg", [0.0, 30.0])
def testCellMax_planeMatchesAnalyticCornerMax(plane: SyntheticScene, heading_deg: float):
    geom = _geometry(plane, heading_deg, 800.0, 20.0)
    model = GroundModel.load(Pack.open(plane.packDir), geom)
    ground = model.cellMaxOrthometric(
        ChunkColumns(geom, slice(None), slice(None)), geom.gridSpacing_m
    )
    np.testing.assert_allclose(ground, _cornerMax(plane, geom, plane.terrainHeight_m), atol=2e-3)


def testCellMax_hillCloseToAnalytic(hill: SyntheticScene):
    geom = _geometry(hill, 15.0, 800.0, 10.0)
    model = GroundModel.load(Pack.open(hill.packDir), geom)
    ground = model.cellMaxOrthometric(
        ChunkColumns(geom, slice(None), slice(None)), geom.gridSpacing_m
    )
    np.testing.assert_allclose(ground, _cornerMax(hill, geom, hill.terrainHeight_m), atol=0.05)


def testCellMax_supersamplesCoarseGrids(hill: SyntheticScene):
    # 40 m cells over a 5 m source -> 8x8 sub-lattice catches the hill top inside a cell.
    geom = _geometry(hill, 0.0, 400.0, 40.0)
    model = GroundModel.load(Pack.open(hill.packDir), geom)
    ground = model.cellMaxOrthometric(
        ChunkColumns(geom, slice(None), slice(None)), geom.gridSpacing_m
    )
    assert np.nanmax(ground) == pytest.approx(40.0, abs=0.1)  # hill peak 10 + 30


def testChunked_equalsWhole(plane: SyntheticScene):
    geom = _geometry(plane, 30.0, 800.0, 20.0)
    model = GroundModel.load(Pack.open(plane.packDir), geom)
    whole = model.cellMaxOrthometric(ChunkColumns(geom, slice(None), slice(None)), 20.0)
    parts = np.full(geom.shape, np.nan)
    for rows, cols in geom.iterChunks(7, 11):
        parts[rows, cols] = model.cellMaxOrthometric(ChunkColumns(geom, rows, cols), 20.0)
    np.testing.assert_array_equal(parts, whole)


def testEllipsoidReferencedSourceIsConvertedToOrthometric(plane: SyntheticScene):
    geom = _geometry(plane, 0.0, 400.0, 20.0)
    model = GroundModel.load(Pack.open(plane.packDir), geom)
    ellModel = dataclasses.replace(model, verticalReference="WGS84")
    lat, lon = np.array([plane.frame.lat0_deg]), np.array([plane.frame.lon0_deg])
    assert float(ellModel.orthometricAt(lat, lon)[0]) == pytest.approx(
        float(model.orthometricAt(lat, lon)[0]) - plane.geoidUndulation_m
    )


def testUnsupportedVerticalReferenceRaises(plane: SyntheticScene):
    geom = _geometry(plane, 0.0, 400.0, 20.0)
    model = dataclasses.replace(
        GroundModel.load(Pack.open(plane.packDir), geom), verticalReference="aboveGround"
    )
    with pytest.raises(ValueError, match="vertical reference"):
        model.orthometricAt(np.array([1.3]), np.array([103.8]))


def testLoad_missingGroundRaises(tmp_path: Path, writeRaster):
    from map3d.pack.ingest import ingestFiles

    root = tmp_path / "g.pack"
    ingestFiles(
        root,
        "egm2008",
        [writeRaster(tmp_path / "g.tif", (103.0, 1.0, 104.0, 2.0))],
        (103.0, 1.0, 104.0, 2.0),
    )
    geom = GridGeometry.fromRequest(SceneRequest((1.5, 103.5), 10.0, np.eye(3), 100.0, 100.0))
    with pytest.raises(ValueError, match="ground"):
        GroundModel.load(Pack.open(root), geom)
