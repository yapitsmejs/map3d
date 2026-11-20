"""Tests for ``map3d.layers.canopy``."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from map3d.gridGeometry import GridGeometry
from map3d.layers.canopy import CanopyModel, treeRadius_m
from map3d.layers.columns import ChunkColumns
from map3d.pack.manifest import Manifest
from map3d.pack.reader import Pack
from map3d.pack.synthetic import SyntheticScene, makeSyntheticPack
from map3d.request import SceneRequest


@pytest.fixture(scope="module")
def plane(tmp_path_factory: pytest.TempPathFactory) -> SyntheticScene:
    return makeSyntheticPack(tmp_path_factory.mktemp("c") / "plane.pack")


def _geometry(
    scene: SyntheticScene, s: float = 2.0, swath: float = 1000.0, heading: float = 0.0
) -> GridGeometry:
    h = math.radians(heading)
    axes = [[math.cos(h), -math.sin(h), 0.0], [math.sin(h), math.cos(h), 0.0], [0.0, 0.0, 1.0]]
    return GridGeometry.fromRequest(
        SceneRequest((scene.frame.lat0_deg, scene.frame.lon0_deg), s, axes, swath, swath)
    )


def _at(geom: GridGeometry, values: np.ndarray, e: float, n: float) -> float:
    ee, nn = geom.nodesEnu()
    idx = np.unravel_index(np.argmin((ee - e) ** 2 + (nn - n) ** 2), ee.shape)
    return float(values[idx])


def _canopy(scene: SyntheticScene, geom: GridGeometry, packDir: Path | None = None) -> np.ndarray:
    model = CanopyModel.load(Pack.open(packDir or scene.packDir), geom)
    return model.cellMax(ChunkColumns(geom, slice(None), slice(None)))


def testTreeRadius():
    assert treeRadius_m(4.0) == 2.0
    assert treeRadius_m(20.0) == 5.0


def testCanopy_rasterPatchAndOpenGround(plane: SyntheticScene):
    geom = _geometry(plane)
    canopy = _canopy(plane, geom)
    assert _at(geom, canopy, -350.0, -350.0) == pytest.approx(15.0)
    assert _at(geom, canopy, 100.0, 100.0) == 0.0


def testCanopy_cellMaxIsConservativeAtPatchEdge(plane: SyntheticScene):
    # Patch edge at e = -300; a 10 m cell straddling it takes the patch height.
    geom = _geometry(plane, s=10.0, swath=1010.0)
    canopy = _canopy(plane, geom)
    assert _at(geom, canopy, -297.0, -350.0) == pytest.approx(15.0, abs=0.5)


def testCanopy_treePointsAddDiscs(plane: SyntheticScene):
    geom = _geometry(plane)
    canopy = _canopy(plane, geom)
    assert _at(geom, canopy, 0.0, -250.0) == pytest.approx(8.0)  # synthetic tree, 8 m
    assert _at(geom, canopy, 0.0, -240.0) == 0.0  # outside its 2 m crown


def testCanopy_chunkedEqualsWhole(plane: SyntheticScene):
    geom = _geometry(plane, s=4.0, heading=25.0)
    model = CanopyModel.load(Pack.open(plane.packDir), geom)
    whole = model.cellMax(ChunkColumns(geom, slice(None), slice(None)))
    parts = np.full(geom.shape, np.nan)
    for rows, cols in geom.iterChunks(41, 67):
        parts[rows, cols] = model.cellMax(ChunkColumns(geom, rows, cols))
    np.testing.assert_array_equal(parts, whole)


def testCanopy_noSourcesGivesNan(plane: SyntheticScene, tmp_path: Path):
    pack = tmp_path / "x.pack"
    shutil.copytree(plane.packDir, pack)
    manifest = Manifest.load(pack)
    del manifest.sources["syntheticCanopy"], manifest.sources["syntheticTrees"]
    manifest.save(pack)
    geom = _geometry(plane, s=20.0)
    model = CanopyModel.load(Pack.open(pack), geom)
    assert model.sourceId is None
    assert np.isnan(model.cellMax(ChunkColumns(geom, slice(None), slice(None)))).all()


def testCanopy_treesOnlyGivesZeroElsewhere(plane: SyntheticScene, tmp_path: Path):
    pack = tmp_path / "t.pack"
    shutil.copytree(plane.packDir, pack)
    manifest = Manifest.load(pack)
    del manifest.sources["syntheticCanopy"]
    manifest.save(pack)
    geom = _geometry(plane)
    canopy = _canopy(plane, geom, pack)
    assert _at(geom, canopy, 0.0, -250.0) == pytest.approx(8.0)
    assert _at(geom, canopy, -350.0, -350.0) == 0.0


def testHeightAt_rasterTreesAndOpenGround(plane: SyntheticScene):
    geom = _geometry(plane)
    model = CanopyModel.load(Pack.open(plane.packDir), geom)
    e = np.array([-350.0, 0.0, 0.0, 100.0])
    n = np.array([-350.0, -250.0, -240.0, 100.0])
    lat, lon, _ = geom.frame.columnPointAtHeight(e, n, 0.0)
    s = geom.gridSpacing_m
    col = (e + 0.5 * geom.nCols * s) / s  # identity axes: grid x = east, y = north
    row = (n + 0.5 * geom.nRows * s) / s
    np.testing.assert_allclose(model.heightAt(lat, lon, col, row), [15.0, 8.0, 0.0, 0.0], atol=1e-6)


def testHeightAt_noSourcesIsZero(plane: SyntheticScene, tmp_path: Path):
    pack = tmp_path / "x.pack"
    shutil.copytree(plane.packDir, pack)
    manifest = Manifest.load(pack)
    del manifest.sources["syntheticCanopy"], manifest.sources["syntheticTrees"]
    manifest.save(pack)
    model = CanopyModel.load(Pack.open(pack), _geometry(plane))
    out = model.heightAt(np.array([1.3]), np.array([103.8]), np.array([5.0]), np.array([5.0]))
    np.testing.assert_array_equal(out, [0.0])
