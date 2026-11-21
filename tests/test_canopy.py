"""Tests for ``map3d.layers.canopy``."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from _support import headingAxes

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
    axes = headingAxes(heading)
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
    assert model.window is None
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
