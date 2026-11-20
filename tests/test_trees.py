"""Tests for ``map3d.mesh.trees`` (individual trees from canopy data)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from map3d.mesh import trees as tr
from map3d.pack.synthetic import SyntheticScene, makeSyntheticPack
from map3d.request import SceneRequest
from map3d.scene import Scene, buildScene

PATCH = (-400.0, -400.0, -300.0, -300.0)  # synthetic 15 m canopy patch (ENU)


@pytest.fixture(scope="module")
def plane(tmp_path_factory: pytest.TempPathFactory) -> SyntheticScene:
    return makeSyntheticPack(tmp_path_factory.mktemp("tr") / "plane.pack", terrain="plane")


@pytest.fixture(scope="module")
def scene(plane: SyntheticScene) -> Scene:
    h = math.radians(20.0)
    axes = [[math.cos(h), -math.sin(h), 0.0], [math.sin(h), math.cos(h), 0.0], [0.0, 0.0, 1.0]]
    return buildScene(
        SceneRequest((plane.frame.lat0_deg, plane.frame.lon0_deg), 5.0, axes, 1000.0, 1000.0),
        plane.packDir,
    )


@pytest.fixture(scope="module")
def trees(scene: Scene) -> tr.TreeSet:
    return tr.TreeSet.fromScene(scene)


def _inPatch(t: tr.TreeSet, pad: float = 0.0) -> np.ndarray:
    return (
        (t.e_m >= PATCH[0] - pad)
        & (t.e_m <= PATCH[2] + pad)
        & (t.n_m >= PATCH[1] - pad)
        & (t.n_m <= PATCH[3] + pad)
    )


def testTrees_fillTheCanopyPatchWithFullHeightCrowns(trees: tr.TreeSet):
    inside = _inPatch(trees, pad=5.0)
    assert inside.sum() > 50  # a forest of crowns, not one blob
    np.testing.assert_allclose(trees.height_m[inside], 15.0, atol=0.05)
    # Crowns are sized from the canopy they cover: together they fill the patch.
    crownArea = np.pi * trees.crownRadius_m[inside] ** 2
    assert crownArea.sum() == pytest.approx(100.0 * 100.0, rel=0.15)
    assert np.all(
        (trees.crownRadius_m >= tr.MIN_CROWN_RADIUS_m)
        & (trees.crownRadius_m <= tr.MAX_CROWN_RADIUS_m)
    )
    assert len(trees) == trees.e_m.size


def testTrees_includeMappedTreePointOnce(trees: tr.TreeSet):
    near = np.hypot(trees.e_m - 0.0, trees.n_m + 250.0) < 1.0
    assert near.sum() == 1
    assert trees.height_m[near][0] == pytest.approx(8.0)
    assert trees.crownRadius_m[near][0] == pytest.approx(tr.crownRadius_m(8.0))  # allometric


def testTrees_onlyOnOpenGroundInsideTheSwath(trees: tr.TreeSet):
    assert np.all(np.abs(trees.gx_m) <= 500.0) and np.all(np.abs(trees.gy_m) <= 500.0)
    # Nothing outside the canopy patch except the mapped tree point.
    assert (~_inPatch(trees, pad=5.0)).sum() == 1


def testTrees_standOnTheGround(trees: tr.TreeSet, scene: Scene, plane: SyntheticScene):
    frame = scene.geometry.frame
    h = plane.terrainHeight_m(trees.e_m, trees.n_m) + plane.geoidUndulation_m
    e, n, u = frame.geodeticToEnu(trees.lat_deg, trees.lon_deg, h)
    np.testing.assert_allclose(e, trees.e_m, atol=0.01)
    np.testing.assert_allclose(n, trees.n_m, atol=0.01)
    np.testing.assert_allclose(trees.baseU_m, u, atol=0.05)


def testTrees_skipBuildingsAndDecks(scene: Scene):
    # A canopy reading on a roof or deck is not a tree on the ground.
    layers = dict(scene.layers)
    layers["landClass"] = np.full_like(np.asarray(scene.layers["landClass"]), 200)
    roofed = dataclasses.replace(scene, layers=layers)
    assert len(tr.TreeSet.fromScene(roofed)) == 0


def testTrees_noSourcesGiveEmptySet(scene: Scene):
    canopy = dataclasses.replace(
        scene.models.canopy,
        window=None,
        treeDiscs=np.array([], dtype=object),
        treeHeights_m=np.array([]),
    )
    bare = dataclasses.replace(scene, models=dataclasses.replace(scene.models, canopy=canopy))
    empty = tr.TreeSet.fromScene(bare)
    assert len(empty) == 0 and empty.select(np.array([], dtype=np.int64)) == []


def testTreeParts_instancesSizedFromHeightAndRadius(trees: tr.TreeSet):
    parts = trees.select(np.arange(len(trees)))
    crowns = [p for p in parts if p.color in tr.CROWN_COLORS]
    trunks = [p for p in parts if p.color == tr.TRUNK_COLOR]
    assert sum(p.translations_ENU_m.shape[0] for p in crowns) == len(trees)
    assert len(trunks) == 1 and trunks[0].translations_ENU_m.shape[0] == len(trees)
    assert len({p.color for p in crowns}) == 3  # shade variation
    # Crown of the mapped 8 m tree: top at base + height, width = crown radius.
    i = int(np.argmin(np.hypot(trees.e_m, trees.n_m + 250.0)))
    found = 0
    for p in crowns:
        hit = np.flatnonzero(
            np.isclose(p.translations_ENU_m[:, 0], trees.e_m[i])
            & np.isclose(p.translations_ENU_m[:, 1], trees.n_m[i])
        )
        for h in hit:
            found += 1
            t, s = p.translations_ENU_m[h], p.scales[h]
            assert s[0] == s[1] == pytest.approx(trees.crownRadius_m[i])
            assert t[2] + s[2] == pytest.approx(trees.baseU_m[i] + trees.height_m[i])
            assert t[2] - s[2] > trees.baseU_m[i]  # crown clears the ground
    assert found == 1
    np.testing.assert_allclose(trunks[0].translations_ENU_m[:, 2], trees.baseU_m)
    assert np.all(trunks[0].scales[:, 2] > 0)


def testCrownRadius_growsWithHeightWithinLimits():
    np.testing.assert_allclose(tr.crownRadius_m([1.0, 5.0, 10.0, 40.0]), [2.0, 2.5, 4.0, 8.0])


def testMaxFilter_findsTreetopsNotCrownEdges():
    # One 9 m crown (cone) and a lone 4 m tree: only their tops are window maxima.
    rows, cols = np.mgrid[0:30, 0:30]
    cone = np.clip(9.0 - 1.5 * np.hypot(rows - 10, cols - 10), 0, None)
    lone = np.where(np.hypot(rows - 22, cols - 22) < 1.0, 4.0, 0.0)
    canopy = np.maximum(cone, lone)
    peak = tr._maxFilter(canopy, 2, 2)
    tops = np.argwhere((canopy >= tr.MIN_TREE_HEIGHT_m) & (canopy >= peak))
    assert tops.tolist() == [[10, 10], [22, 22]]
    np.testing.assert_array_equal(tr._maxFilter(canopy, 0, 0), canopy)


def testSegment_smoothWoodIsFilledWithCrowns():
    # A wide, smooth 12 m dome (one treetop) over 80 x 80 m of 1 m pixels: the
    # treetop alone reaches 6 m, so gap filling must seed the rest of the wood.
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    from map3d.pack.reader import RasterWindow

    rows, cols = np.mgrid[0:100, 0:100]
    d = np.hypot(rows - 50, cols - 50)
    dome = np.where(d < 40, 12.0 - 2.0 * (d / 40) ** 2, 0.0)
    window = RasterWindow(
        dome, Affine(1.0, 0, 11_550_000.0, 0, -1.0, 150_000.0), CRS.from_epsg(3857), "c"
    )
    lat, lon, h, r = tr._segmentRaster(window, np.zeros(0), np.zeros(0), np.zeros(0))
    assert lat.size > 30  # many crowns, not one
    assert h.max() == pytest.approx(12.0, abs=0.1)
    assert np.all(r <= tr.MAX_CROWN_RADIUS_m) and np.all(r >= tr.MIN_CROWN_RADIUS_m)
    # The crowns account for most of the wood's area.
    assert np.sum(np.pi * r**2) == pytest.approx(np.pi * 40.0**2, rel=0.25)
