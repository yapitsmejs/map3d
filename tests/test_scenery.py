"""Tests for ``map3d.mesh.scenery`` (terrain / building / deck meshes and textures)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from map3d.mesh import scenery
from map3d.pack.synthetic import SyntheticScene, makeSyntheticPack
from map3d.request import SceneRequest
from map3d.scene import Scene, buildScene


@pytest.fixture(scope="module")
def plane(tmp_path_factory: pytest.TempPathFactory) -> SyntheticScene:
    return makeSyntheticPack(tmp_path_factory.mktemp("m") / "plane.pack", terrain="plane")


@pytest.fixture(scope="module")
def scene(plane: SyntheticScene) -> Scene:
    h = math.radians(20.0)
    axes = [[math.cos(h), -math.sin(h), 0.0], [math.sin(h), math.cos(h), 0.0], [0.0, 0.0, 1.0]]
    return buildScene(
        SceneRequest((plane.frame.lat0_deg, plane.frame.lon0_deg), 5.0, axes, 1000.0, 1000.0),
        plane.packDir,
    )


def _full(scene: Scene) -> scenery.TileExtent:
    return scenery.TileExtent(-500.0, 500.0, -500.0, 500.0)


def testGridEnuRoundTrip(scene: Scene):
    gx, gy = np.array([10.0, -200.0]), np.array([30.0, 400.0])
    e, n = scenery.gridToEnu(scene.geometry, gx, gy)
    bx, by = scenery.enuToGrid(scene.geometry, e, n)
    np.testing.assert_allclose([bx, by], [gx, gy], atol=1e-9)


def testTerrain_verticesSitOnGroundColumns(scene: Scene, plane: SyntheticScene):
    part = scenery.terrainPart(scene, _full(scene), gridSize=33, maxError_m=0.0, withCanopy=False)
    e, n, u = part.positions_ENU_m.T
    lat, lon, uRef = scene.geometry.frame.columnPointAtHeight(
        e, n, plane.terrainHeight_m(e, n) + 8.0
    )
    np.testing.assert_allclose(u, uRef, atol=2e-3)  # bilinear raster sample of a linear plane
    assert part.uv is not None and part.uv.min() >= 0.0 and part.uv.max() <= 1.0
    # Upward-facing: every triangle's normal has +U.
    p = part.positions_ENU_m[part.triangles]
    normalU = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])[:, 2]
    assert (normalU > 0).all()


def testTerrain_flatPlaneSimplifiesWithTolerance(scene: Scene):
    exact = scenery.terrainPart(scene, _full(scene), gridSize=65, maxError_m=0.0)
    coarse = scenery.terrainPart(scene, _full(scene), gridSize=65, maxError_m=0.5)
    assert coarse.triangles.shape[0] < exact.triangles.shape[0]


def testBuildings_roofAtRoofHeightAndWallsColored(scene: Scene):
    buildings = scene.models.buildings
    roofs, walls = scenery.buildingParts(scene, np.array([0]), _full(scene))
    # Flat in ellipsoidal height, so corners differ by Earth curvature (mm) in ENU Up.
    lon, lat = np.asarray(buildings.geometries[0].exterior.coords).T
    expectedU = scene.geometry.frame.geodeticToEnu(lat, lon, buildings.roofOrthometric_m[0] + 8.0)[
        2
    ]
    assert roofs.positions_ENU_m[:, 2].max() == pytest.approx(float(expectedU.max()), abs=1e-6)
    assert roofs.positions_ENU_m[:, 2].min() == pytest.approx(float(expectedU.min()), abs=1e-6)
    assert roofs.uv is not None and walls.colors is not None
    # Rectangle: 2 roof triangles, 4 walls x 2 triangles.
    assert roofs.triangles.shape[0] == 2 and walls.triangles.shape[0] == 8
    p = roofs.positions_ENU_m[roofs.triangles]
    assert (np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])[:, 2] > 0).all()


def testDecks_slabFollowsProfile(scene: Scene, plane: SyntheticScene):
    tops, sides = scenery.deckParts(scene, np.array([0]), _full(scene))
    assert not tops.isEmpty and not sides.isEmpty
    e, n, u = tops.positions_ENU_m.T
    # Top = plane ground along n = 200 (interpolated between the bridge ends) + 7 m.
    _, _, uRef = scene.geometry.frame.columnPointAtHeight(
        e, n, plane.terrainHeight_m(e, 200.0) + 7.0 + 8.0
    )
    np.testing.assert_allclose(u, uRef, atol=0.05)
    assert sides.positions_ENU_m[:, 2].min() < u.min()  # underside is below the top


def testEmptySelections_giveEmptyParts(scene: Scene):
    empty = np.array([], dtype=np.int64)
    assert all(p.isEmpty for p in scenery.buildingParts(scene, empty, _full(scene)))
    assert all(p.isEmpty for p in scenery.deckParts(scene, empty, _full(scene)))


def testTexture_landClassFallbackWithoutImagery(scene: Scene):
    jpeg = scenery.renderTexture(scene, None, _full(scene), size=64)
    assert jpeg[:2] == b"\xff\xd8"


def testTexture_fromByoImageryFile(scene: Scene, tmp_path: Path):
    bounds = scene.geometry.footprintBounds_deg(margin_m=50.0)
    path = tmp_path / "ortho.tif"
    data = np.stack([np.full((50, 50), v, np.uint8) for v in (200, 30, 30)])
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=50,
        height=50,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_bounds(*bounds, 50, 50),
    ) as dst:
        dst.write(data)
    imagery = scenery.ImagerySource.fromFile(path, bounds)
    rgb = imagery.sampleRgb(
        np.array([scene.geometry.frame.lat0_deg]), np.array([scene.geometry.frame.lon0_deg])
    )
    np.testing.assert_array_equal(rgb, [[200, 30, 30]])
    assert scenery.renderTexture(scene, imagery, _full(scene), size=32)[:2] == b"\xff\xd8"


def testImagerySource_fromPackNoneWithoutImageryRole(scene: Scene):
    assert scenery.ImagerySource.fromPack(scene.pack, scene.geometry.footprintBounds_deg()) is None


def testTerrain_includesCanopyAboveThresholdWhenAsked(scene: Scene, plane: SyntheticScene):
    # Synthetic canopy: a 15 m patch over [-400, -300]^2 (scene axes rotated 20 deg).
    bare = scenery.terrainPart(scene, _full(scene), gridSize=129, maxError_m=0.0, withCanopy=False)
    vegetated = scenery.terrainPart(scene, _full(scene), gridSize=129, maxError_m=0.0)

    def heightNear(part, e0, n0):
        e, n, u = part.positions_ENU_m.T
        i = np.argmin((e - e0) ** 2 + (n - n0) ** 2)
        _, _, uGround = scene.geometry.frame.columnPointAtHeight(
            e[i], n[i], plane.terrainHeight_m(e[i], n[i]) + 8.0
        )
        return u[i] - float(uGround)

    assert heightNear(vegetated, -350.0, -350.0) == pytest.approx(15.0, abs=0.05)
    assert heightNear(bare, -350.0, -350.0) == pytest.approx(0.0, abs=0.05)
    assert heightNear(vegetated, 150.0, 150.0) == pytest.approx(0.0, abs=0.05)  # open ground


def testTerrain_ignoresCanopyBelowThreshold(scene: Scene, monkeypatch: pytest.MonkeyPatch):
    from map3d.layers.canopy import CanopyModel

    monkeypatch.setattr(
        CanopyModel, "heightAt", lambda self, lat, lon, col, row: np.full(lat.shape, 1.5)
    )
    low = scenery.terrainPart(scene, _full(scene), gridSize=33, maxError_m=0.0)
    bare = scenery.terrainPart(scene, _full(scene), gridSize=33, maxError_m=0.0, withCanopy=False)
    np.testing.assert_allclose(low.positions_ENU_m, bare.positions_ENU_m)
