"""Tests for ``map3d.mesh.scenery`` (terrain / building / deck meshes and textures)."""

from __future__ import annotations

import numpy as np
import pytest
from _meshSupport import decodePng, sceneOver

from map3d.mesh import scenery
from map3d.mesh.buildingAppearance import BuildingAppearance
from map3d.pack.synthetic import SyntheticScene, makeSyntheticPack
from map3d.scene import Scene


@pytest.fixture(scope="module")
def plane(tmp_path_factory: pytest.TempPathFactory) -> SyntheticScene:
    return makeSyntheticPack(tmp_path_factory.mktemp("m") / "plane.pack", terrain="plane")


@pytest.fixture(scope="module")
def scene(plane: SyntheticScene) -> Scene:
    return sceneOver(plane, angle_deg=20.0)


@pytest.fixture
def neutral(scene: Scene) -> BuildingAppearance:
    return BuildingAppearance.neutral(scene.models.buildings)


FULL = scenery.TileExtent(-500.0, 500.0, -500.0, 500.0)  # the whole 1 km swath


def testGridEnuRoundTrip(scene: Scene):
    gx, gy = np.array([10.0, -200.0]), np.array([30.0, 400.0])
    e, n = scenery.gridToEnu(scene.geometry, gx, gy)
    bx, by = scenery.enuToGrid(scene.geometry, e, n)
    np.testing.assert_allclose([bx, by], [gx, gy], atol=1e-9)


def testTerrain_verticesSitOnGroundColumns(scene: Scene, plane: SyntheticScene):
    part = scenery.terrainPart(scene, FULL, gridSize=33, maxError_m=0.0)
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
    exact = scenery.terrainPart(scene, FULL, gridSize=65, maxError_m=0.0)
    coarse = scenery.terrainPart(scene, FULL, gridSize=65, maxError_m=0.5)
    assert coarse.triangles.shape[0] < exact.triangles.shape[0]


def testBuildings_roofAtRoofHeightAndWallsColored(scene: Scene, neutral: BuildingAppearance):
    buildings = scene.models.buildings
    roofs, walls = scenery.buildingParts(scene, np.array([0]), FULL, neutral)
    # Flat in ellipsoidal height, so corners differ by Earth curvature (mm) in ENU Up.
    lon, lat = np.asarray(buildings.geometries[0].exterior.coords).T
    expectedU = scene.geometry.frame.geodeticToEnu(lat, lon, buildings.roofOrthometric_m[0] + 8.0)[
        2
    ]
    assert roofs.positions_ENU_m[:, 2].max() == pytest.approx(float(expectedU.max()), abs=1e-6)
    assert roofs.positions_ENU_m[:, 2].min() == pytest.approx(float(expectedU.min()), abs=1e-6)
    assert roofs.uv is not None and walls.colors is not None and walls.uv is not None
    assert walls.uv.shape == (walls.positions_ENU_m.shape[0], 2)
    assert walls.uv[:, 0].max() > 1.0  # physical window bays repeat around the facade
    assert walls.uv[:, 1].max() == pytest.approx(10.0)  # 30 m / 3 m floors
    # Rectangle: 2 roof triangles, 4 walls x 2 triangles.
    assert roofs.triangles.shape[0] == 2 and walls.triangles.shape[0] == 8
    p = roofs.positions_ENU_m[roofs.triangles]
    assert (np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])[:, 2] > 0).all()


def testDecks_slabFollowsProfile(scene: Scene, plane: SyntheticScene):
    tops, sides = scenery.deckParts(scene, np.array([0]), FULL)
    assert not tops.isEmpty and not sides.isEmpty
    e, n, u = tops.positions_ENU_m.T
    # Top = ground at the vertex + the end-tapered profile: the synthetic
    # bridge runs E-W at n = 200 from e = -400 to 400 (800 m, free ends).
    from map3d.layers.decks import END_TAPER_GRADE, END_RESIDUAL_m, ROAD_LEVEL_HEIGHT_m

    length_m = 800.0
    taper = (ROAD_LEVEL_HEIGHT_m - END_RESIDUAL_m) / END_TAPER_GRADE
    d = np.clip(e + 400.0, 0.0, length_m)  # the bridge runs E-W in ENU, e = -400..400
    above = END_RESIDUAL_m + (ROAD_LEVEL_HEIGHT_m - END_RESIDUAL_m) * np.clip(
        np.minimum(d, length_m - d) / taper, 0.0, 1.0
    )
    _, _, uRef = scene.geometry.frame.columnPointAtHeight(
        e, n, plane.terrainHeight_m(e, n) + above + 8.0
    )
    np.testing.assert_allclose(u, uRef, atol=0.05)
    assert sides.positions_ENU_m[:, 2].min() < u.min()  # underside is below the top


def testEmptySelections_giveEmptyParts(scene: Scene, neutral: BuildingAppearance):
    empty = np.array([], dtype=np.int64)
    assert all(p.isEmpty for p in scenery.buildingParts(scene, empty, FULL, neutral))
    assert all(p.isEmpty for p in scenery.deckParts(scene, empty, FULL))


def testTexture_conceptualIsPng(scene: Scene, neutral: BuildingAppearance):
    png = scenery.renderTexture(scene, FULL, neutral, size=64)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize(
    ("e", "n", "name"),
    [
        (0.0, 380.0, "water"),
        (100.0, -390.0, "grass"),
        (-160.0, -100.0, "roof"),
        (-350.0, -350.0, "underTrees"),  # canopy patch: the ground under the crowns
        (0.0, -450.0, "road"),  # ground-level road (centre line)
        (0.0, 200.0, "road"),  # bridges burn as the roads they carry
    ],
)
def testTexture_conceptualColours(
    scene: Scene, neutral: BuildingAppearance, e: float, n: float, name: str
):
    extent = FULL
    size = 500  # 2 m pixels
    rgb = decodePng(scenery.renderTexture(scene, extent, neutral, size=size))
    assert rgb.shape == (size, size, 3)
    gx, gy = scenery.enuToGrid(scene.geometry, np.array(e), np.array(n))
    col = int((gx - extent.x0) / (extent.x1 - extent.x0) * size)
    row = int((extent.y1 - gy) / (extent.y1 - extent.y0) * size)
    np.testing.assert_array_equal(rgb[row, col], scenery.CONCEPT_COLORS[name])


def testTexture_imageryDerivedBuildingRoofColour(scene: Scene):
    appearance = BuildingAppearance.neutral(scene.models.buildings)
    appearance.roofColors[0] = [160, 100, 80]
    rgb = decodePng(scenery.renderTexture(scene, FULL, appearance, size=500))
    gx, gy = scenery.enuToGrid(scene.geometry, np.array(-160.0), np.array(-100.0))
    np.testing.assert_array_equal(rgb[int(500 - gy) // 2, int(gx + 500) // 2], [160, 100, 80])


def testTexture_conceptualSoftensLandCoverEdges(scene: Scene, neutral: BuildingAppearance):
    # Grass meets water at n = 350 m: the land-cover colours blend across the
    # 5 m source pixels instead of switching in a hard step.
    extent = scenery.TileExtent(-60.0, 60.0, 290.0, 410.0)
    rgb = decodePng(scenery.renderTexture(scene, extent, neutral, size=240)).astype(int)
    grass, water = (
        np.array(scenery.CONCEPT_COLORS["grass"]),
        np.array(scenery.CONCEPT_COLORS["water"]),
    )
    between = ~(np.all(rgb == grass, axis=-1) | np.all(rgb == water, axis=-1))
    assert between.mean() > 0.005


def testTexture_conceptualWithoutLandCoverUsesGridClasses(
    scene: Scene, neutral: BuildingAppearance
):
    import dataclasses

    models = dataclasses.replace(
        scene.models, landCover=dataclasses.replace(scene.models.landCover, window=None)
    )
    bare = dataclasses.replace(scene, models=models)
    rgb = decodePng(scenery.renderTexture(bare, FULL, neutral, size=100))
    gx, gy = scenery.enuToGrid(scene.geometry, np.array(100.0), np.array(-390.0))
    col, row = int((gx + 500) / 10), int((500 - gy) / 10)
    np.testing.assert_array_equal(rgb[row, col], scenery.CONCEPT_COLORS["grass"])


def testDecks_piersGroundTheSlab(scene: Scene, plane: SyntheticScene):
    """Centreline decks get piers from the ground up to the slab underside."""
    import shapely

    from map3d.mesh.scenery import PIER_COLOR, PIER_END_MARGIN_m, PIER_SPACING_m

    tops, sides = scenery.deckParts(scene, np.array([0]), FULL)
    pierVertices = (sides.colors == np.array(PIER_COLOR, dtype=np.uint8)).all(axis=1)
    nPierQuads = int(pierVertices.sum()) // 4
    d = scene.models.decks
    line = d.centerlines[0]
    length_m = shapely.length(line) * scene.geometry.gridSpacing_m
    positions = np.arange(PIER_END_MARGIN_m, length_m - PIER_END_MARGIN_m, PIER_SPACING_m)
    taper = (7.0 - 0.5) / 0.08  # END_RESIDUAL_m / END_TAPER_GRADE
    topAbove = 0.5 + 6.5 * np.clip(np.minimum(positions, length_m - positions) / taper, 0, 1)
    expected = int((topAbove - 1.5 > 3.0).sum())  # PIER_CLEARANCE_m, DECK_THICKNESS_m
    assert nPierQuads == expected * 4  # four side quads per pier
    assert expected > 10  # the middle of the slab is well clear of the ground
    # pier feet sit on the terrain at their position
    pe, pn, pu = sides.positions_ENU_m[pierVertices].T
    pu = pu.reshape(-1, 4)  # per quad: two feet, two tops
    fe = pe.reshape(-1, 4)[:, :2].ravel()
    fn = pn.reshape(-1, 4)[:, :2].ravel()
    feet = pu[:, :2].ravel()
    _, _, uGround = scene.geometry.frame.columnPointAtHeight(
        fe, fn, plane.terrainHeight_m(fe, fn) + 8.0
    )
    np.testing.assert_allclose(feet, uGround, atol=1.0)
    # every pier spans its slab underside: height = clearance (3 m .. 5.5 m)
    pierHeight = (pu[:, 2:] - pu[:, :2]).mean(axis=1)
    assert (pierHeight > 3.0).all() and (pierHeight <= 5.9).all()  # + 2x PIER_EMBED_m


def testDecks_slabClippedToTileKeepsUVsInRange(scene: Scene):
    """A slab longer than the tile clips to it: UVs stay inside [0, 1]."""
    extent = scenery.TileExtent(0.0, 256.0, 128.0, 384.0)  # half of the bridge (n = 200)
    tops, sides = scenery.deckParts(scene, np.array([0]), extent)
    assert not tops.isEmpty
    u, v = tops.uv.T
    assert (u >= -1e-6).all() and (u <= 1 + 1e-6).all()
    assert (v >= -1e-6).all() and (v <= 1 + 1e-6).all()
    # tile extents are in grid metres (this scene is rotated 20 degrees)
    e, n, _ = tops.positions_ENU_m.T
    gx, gy = scenery.enuToGrid(scene.geometry, e, n)
    assert (gx >= extent.x0 - 1e-6).all() and (gx <= extent.x1 + 1e-6).all()
    assert (gy >= extent.y0 - 1e-6).all() and (gy <= extent.y1 + 1e-6).all()

    # a tile the deck does not touch yields no deck geometry
    farAway = scenery.TileExtent(-500.0, -480.0, -500.0, -480.0)
    tops2, sides2 = scenery.deckParts(scene, np.array([0]), farAway)
    assert tops2.isEmpty and sides2.isEmpty
