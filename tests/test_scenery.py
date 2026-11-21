"""Tests for ``map3d.mesh.scenery`` (terrain / building / deck meshes and textures)."""

from __future__ import annotations

import shutil

import numpy as np
import pytest
import shapely
from _support import decodePng, sceneOver

from map3d.mesh import scenery
from map3d.mesh.buildingAppearance import BuildingAppearance, FacadeStyle
from map3d.pack.geoparquet import writeGeoParquet
from map3d.pack.ingest import ingestFiles
from map3d.pack.synthetic import SyntheticScene, makeSyntheticPack
from map3d.request import SceneRequest
from map3d.scene import Scene, buildScene


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
    gx, gy = scene.geometry.enuToGrid(np.array(e), np.array(n))
    col = int((gx - extent.x0) / (extent.x1 - extent.x0) * size)
    row = int((extent.y1 - gy) / (extent.y1 - extent.y0) * size)
    np.testing.assert_array_equal(rgb[row, col], scenery.CONCEPT_COLORS[name])


def testTexture_imageryDerivedBuildingRoofColour(scene: Scene):
    appearance = BuildingAppearance.neutral(scene.models.buildings)
    appearance.roofColors[0] = [160, 100, 80]
    rgb = decodePng(scenery.renderTexture(scene, FULL, appearance, size=500))
    gx, gy = scene.geometry.enuToGrid(np.array(-160.0), np.array(-100.0))
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
    gx, gy = scene.geometry.enuToGrid(np.array(100.0), np.array(-390.0))
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


def testDecks_piersBelongToTheTileTheyStandIn(scene: Scene):
    """On a rotated grid, each pier is drawn once, by the tile (grid metres) it stands in."""
    from map3d.mesh.scenery import PIER_COLOR

    def pierCentresGrid(extent: scenery.TileExtent) -> np.ndarray:
        _, sides = scenery.deckParts(scene, np.array([0]), extent)
        isPier = (sides.colors == np.array(PIER_COLOR, dtype=np.uint8)).all(axis=1)
        e, n = sides.positions_ENU_m[isPier, :2].reshape(-1, 16, 2).mean(axis=1).T  # 4 quads
        return np.column_stack(scene.geometry.enuToGrid(e, n))

    total = pierCentresGrid(FULL).shape[0]
    found = 0
    for x0, y0 in [(-500.0, -500.0), (0.0, -500.0), (-500.0, 0.0), (0.0, 0.0)]:
        tile = scenery.TileExtent(x0, x0 + 500.0, y0, y0 + 500.0)
        centres = pierCentresGrid(tile)
        assert ((centres[:, 0] >= x0) & (centres[:, 0] <= x0 + 500.0)).all()
        assert ((centres[:, 1] >= y0) & (centres[:, 1] <= y0 + 500.0)).all()
        found += centres.shape[0]
    assert total > 10 and found == total


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
    gx, gy = scene.geometry.enuToGrid(e, n)
    assert (gx >= extent.x0 - 1e-6).all() and (gx <= extent.x1 + 1e-6).all()
    assert (gy >= extent.y0 - 1e-6).all() and (gy <= extent.y1 + 1e-6).all()

    # a tile the deck does not touch yields no deck geometry
    farAway = scenery.TileExtent(-500.0, -480.0, -500.0, -480.0)
    tops2, sides2 = scenery.deckParts(scene, np.array([0]), farAway)
    assert tops2.isEmpty and sides2.isEmpty


@pytest.fixture(scope="module")
def shaped(plane: SyntheticScene, tmp_path_factory: pytest.TempPathFactory) -> Scene:
    """A domed hall, a gabled house and a raised walkway on the synthetic plane."""
    tmp = tmp_path_factory.mktemp("shaped")
    pack = tmp / "shaped.pack"
    shutil.copytree(plane.packDir, pack)

    def lonLat(poly: shapely.Polygon) -> shapely.Polygon:
        e, n = shapely.get_coordinates(poly.exterior).T
        lat, lon, _ = plane.frame.columnPointAtHeight(e, n, 0.0)
        return shapely.Polygon(np.column_stack([lon, lat]))

    geoms = [
        lonLat(shapely.Point(-300.0, 360.0).buffer(20.0, quad_segs=16)),
        lonLat(shapely.box(-250.0, 340.0, -230.0, 350.0)),
        lonLat(shapely.box(-380.0, 300.0, -340.0, 304.0)),
    ]
    writeGeoParquet(
        tmp / "o.parquet",
        np.array(geoms, dtype=object),
        {
            "sourceFeatureId": ["hall", "house", "walkway"],
            "height_m": [30.0, 8.0, 15.0],
            "numFloors": [None, None, None],
            "minHeight_m": [None, None, 12.0],
            "roofShape": ["dome", "gabled", None],
            "roofHeight_m": [None, 3.0, None],
        },
    )
    ingestFiles(pack, "overtureBuildings", [tmp / "o.parquet"])
    axes = np.eye(3)
    centre = (plane.frame.lat0_deg, plane.frame.lon0_deg)
    return buildScene(SceneRequest(centre, 5.0, axes, 1000.0, 1000.0), pack)


def _index(scene: Scene, featureId: str) -> int:
    return list(scene.models.buildings.sourceFeatureIds).index(featureId)


def _ellToUp(scene: Scene, i: int, orthometric: float) -> float:
    b = scene.models.buildings
    c = shapely.centroid(b.geometries[i])
    lon, lat = float(shapely.get_x(c)), float(shapely.get_y(c))
    und = float(scene.models.ground.undulation_m(np.array([lat]), np.array([lon]))[0])
    return float(scene.geometry.frame.geodeticToEnu(lat, lon, orthometric + und)[2])


def testShapedRoof_domeIsSmoothAndReachesTheTop(shaped: Scene):
    i = _index(shaped, "hall")
    b = shaped.models.buildings
    roofs, walls = scenery.buildingParts(shaped, np.array([i]), FULL, BuildingAppearance.neutral(b))
    top = roofs.positions_ENU_m[:, 2].max()
    assert top == pytest.approx(_ellToUp(shaped, i, b.roofOrthometric_m[i]), abs=0.05)
    # Shared vertices (smooth shading): far fewer vertices than triangle corners.
    assert roofs.positions_ENU_m.shape[0] < roofs.triangles.shape[0]
    p = roofs.positions_ENU_m[roofs.triangles]
    assert (np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])[:, 2] > 0).all()
    # Walls stop at the eaves; the dome sits on them.
    eave = _ellToUp(shaped, i, b.eaveOrthometric_m[i])
    assert walls.positions_ENU_m[:, 2].max() == pytest.approx(eave, abs=0.05)


def testShapedRoof_gableWallsRiseToTheRidge(shaped: Scene):
    i = _index(shaped, "house")
    b = shaped.models.buildings
    roofs, walls = scenery.buildingParts(shaped, np.array([i]), FULL, BuildingAppearance.neutral(b))
    ridge = _ellToUp(shaped, i, b.roofOrthometric_m[i])
    assert roofs.positions_ENU_m[:, 2].max() == pytest.approx(ridge, abs=0.02)
    assert walls.positions_ENU_m[:, 2].max() == pytest.approx(ridge, abs=0.02)
    # Faceted: every triangle keeps its own corners (sharp ridge).
    assert roofs.positions_ENU_m.shape[0] == 3 * roofs.triangles.shape[0]
    assert walls.uv is not None and np.isfinite(walls.uv).all()


def testRaisedPart_hasAnUndersideAndWallsFromItsMinHeight(shaped: Scene):
    i = _index(shaped, "walkway")
    b = shaped.models.buildings
    roofs, walls = scenery.buildingParts(shaped, np.array([i]), FULL, BuildingAppearance.neutral(b))
    bottom = _ellToUp(shaped, i, b.bottomOrthometric_m[i])
    assert walls.positions_ENU_m[:, 2].min() == pytest.approx(bottom, abs=0.02)
    p = roofs.positions_ENU_m[roofs.triangles]
    nz = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])[:, 2]
    assert (nz > 0).sum() == (nz < 0).sum() == 2  # a roof and an underside
    assert p[nz < 0][..., 2].max() == pytest.approx(bottom, abs=0.02)


def testTexture_mappedWaterIsCrispAndUnmappedRasterWaterIsNot(
    plane: SyntheticScene, tmp_path_factory: pytest.TempPathFactory
):
    tmp = tmp_path_factory.mktemp("water")
    pack = tmp / "water.pack"
    shutil.copytree(plane.packDir, pack)
    e, n = shapely.get_coordinates(shapely.Point(100.0, 100.0).buffer(30.0).exterior).T
    lat, lon, _ = plane.frame.columnPointAtHeight(e, n, 0.0)
    writeGeoParquet(
        tmp / "w.parquet",
        np.array([shapely.Polygon(np.column_stack([lon, lat]))], dtype=object),
        {"sourceFeatureId": ["pond"], "landClass": [80]},
    )
    ingestFiles(pack, "overtureWater", [tmp / "w.parquet"])
    centre = (plane.frame.lat0_deg, plane.frame.lon0_deg)
    scene = buildScene(SceneRequest(centre, 5.0, np.eye(3), 1000.0, 1000.0), pack)
    size = 500  # 2 m pixels
    rgb = decodePng(
        scenery.renderTexture(scene, FULL, BuildingAppearance.neutral(scene.models.buildings), size)
    )

    def pixel(e_m: float, n_m: float) -> np.ndarray:
        gx, gy = scene.geometry.enuToGrid(np.array(e_m), np.array(n_m))
        return rgb[int((FULL.y1 - gy) / 1000.0 * size), int((gx - FULL.x0) / 1000.0 * size)]

    water, grass = scenery.CONCEPT_COLORS["water"], scenery.CONCEPT_COLORS["grass"]
    np.testing.assert_array_equal(pixel(100.0, 100.0), water)
    np.testing.assert_array_equal(pixel(100.0, 128.0), water)  # 2 m inside the edge
    np.testing.assert_array_equal(pixel(100.0, 133.0), grass)  # 3 m outside: no blur
    assert not np.array_equal(pixel(0.0, 450.0), water)  # raster water without a polygon
    assert "overtureWater" in {s["sourceId"] for s in scene.metadata["sources"]}  # attributed


def testLoft_supertreeStackIsOneSmoothSurfaceWithACap(
    plane: SyntheticScene, tmp_path_factory: pytest.TempPathFactory
):
    tmp = tmp_path_factory.mktemp("loft")
    pack = tmp / "loft.pack"
    shutil.copytree(plane.packDir, pack)

    def disc(r: float) -> shapely.Polygon:
        e, n = shapely.get_coordinates(shapely.Point(-300.0, 360.0).buffer(r).exterior).T
        lat, lon, _ = plane.frame.columnPointAtHeight(e, n, 0.0)
        return shapely.Polygon(np.column_stack([lon, lat]))

    writeGeoParquet(
        tmp / "o.parquet",
        np.array([disc(3.0), disc(5.0), disc(7.0)], dtype=object),
        {
            "sourceFeatureId": ["trunk", "ring1", "ring2"],
            "height_m": [30.0, 33.0, 41.0],
            "numFloors": [None, None, None],
            "minHeight_m": [None, 30.0, 36.0],
            "facadeColor": ["#76C82E"] * 3,
        },
    )
    ingestFiles(pack, "overtureBuildings", [tmp / "o.parquet"])
    centre = (plane.frame.lat0_deg, plane.frame.lon0_deg)
    scene = buildScene(SceneRequest(centre, 5.0, np.eye(3), 1000.0, 1000.0), pack)
    b = scene.models.buildings
    ids = list(b.sourceFeatureIds)
    trunk, ring1, ring2 = (ids.index(k) for k in ("trunk", "ring1", "ring2"))
    assert [loft.members for loft in b.lofts] == [(trunk, ring1, ring2)]
    appearance = BuildingAppearance.fromScene(scene)
    assert appearance.facadeStyles is not None
    assert set(appearance.facadeStyles[[trunk, ring1, ring2]]) == {FacadeStyle.LATTICE}
    caps, side = scenery.buildingParts(scene, np.array([trunk, ring1, ring2]), FULL, appearance)
    # One loft: 4 profile rings x 37 ring vertices, shared (smooth); a fan cap on top.
    assert side.positions_ENU_m.shape[0] == 4 * 37
    assert side.triangles.shape[0] == 2 * 3 * 36
    assert caps.triangles.shape[0] == 36
    top = _ellToUp(scene, ring2, b.roofOrthometric_m[ring2])
    assert caps.positions_ENU_m[:, 2].max() == pytest.approx(top, abs=0.05)
    # Only the bottom part draws the loft; the rings alone draw nothing.
    alone = scenery.buildingParts(scene, np.array([ring1, ring2]), FULL, appearance)
    assert alone[0].isEmpty and alone[1].isEmpty


@pytest.fixture(scope="module")
def towers(plane: SyntheticScene, tmp_path_factory: pytest.TempPathFactory) -> Scene:
    """A round tower and a square block of 12 mapped storeys on the synthetic plane."""
    tmp = tmp_path_factory.mktemp("towers")
    pack = tmp / "towers.pack"
    shutil.copytree(plane.packDir, pack)

    def lonLat(poly: shapely.Polygon) -> shapely.Polygon:
        e, n = shapely.get_coordinates(poly.exterior).T
        lat, lon, _ = plane.frame.columnPointAtHeight(e, n, 0.0)
        return shapely.Polygon(np.column_stack([lon, lat]))

    writeGeoParquet(
        tmp / "o.parquet",
        np.array(
            [
                lonLat(shapely.Point(-300.0, 300.0).buffer(15.0, quad_segs=12)),
                lonLat(shapely.box(-250.0, 300.0, -230.0, 320.0)),
            ],
            dtype=object,
        ),
        {
            "sourceFeatureId": ["round", "block"],
            "height_m": [60.0, 30.0],
            "numFloors": [None, 12],
        },
    )
    ingestFiles(pack, "overtureBuildings", [tmp / "o.parquet"])
    centre = (plane.frame.lat0_deg, plane.frame.lon0_deg)
    return buildScene(SceneRequest(centre, 5.0, np.eye(3), 1000.0, 1000.0), pack)


def testWalls_roundTowerShadesSmoothlyWithContinuousBays(towers: Scene):
    i = _index(towers, "round")
    _, walls = scenery.buildingParts(
        towers, np.array([i]), FULL, BuildingAppearance.neutral(towers.models.buildings)
    )
    segments = 48  # quad_segs=12 -> 48 outline segments
    assert walls.triangles.shape[0] == 2 * segments
    # One run round the whole tower: 49 columns of shared bottom/top vertices.
    assert walls.positions_ENU_m.shape[0] == 2 * (segments + 1)
    u = walls.uv[: segments + 1, 0]
    assert np.all(np.diff(u) > 0) and u[-1] == pytest.approx(round(2 * np.pi * 15.0 / 3.0), abs=1)


def testWalls_floorBandsFollowMappedStoreys(towers: Scene):
    b = towers.models.buildings
    i = _index(towers, "block")
    _, walls = scenery.buildingParts(towers, np.array([i]), FULL, BuildingAppearance.neutral(b))
    assert b.floorHeight_m[i] == pytest.approx(2.5)
    # Square corners stay sharp: 4 runs of 2 columns each.
    assert walls.positions_ENU_m.shape[0] == 4 * 2 * 2
    wallHeight = b.roofOrthometric_m[i] - b.baseOrthometric_m[i]
    assert walls.uv[:, 1].max() == pytest.approx(wallHeight / 2.5, rel=1e-4)
    unmapped = _index(towers, "round")
    assert np.isnan(b.floorHeight_m[unmapped])
