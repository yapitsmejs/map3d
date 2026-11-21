"""Tests for ``map3d.layers.decks`` (elevated structures)."""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import shapely

from map3d.gridGeometry import GridGeometry
from map3d.layers import decks as dk
from map3d.layers.columns import ChunkColumns
from map3d.layers.pixels import toPixelGeometries
from map3d.layers.terrain import GroundModel
from map3d.pack.geoparquet import writeGeoParquet
from map3d.pack.ingest import ingestFiles
from map3d.pack.manifest import Manifest
from map3d.pack.reader import Pack
from map3d.pack.synthetic import SyntheticScene, makeSyntheticPack
from map3d.request import SceneRequest


@pytest.fixture(scope="module")
def plane(tmp_path_factory: pytest.TempPathFactory) -> SyntheticScene:
    return makeSyntheticPack(tmp_path_factory.mktemp("d") / "plane.pack")


def _geometry(
    scene: SyntheticScene, s: float = 2.0, heading: float = 0.0, swath: float = 1000.0
) -> GridGeometry:
    h = math.radians(heading)
    axes = [[math.cos(h), -math.sin(h), 0.0], [math.sin(h), math.cos(h), 0.0], [0.0, 0.0, 1.0]]
    return GridGeometry.fromRequest(
        SceneRequest((scene.frame.lat0_deg, scene.frame.lon0_deg), s, axes, swath, swath)
    )


def _load(scene: SyntheticScene, geom: GridGeometry, packDir: Path | None = None) -> dk.DeckSet:
    pack = Pack.open(packDir or scene.packDir)
    return dk.loadDecks(pack, geom, GroundModel.load(pack, geom))


def _ground(scene: SyntheticScene, geom: GridGeometry, packDir: Path | None = None) -> GroundModel:
    return GroundModel.load(Pack.open(packDir or scene.packDir), geom)


def _raster(
    decks: dk.DeckSet,
    geom: GridGeometry,
    ground: GroundModel,
    maxLevels: int = 5,
    rows: slice | None = None,
    cols: slice | None = None,
) -> dk.DeckChunk:
    return dk.rasterizeDecks(
        decks,
        ChunkColumns(
            geom,
            rows if rows is not None else slice(None),
            cols if cols is not None else slice(None),
        ),
        ground,
        maxLevels,
    )


def _rc(geom: GridGeometry, e: float, n: float) -> tuple[int, int]:
    ee, nn = geom.nodesEnu()
    idx = np.unravel_index(np.argmin((ee - e) ** 2 + (nn - n) ** 2), ee.shape)
    return int(idx[0]), int(idx[1])


def _nodeE(geom: GridGeometry, r: int, c: int) -> float:
    return float(geom.nodesEnu()[0][r, c])


def _nodeN(geom: GridGeometry, r: int, c: int) -> float:
    return float(geom.nodesEnu()[1][r, c])


def _plane(e: float, n: float) -> float:
    return 10.0 + 0.02 * e - 0.01 * n


def _enuLine(
    scene: SyntheticScene, e0: float, n0: float, e1: float, n1: float
) -> shapely.LineString:
    t = np.linspace(0, 1, 9)
    lat, lon, _ = scene.frame.columnPointAtHeight(e0 + t * (e1 - e0), n0 + t * (n1 - n0), 0.0)
    return shapely.LineString(np.column_stack([lon, lat]))


def _packWithTransport(scene: SyntheticScene, tmp_path: Path, rows: list[tuple]) -> Path:
    pack = tmp_path / "t.pack"
    shutil.copytree(scene.packDir, pack)
    manifest = Manifest.load(pack)
    del manifest.sources["syntheticTransport"], manifest.sources["syntheticElevatedStructures"]
    manifest.save(pack)
    writeGeoParquet(
        tmp_path / "t.parquet",
        np.array([r[0] for r in rows], dtype=object),
        {
            "sourceFeatureId": [f"r{i}" for i in range(len(rows))],
            "kind": [r[1] for r in rows],
            "isBridge": [r[2] for r in rows],
            "level": [r[3] for r in rows],
            "width_m": [r[4] for r in rows],
        },
    )
    ingestFiles(pack, "overtureTransportation", [tmp_path / "t.parquet"])
    return pack


def _packWithOutlines(
    scene: SyntheticScene, tmp_path: Path, rows: list[tuple], outlines: list[shapely.Polygon]
) -> Path:
    """A pack with custom transport segments and elevated-structure outlines."""
    pack = _packWithTransport(scene, tmp_path, rows)
    writeGeoParquet(
        tmp_path / "o.parquet",
        np.array(outlines, dtype=object),
        {
            "sourceFeatureId": [f"o{i}" for i in range(len(outlines))],
            "structureType": ["Flyover"] * len(outlines),
        },
    )
    ingestFiles(pack, "syntheticElevatedStructures", [tmp_path / "o.parquet"])
    return pack


def _enuBox(scene: SyntheticScene, e0: float, n0: float, e1: float, n1: float) -> shapely.Polygon:
    lat, lon, _ = scene.frame.columnPointAtHeight([e0, e1, e1, e0], [n0, n0, n1, n1], 0.0)
    return shapely.Polygon(np.column_stack([lon, lat]))


def _pixelPoint(scene: SyntheticScene, geom: GridGeometry, e: float, n: float) -> Any:
    lat, lon, _ = scene.frame.columnPointAtHeight(e, n, 0.0)
    return toPixelGeometries(np.array([shapely.points(lon, lat)], dtype=object), 0.0, geom)[0]


def testLoad_syntheticBridgeDedupesItsOutline(plane: SyntheticScene):
    decks = _load(plane, _geometry(plane))
    assert len(decks) == 1
    assert list(decks.sourceIds) == ["syntheticTransport"]
    assert decks.levelHeight_m[0] == dk.ROAD_LEVEL_HEIGHT_m
    table = decks.featureTable(firstFeatureId=4)
    assert table["featureId"].tolist() == [4]
    assert table["kind"].tolist() == ["deck"]
    assert table["heightSource"].tolist() == ["DECK_ESTIMATED"]
    assert table["deckLevel"].tolist() == [1]


def testRasterize_topFollowsGroundAlongCenterlinePlusLevel(plane: SyntheticScene):
    geom = _geometry(plane)
    decks = _load(plane, geom)
    ground = _ground(plane, geom)
    chunk = _raster(decks, geom, ground)
    assert chunk.tops.shape == (5, 500, 500)
    r, c = _rc(geom, 0.0, 200.0)
    # The deck follows the ground sampled at the node (the bridge midpoint is
    # past the end tapers), plus the full road level height.
    expected = _plane(_nodeE(geom, r, c), _nodeN(geom, r, c)) + dk.ROAD_LEVEL_HEIGHT_m
    assert chunk.tops[0, r, c] == pytest.approx(expected, abs=1e-3)
    assert chunk.bottoms[0, r, c] == pytest.approx(expected - dk.DECK_THICKNESS_m, abs=1e-3)
    assert np.isnan(chunk.tops[1, r, c])
    assert chunk.maxTop[r, c] == chunk.tops[0, r, c]
    assert chunk.maxDeck[r, c] == 1
    r2, c2 = _rc(geom, 0.0, 212.0)
    assert np.isnan(chunk.maxTop[r2, c2]) and chunk.maxDeck[r2, c2] == 0
    assert chunk.overflowNodes == 0


def testRasterize_interchangeStacksLevelsMergesParallelsAndCountsOverflow(
    plane: SyntheticScene, tmp_path: Path
):
    rows = [
        (_enuLine(plane, -300, 0, 300, 0), "road", True, 1, 10.0),  # A: E-W level 1
        (
            _enuLine(plane, -300, 4, 300, 4),
            "road",
            True,
            1,
            10.0,
        ),  # A': parallel carriageway, overlaps A
        (_enuLine(plane, 0, -300, 0, 300), "road", True, 2, 10.0),  # B: N-S level 2
        (_enuLine(plane, -300, -100, 300, -100), "road", False, 0, 7.0),  # ground road: not a deck
    ]
    pack = _packWithTransport(plane, tmp_path, rows)
    geom = _geometry(plane)
    decks = _load(plane, geom, pack)
    assert len(decks) == 3
    chunk = _raster(decks, geom, _ground(plane, geom, pack))
    r, c = _rc(geom, 0.0, 2.0)
    e = _nodeE(geom, r, c)
    # Level 1 is A/A' merged (tops along n = 0 and n = 4 differ by 0.04 m); level 2 is B along e = 0.
    assert chunk.tops[0, r, c] == pytest.approx(_plane(e, _nodeN(geom, r, c)) + 7.0, abs=0.05)
    nNode = float(geom.nodesEnu()[1][r, c])
    assert chunk.tops[1, r, c] == pytest.approx(_plane(_nodeE(geom, r, c), nNode) + 14.0, abs=1e-3)
    assert np.isnan(chunk.tops[2, r, c])  # A and A' merged into one level
    assert decks.sourceFeatureIds[chunk.maxDeck[r, c] - 1] == "r2"
    one = _raster(decks, geom, _ground(plane, geom, pack), maxLevels=1)
    assert one.tops.shape[0] == 1
    assert one.tops[0, r, c] == pytest.approx(chunk.tops[0, r, c])
    assert one.maxTop[r, c] == pytest.approx(
        chunk.tops[1, r, c]
    )  # top deck still reaches the surface
    assert one.overflowNodes > 0


def testRasterize_railUsesViaductHeightAndLevelZeroCountsAsOne(
    plane: SyntheticScene, tmp_path: Path
):
    pack = _packWithTransport(
        plane, tmp_path, [(_enuLine(plane, -300, 0, 300, 0), "rail", True, 0, None)]
    )
    geom = _geometry(plane)
    decks = _load(plane, geom, pack)
    assert decks.levelHeight_m.tolist() == [dk.RAIL_LEVEL_HEIGHT_m]
    chunk = _raster(decks, geom, _ground(plane, geom, pack), maxLevels=2)
    r, c = _rc(geom, 0.0, 0.0)
    assert chunk.tops[0, r, c] == pytest.approx(
        _plane(_nodeE(geom, r, c), _nodeN(geom, r, c)) + dk.RAIL_LEVEL_HEIGHT_m, abs=1e-3
    )
    r2, c2 = _rc(geom, 0.0, dk.DEFAULT_DECK_WIDTH_m["rail"] / 2 + 3.0)
    assert np.isnan(chunk.maxTop[r2, c2])


def testOutlineOnlyStructure_usesMedianOutlineGround(plane: SyntheticScene, tmp_path: Path):
    pack = tmp_path / "o.pack"
    shutil.copytree(plane.packDir, pack)
    manifest = Manifest.load(pack)
    del manifest.sources["syntheticTransport"]
    manifest.save(pack)
    decks = _load(plane, _geometry(plane), pack)
    assert list(decks.sourceIds) == ["syntheticElevatedStructures"]
    assert decks.centerlines[0] is None
    geom = _geometry(plane)
    chunk = _raster(decks, geom, _ground(plane, geom, pack), maxLevels=2)
    r, c = _rc(geom, 300.0, 200.0)
    # Outline of [-400, 400] x [194, 206] on the plane: median ground = plane(0, 200) = 8 m.
    assert chunk.tops[0, r, c] == pytest.approx(8.0 + 7.0, abs=0.1)


def testRasterize_chunkedEqualsWhole(plane: SyntheticScene, tmp_path: Path):
    rows = [
        (_enuLine(plane, -300, 0, 300, 0), "road", True, 1, 10.0),
        (_enuLine(plane, 0, -300, 0, 300), "road", True, 2, 10.0),
    ]
    pack = _packWithTransport(plane, tmp_path, rows)
    geom = _geometry(plane, s=4.0, heading=30.0)
    decks = _load(plane, geom, pack)
    ground = _ground(plane, geom, pack)
    whole = _raster(decks, geom, ground, maxLevels=3)
    tops = np.full(whole.tops.shape, np.nan)
    for rows_, cols in geom.iterChunks(37, 51):
        part = _raster(decks, geom, ground, maxLevels=3, rows=rows_, cols=cols)
        tops[:, rows_, cols] = part.tops
    np.testing.assert_array_equal(tops, whole.tops)


def testNoDeckSources_emptySetAndNanChunk(plane: SyntheticScene, tmp_path: Path):
    pack = tmp_path / "n.pack"
    shutil.copytree(plane.packDir, pack)
    manifest = Manifest.load(pack)
    del manifest.sources["syntheticTransport"], manifest.sources["syntheticElevatedStructures"]
    manifest.save(pack)
    geom = _geometry(plane, s=10.0)
    decks = _load(plane, geom, pack)
    assert len(decks) == 0
    chunk = _raster(decks, geom, _ground(plane, geom, pack), maxLevels=0)
    assert chunk.tops.shape == (0, 100, 100)
    assert np.isnan(chunk.maxTop).all() and chunk.maxDeck.max() == 0


def testFootbridge_narrowAndLower(plane: SyntheticScene, tmp_path: Path):
    pack = _packWithTransport(
        plane, tmp_path, [(_enuLine(plane, -300, 0, 300, 0), "footway", True, 1, None)]
    )
    geom = _geometry(plane)
    decks = _load(plane, geom, pack)
    assert decks.levelHeight_m.tolist() == [dk.FOOTBRIDGE_LEVEL_HEIGHT_m]
    chunk = _raster(decks, geom, _ground(plane, geom, pack), maxLevels=1)
    r, c = _rc(geom, 0.0, 0.0)
    assert chunk.tops[0, r, c] == pytest.approx(
        _plane(_nodeE(geom, r, c), _nodeN(geom, r, c)) + 6.0, abs=1e-3
    )
    assert np.isnan(chunk.maxTop[_rc(geom, 0.0, 5.0)])  # 3 m wide: 5 m off is outside


def testRasterize_topFollowsGroundAlongCenterline_overHill(tmp_path: Path):
    """A bridge over hilly terrain sits on the ground beneath it, not the ends' lerp."""
    hill = makeSyntheticPack(tmp_path / "hill.pack", terrain="hill")
    rows = [(_enuLine(hill, -400, 0, 400, 0), "road", True, 1, 8.0)]
    pack = _packWithOutlines(hill, tmp_path, rows, [])
    geom = _geometry(hill)
    ground = _ground(hill, geom, pack)
    decks = dk.loadDecks(Pack.open(pack), geom, ground)
    chunk = _raster(decks, geom, ground, maxLevels=2)
    r, c = _rc(geom, 0.0, 0.0)
    lat, lon = ChunkColumns(geom, slice(None), slice(None)).nodeLatLon()
    expected = ground.orthometricAt(lat, lon)[r, c] + dk.ROAD_LEVEL_HEIGHT_m
    assert chunk.tops[0, r, c] == pytest.approx(expected, abs=1e-6)
    # The ends' lerp would miss the 30 m hill by more than 15 m here.
    assert chunk.tops[0, r, c] > 35.0


def testRasterize_freeEndsTaperToGrade(plane: SyntheticScene, tmp_path: Path):
    """A bridge piece with free ends ramps down to END_RESIDUAL_m above the ground."""
    rows = [(_enuLine(plane, -300, 0, 300, 0), "road", True, 1, 8.0)]
    pack = _packWithTransport(plane, tmp_path, rows)
    geom = _geometry(plane)
    ground = _ground(plane, geom, pack)
    decks = dk.loadDecks(Pack.open(pack), geom, ground)
    taper = (dk.ROAD_LEVEL_HEIGHT_m - dk.END_RESIDUAL_m) / dk.END_TAPER_GRADE
    assert decks.taperStart_m.tolist() == [pytest.approx(taper)]
    assert decks.taperEnd_m.tolist() == [pytest.approx(taper)]
    chunk = _raster(decks, geom, ground)
    columns = ChunkColumns(geom, slice(None), slice(None))
    nodeGround = ground.orthometricAt(*columns.nodeLatLon())
    line = decks.centerlines[0]
    length_m = shapely.length(line) * geom.gridSpacing_m
    for e_m in (-270.0, -100.0, 0.0, 100.0, 270.0):
        r, c = _rc(geom, e_m, 0.0)
        frac = float(
            shapely.line_locate_point(line, shapely.points(c + 0.5, r + 0.5), normalized=True)
        )
        d = frac * length_m
        expectedAbove = dk.END_RESIDUAL_m + (dk.ROAD_LEVEL_HEIGHT_m - dk.END_RESIDUAL_m) * min(
            1.0, min(d, length_m - d) / taper
        )
        assert chunk.tops[0, r, c] == pytest.approx(nodeGround[r, c] + expectedAbove, abs=1e-6), (
            f"e={e_m}"
        )


def testLoad_joinedBridgeEndsDoNotTaper(plane: SyntheticScene, tmp_path: Path):
    """Ends shared with another bridge piece keep full height (no mid-flyover dip)."""
    rows = [
        (_enuLine(plane, -300, 0, 0, 0), "road", True, 1, 8.0),
        (_enuLine(plane, 0, 0, 300, 0), "road", True, 1, 8.0),
    ]
    pack = _packWithTransport(plane, tmp_path, rows)
    geom = _geometry(plane)
    ground = _ground(plane, geom, pack)
    decks = dk.loadDecks(Pack.open(pack), geom, ground)
    assert len(decks) == 2
    taper = (dk.ROAD_LEVEL_HEIGHT_m - dk.END_RESIDUAL_m) / dk.END_TAPER_GRADE
    assert decks.taperStart_m.tolist() == [pytest.approx(taper), 0.0]
    assert decks.taperEnd_m.tolist() == [0.0, pytest.approx(taper)]
    chunk = _raster(decks, geom, ground)
    r, c = _rc(geom, 0.0, 0.0)  # the joint
    assert chunk.maxTop[r, c] == pytest.approx(
        _plane(_nodeE(geom, r, c), _nodeN(geom, r, c)) + dk.ROAD_LEVEL_HEIGHT_m, abs=1e-6
    )


def testOutlineDeck_leavesAtGradeCarriagewaysAtGround(plane: SyntheticScene, tmp_path: Path):
    """An at-grade road inside an elevated-structure outline stays at grade."""
    rows = [(_enuLine(plane, -300, 0, 300, 0), "road", False, 0, 8.0)]
    pack = _packWithOutlines(plane, tmp_path, rows, [_enuBox(plane, -50, -50, 50, 50)])
    geom = _geometry(plane)
    ground = _ground(plane, geom, pack)
    decks = dk.loadDecks(Pack.open(pack), geom, ground)
    assert len(decks) == 1
    assert decks.centerlines[0] is None
    shape = decks.pixelShapes[0]
    assert not shapely.contains(shape, _pixelPoint(plane, geom, 0.0, 0.0))
    assert shapely.contains(shape, _pixelPoint(plane, geom, 0.0, 20.0))
    chunk = _raster(decks, geom, ground, maxLevels=2)
    r, c = _rc(geom, 0.0, 0.0)
    assert np.isnan(chunk.maxTop[r, c]) and chunk.maxDeck[r, c] == 0
    r2, c2 = _rc(geom, 0.0, 20.0)
    assert chunk.maxTop[r2, c2] == pytest.approx(_plane(0.0, 20.0) + 7.0, abs=0.5)


def testOutlineDeck_keepsDeckOverRoadUndercrossings(plane: SyntheticScene, tmp_path: Path):
    """An at-grade road crossing under a bridge keeps the outline deck there."""
    rows = [
        (_enuLine(plane, -300, 0, 300, 0), "road", False, 0, 8.0),
        (_enuLine(plane, 0, -300, 0, 300), "road", True, 1, 8.0),
    ]
    pack = _packWithOutlines(plane, tmp_path, rows, [_enuBox(plane, -50, -50, 50, 50)])
    geom = _geometry(plane)
    decks = _load(plane, geom, pack)
    outlines = [i for i in range(len(decks)) if decks.centerlines[i] is None]
    assert len(outlines) == 1
    shape = decks.pixelShapes[outlines[0]]
    assert shapely.contains(shape, _pixelPoint(plane, geom, 0.0, 0.0))
    assert not shapely.contains(shape, _pixelPoint(plane, geom, 20.0, 0.0))


def testRasterize_shortBridgeReachesFullHeight(plane: SyntheticScene, tmp_path: Path):
    """The end taper never exceeds half the piece: short bridges keep their height."""
    rows = [(_enuLine(plane, -50, 0, 50, 0), "road", True, 1, 8.0)]  # 100 m piece
    pack = _packWithTransport(plane, tmp_path, rows)
    geom = _geometry(plane)
    ground = _ground(plane, geom, pack)
    decks = dk.loadDecks(Pack.open(pack), geom, ground)
    taper = (dk.ROAD_LEVEL_HEIGHT_m - dk.END_RESIDUAL_m) / dk.END_TAPER_GRADE
    assert decks.taperStart_m.tolist() == [pytest.approx(50.0)]  # capped at half length
    assert decks.taperStart_m[0] < taper
    chunk = _raster(decks, geom, ground)
    r, c = _rc(geom, 0.0, 0.0)
    # full height (the node may sit up to 1 m down the ramp from the exact middle)
    assert chunk.maxTop[r, c] > _plane(_nodeE(geom, r, c), _nodeN(geom, r, c)) + 6.5
    # the uncapped 81 m taper would have peaked at only ~4.5 m


def testLoad_joinedToElevatedWayKeepsFullHeight(plane: SyntheticScene, tmp_path: Path):
    """A bridge end coinciding with an elevated (non-bridge, level 1) way stays full."""
    rows = [
        (_enuLine(plane, -300, 0, 0, 0), "road", True, 1, 8.0),
        (_enuLine(plane, 0, 0, 300, 0), "road", False, 1, 8.0),
    ]
    pack = _packWithTransport(plane, tmp_path, rows)
    geom = _geometry(plane)
    decks = _load(plane, geom, pack)
    assert len(decks) == 1
    taper = (dk.ROAD_LEVEL_HEIGHT_m - dk.END_RESIDUAL_m) / dk.END_TAPER_GRADE
    assert decks.taperStart_m.tolist() == [pytest.approx(taper)]  # free start tapers
    assert decks.taperEnd_m.tolist() == [0.0]  # joined to the elevated way


def testCorridorWidths_mappedWidthElsePerKindDefault():
    kinds = np.array(["road", "rail", "footway", "cycleway", None], dtype=object)
    widths = np.array([None, None, None, None, 12.5], dtype=object)
    assert dk.corridorWidths_m(kinds, widths).tolist() == [8.0, 8.0, 3.0, 8.0, 12.5]
