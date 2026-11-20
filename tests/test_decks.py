"""Tests for ``map3d.layers.decks`` (elevated structures)."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import shapely

from map3d.gridGeometry import GridGeometry
from map3d.layers import decks as dk
from map3d.layers.columns import ChunkColumns
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


def _rc(geom: GridGeometry, e: float, n: float) -> tuple[int, int]:
    ee, nn = geom.nodesEnu()
    idx = np.unravel_index(np.argmin((ee - e) ** 2 + (nn - n) ** 2), ee.shape)
    return int(idx[0]), int(idx[1])


def _nodeE(geom: GridGeometry, r: int, c: int) -> float:
    return float(geom.nodesEnu()[0][r, c])


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


def testRasterize_topFollowsInterpolatedGroundPlusLevel(plane: SyntheticScene):
    geom = _geometry(plane)
    decks = _load(plane, geom)
    chunk = dk.rasterizeDecks(
        decks, geom, ChunkColumns(geom, slice(None), slice(None)), maxLevels=5
    )
    assert chunk.tops.shape == (5, 500, 500)
    r, c = _rc(geom, 0.0, 200.0)
    # Ends at (-400, 200) = 0 m and (400, 200) = 16 m: the lerp equals the plane along n = 200.
    expected = _plane(_nodeE(geom, r, c), 200.0) + dk.ROAD_LEVEL_HEIGHT_m
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
    chunk = dk.rasterizeDecks(
        decks, geom, ChunkColumns(geom, slice(None), slice(None)), maxLevels=5
    )
    r, c = _rc(geom, 0.0, 2.0)
    e = _nodeE(geom, r, c)
    # Level 1 is A/A' merged (tops along n = 0 and n = 4 differ by 0.04 m); level 2 is B along e = 0.
    assert chunk.tops[0, r, c] == pytest.approx(_plane(e, 0.0) + 7.0, abs=0.05)
    nNode = float(geom.nodesEnu()[1][r, c])
    assert chunk.tops[1, r, c] == pytest.approx(_plane(0.0, nNode) + 14.0, abs=1e-3)
    assert np.isnan(chunk.tops[2, r, c])  # A and A' merged into one level
    assert decks.sourceFeatureIds[chunk.maxDeck[r, c] - 1] == "r2"
    one = dk.rasterizeDecks(decks, geom, ChunkColumns(geom, slice(None), slice(None)), maxLevels=1)
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
    chunk = dk.rasterizeDecks(
        decks, geom, ChunkColumns(geom, slice(None), slice(None)), maxLevels=2
    )
    r, c = _rc(geom, 0.0, 0.0)
    assert chunk.tops[0, r, c] == pytest.approx(
        _plane(_nodeE(geom, r, c), 0.0) + dk.RAIL_LEVEL_HEIGHT_m, abs=1e-3
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
    chunk = dk.rasterizeDecks(
        decks, geom, ChunkColumns(geom, slice(None), slice(None)), maxLevels=2
    )
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
    whole = dk.rasterizeDecks(
        decks, geom, ChunkColumns(geom, slice(None), slice(None)), maxLevels=3
    )
    tops = np.full(whole.tops.shape, np.nan)
    for rows_, cols in geom.iterChunks(37, 51):
        part = dk.rasterizeDecks(decks, geom, ChunkColumns(geom, rows_, cols), maxLevels=3)
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
    chunk = dk.rasterizeDecks(
        decks, geom, ChunkColumns(geom, slice(None), slice(None)), maxLevels=0
    )
    assert chunk.tops.shape == (0, 100, 100)
    assert np.isnan(chunk.maxTop).all() and chunk.maxDeck.max() == 0


def testFootbridge_narrowAndLower(plane: SyntheticScene, tmp_path: Path):
    pack = _packWithTransport(
        plane, tmp_path, [(_enuLine(plane, -300, 0, 300, 0), "footway", True, 1, None)]
    )
    geom = _geometry(plane)
    decks = _load(plane, geom, pack)
    assert decks.levelHeight_m.tolist() == [dk.FOOTBRIDGE_LEVEL_HEIGHT_m]
    chunk = dk.rasterizeDecks(
        decks, geom, ChunkColumns(geom, slice(None), slice(None)), maxLevels=1
    )
    r, c = _rc(geom, 0.0, 0.0)
    assert chunk.tops[0, r, c] == pytest.approx(_plane(_nodeE(geom, r, c), 0.0) + 6.0, abs=1e-3)
    assert np.isnan(chunk.maxTop[_rc(geom, 0.0, 5.0)])  # 3 m wide: 5 m off is outside
