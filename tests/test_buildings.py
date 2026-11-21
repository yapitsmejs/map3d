"""Tests for ``map3d.layers.buildings`` (fusion, height precedence, rasterization)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import shapely
from _support import headingAxes

from map3d.gridGeometry import GridGeometry
from map3d.layers.buildings import (
    DEFAULT_BUILDING_HEIGHT_m,
    STOREY_HEIGHT_m,
    loadBuildings,
    rasterizeBuildings,
)
from map3d.layers.columns import ChunkColumns
from map3d.layers.provenance import HeightSource
from map3d.layers.terrain import GroundModel
from map3d.pack.geoparquet import writeGeoParquet
from map3d.pack.ingest import ingestFiles
from map3d.pack.reader import Pack
from map3d.pack.synthetic import SyntheticScene, makeSyntheticPack
from map3d.request import SceneRequest


@pytest.fixture(scope="module")
def plane(tmp_path_factory: pytest.TempPathFactory) -> SyntheticScene:
    return makeSyntheticPack(tmp_path_factory.mktemp("b") / "plane.pack", terrain="plane")


def _geometry(
    scene: SyntheticScene, heading_deg: float = 0.0, swath_m: float = 800.0, s: float = 2.0
):
    axes = headingAxes(heading_deg)
    req = SceneRequest((scene.frame.lat0_deg, scene.frame.lon0_deg), s, axes, swath_m, swath_m)
    return GridGeometry.fromRequest(req)


def _load(scene: SyntheticScene, geom: GridGeometry, packDir: Path | None = None):
    pack = Pack.open(packDir or scene.packDir)
    return loadBuildings(pack, geom, GroundModel.load(pack, geom))


def _copyPack(scene: SyntheticScene, tmp_path: Path) -> Path:
    dst = tmp_path / "copy.pack"
    shutil.copytree(scene.packDir, dst)
    return dst


def _enuBox(scene: SyntheticScene, e0: float, n0: float, e1: float, n1: float) -> shapely.Polygon:
    lat, lon, _ = scene.frame.columnPointAtHeight(
        np.array([e0, e1, e1, e0]), np.array([n0, n0, n1, n1]), 0.0
    )
    return shapely.Polygon(np.column_stack([lon, lat]))


def _writeBuildings(path: Path, geoms: list, heights: list, floors: list) -> Path:
    writeGeoParquet(
        path,
        np.array(geoms, dtype=object),
        {
            "sourceFeatureId": [f"x{i}" for i in range(len(geoms))],
            "height_m": heights,
            "numFloors": floors,
        },
    )
    return path


def testLoad_syntheticHeightsAndSources(plane: SyntheticScene):
    buildings = _load(plane, _geometry(plane))
    assert len(buildings) == 3
    np.testing.assert_allclose(buildings.height_m, [30.0, 12.0, 4 * STOREY_HEIGHT_m])
    assert list(buildings.heightSource) == [
        HeightSource.EXPLICIT_HEIGHT,
        HeightSource.EXPLICIT_HEIGHT,
        HeightSource.FLOORS,
    ]
    assert list(buildings.sourceFeatureIds) == ["synB0", "synB1", "synB2"]


def testLoad_baseIsLowestGroundUnderFootprint(plane: SyntheticScene):
    buildings = _load(plane, _geometry(plane))
    # Plane 10 + 0.02 e - 0.01 n over [-200, -120] x [-150, -50]: min at (-200, -50) = 6.5.
    assert buildings.baseOrthometric_m[0] == pytest.approx(6.5, abs=1e-3)
    assert buildings.roofOrthometric_m[0] == pytest.approx(36.5, abs=1e-3)


def testLoad_authoritativeSourceWinsAndDuplicatesDropped(plane: SyntheticScene, tmp_path: Path):
    pack = _copyPack(plane, tmp_path)
    b0 = plane.buildings[0]
    hdb = _enuBox(plane, b0.e0_m, b0.n0_m, b0.e1_m, b0.n1_m)
    ingestFiles(
        pack, "sgHdbBuildings", [_writeBuildings(tmp_path / "hdb.parquet", [hdb], [None], [10])]
    )
    buildings = _load(plane, _geometry(plane), pack)
    assert len(buildings) == 3  # synthetic B0 dropped as a duplicate of the HDB block
    assert buildings.sourceIds[0] == "sgHdbBuildings"
    assert buildings.height_m[0] == pytest.approx(10 * STOREY_HEIGHT_m)
    assert buildings.heightSource[0] == HeightSource.AUTHORITATIVE_FLOORS
    assert "synB0" not in list(buildings.sourceFeatureIds)


def testLoad_rasterHeightThenDefault(plane: SyntheticScene, tmp_path: Path, writeRaster):
    pack = _copyPack(plane, tmp_path)
    a = _enuBox(plane, -350.0, 250.0, -300.0, 300.0)
    b = _enuBox(plane, 250.0, 250.0, 300.0, 300.0)
    ingestFiles(
        pack,
        "overtureBuildings",
        [_writeBuildings(tmp_path / "o.parquet", [a, b], [None, None], [None, None])],
    )
    bounds = Pack.open(pack).bounds_deg
    raster = np.zeros((200, 200), dtype=np.float32)
    raster[:, :100] = 25.0  # west half only -> building a gets 25 m, b falls back
    ingestFiles(
        pack, "googleOpenBuildings25d", [writeRaster(tmp_path / "gob.tif", bounds, values=raster)]
    )
    buildings = _load(plane, _geometry(plane), pack)
    ids = list(buildings.sourceFeatureIds)
    ia, ib = ids.index("x0"), ids.index("x1")
    assert buildings.height_m[ia] == pytest.approx(25.0)
    assert buildings.heightSource[ia] == HeightSource.BUILDING_HEIGHT_RASTER
    assert buildings.nativeResolution_m[ia] == 4.0
    assert buildings.height_m[ib] == DEFAULT_BUILDING_HEIGHT_m
    assert buildings.heightSource[ib] == HeightSource.DEFAULT_HEIGHT


def testLoad_tinyBuildingSampledAtCentroid(plane: SyntheticScene, tmp_path: Path, writeRaster):
    pack = _copyPack(plane, tmp_path)
    tiny = _enuBox(plane, -301.0, 251.0, -300.0, 252.0)  # far smaller than a raster pixel
    ingestFiles(
        pack, "overtureBuildings", [_writeBuildings(tmp_path / "o.parquet", [tiny], [None], [None])]
    )
    bounds = Pack.open(pack).bounds_deg
    ingestFiles(
        pack,
        "googleOpenBuildings25d",
        [writeRaster(tmp_path / "g.tif", bounds, values=np.full((20, 20), 7.0, np.float32))],
    )
    buildings = _load(plane, _geometry(plane), pack)
    i = list(buildings.sourceFeatureIds).index("x0")
    assert buildings.height_m[i] == pytest.approx(7.0)


def testLoad_noBuildingSourcesGivesEmptySet(tmp_path: Path):
    scene = makeSyntheticPack(tmp_path / "s.pack")
    pack = tmp_path / "nob.pack"
    shutil.copytree(scene.packDir, pack)
    from map3d.pack.manifest import Manifest

    manifest = Manifest.load(pack)
    del manifest.sources["syntheticBuildings"]
    manifest.save(pack)
    buildings = _load(scene, _geometry(scene), pack)
    assert len(buildings) == 0
    assert (
        rasterizeBuildings(
            buildings, ChunkColumns(_geometry(scene), slice(None), slice(None))
        ).max()
        == 0
    )


def testLoad_dropsBuildingsOutsideGrid(plane: SyntheticScene):
    # A 200 m swath (half-size 100 m) only reaches B1 at [50, 150] x [50, 90].
    buildings = _load(plane, _geometry(plane, swath_m=200.0))
    assert list(buildings.sourceFeatureIds) == ["synB1"]


def testRasterize_interiorAndExterior(plane: SyntheticScene):
    geom = _geometry(plane)
    buildings = _load(plane, geom)
    ids = rasterizeBuildings(buildings, ChunkColumns(geom, slice(None), slice(None)))
    s, half = geom.gridSpacing_m, 400.0
    col = lambda e: int((e + half) / s)  # noqa: E731
    row = lambda n: int((n + half) / s)  # noqa: E731
    assert ids[row(-100.0), col(-160.0)] == 1  # inside B0
    assert ids[row(70.0), col(100.0)] == 2  # inside B1
    assert ids[row(-100.0), col(-205.0)] == 0  # 5 m west of B0
    assert ids[row(0.0), col(0.0)] == 0


def testRasterize_conservativeCoverageOfPartialCells(plane: SyntheticScene):
    # 10 m cells; B2 is a 20 x 20 m box at [300, 320] x [-300, -280] offset by 5 m from the cell grid.
    geom = _geometry(plane, swath_m=810.0, s=10.0)
    buildings = _load(plane, geom)
    ids = rasterizeBuildings(buildings, ChunkColumns(geom, slice(None), slice(None)))
    b2 = list(buildings.sourceFeatureIds).index("synB2") + 1
    assert (ids == b2).sum() == 9  # 3 x 3 cells touched, not 2 x 2


def testRasterize_rotatedGridHitsBuildingCenter(plane: SyntheticScene):
    geom = _geometry(plane, heading_deg=30.0)
    buildings = _load(plane, geom)
    ids = rasterizeBuildings(buildings, ChunkColumns(geom, slice(None), slice(None)))
    e_m, n_m = geom.nodesEnu()
    nearest = np.unravel_index(np.argmin((e_m + 160.0) ** 2 + (n_m + 100.0) ** 2), e_m.shape)
    assert ids[nearest] == 1


def testRasterize_tallestWinsWhereFootprintsOverlap(plane: SyntheticScene, tmp_path: Path):
    pack = _copyPack(plane, tmp_path)
    low = _enuBox(plane, -350.0, 250.0, -250.0, 300.0)
    high = _enuBox(plane, -300.0, 280.0, -200.0, 320.0)  # overlaps < 50%: not a duplicate
    ingestFiles(
        pack,
        "overtureBuildings",
        [_writeBuildings(tmp_path / "o.parquet", [low, high], [10.0, 40.0], [None, None])],
    )
    geom = _geometry(plane)
    buildings = _load(plane, geom, pack)
    ids = rasterizeBuildings(buildings, ChunkColumns(geom, slice(None), slice(None)))
    iHigh = list(buildings.sourceFeatureIds).index("x1") + 1
    assert ids[int((290.0 + 400) / 2), int((-275.0 + 400) / 2)] == iHigh


def testRasterize_chunkedEqualsWhole(plane: SyntheticScene):
    geom = _geometry(plane, heading_deg=30.0, s=4.0)
    buildings = _load(plane, geom)
    whole = rasterizeBuildings(buildings, ChunkColumns(geom, slice(None), slice(None)))
    parts = np.zeros_like(whole)
    for rows, cols in geom.iterChunks(37, 53):
        parts[rows, cols] = rasterizeBuildings(buildings, ChunkColumns(geom, rows, cols))
    np.testing.assert_array_equal(parts, whole)


def testFeatureTable_columns(plane: SyntheticScene):
    table = _load(plane, _geometry(plane)).featureTable()
    np.testing.assert_array_equal(table["featureId"], [1, 2, 3])
    assert list(table["kind"]) == ["building"] * 3
    assert list(table["sourceDataset"]) == ["syntheticBuildings"] * 3
    assert list(table["heightSource"]) == ["EXPLICIT_HEIGHT", "EXPLICIT_HEIGHT", "FLOORS"]
    np.testing.assert_array_equal(table["deckLevel"], [-1, -1, -1])


def testResolveHeights_rasterBelowMinimumIsMissingAndSmallDefault():
    from map3d.layers import buildings as bl

    nan = np.nan
    heights = np.array([nan, nan, nan, nan, nan])
    floors = np.array([nan, 2.0, nan, nan, nan])
    authoritative = np.zeros(5, dtype=bool)
    raster = np.array([0.1, 0.4, 2.5, 1.9, 0.2])
    area = np.array([30.0, 30.0, 30.0, 400.0, 49.9])
    height, code, _ = bl._resolveHeights(heights, floors, authoritative, raster, 4.0, area)
    np.testing.assert_allclose(height, [3.0, 6.0, 2.5, 9.0, 3.0])
    assert list(code) == [
        HeightSource.DEFAULT_HEIGHT,  # 0.1 m raster -> missing; small footprint -> 3 m
        HeightSource.FLOORS,  # 0.4 m raster -> missing; falls through to 2 floors
        HeightSource.BUILDING_HEIGHT_RASTER,  # 2.5 m raster is plausible
        HeightSource.DEFAULT_HEIGHT,  # large footprint -> 9 m
        HeightSource.DEFAULT_HEIGHT,  # just under 50 m2 -> 3 m
    ]
    assert (
        bl.SMALL_DEFAULT_HEIGHT_m == 3.0
        and bl.SMALL_FOOTPRINT_m2 == 50.0
        and bl.MIN_RASTER_HEIGHT_m == 2.0
    )


def testLoad_smallFootprintWithoutDataGetsOneStorey(plane: SyntheticScene, tmp_path: Path):
    pack = _copyPack(plane, tmp_path)
    shed = _enuBox(plane, -301.0, 251.0, -295.0, 257.0)  # 6 x 6 m = 36 m2
    ingestFiles(
        pack, "overtureBuildings", [_writeBuildings(tmp_path / "o.parquet", [shed], [None], [None])]
    )
    buildings = _load(plane, _geometry(plane), pack)
    i = list(buildings.sourceFeatureIds).index("x0")
    assert buildings.height_m[i] == 3.0
    assert buildings.heightSource[i] == HeightSource.DEFAULT_HEIGHT
