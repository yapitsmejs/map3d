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
    ground = model.cellMaxOrthometric(ChunkColumns(geom, slice(None), slice(None)))
    np.testing.assert_allclose(ground, _cornerMax(plane, geom, plane.terrainHeight_m), atol=2e-3)


def testCellMax_hillCloseToAnalytic(hill: SyntheticScene):
    geom = _geometry(hill, 15.0, 800.0, 10.0)
    model = GroundModel.load(Pack.open(hill.packDir), geom)
    ground = model.cellMaxOrthometric(ChunkColumns(geom, slice(None), slice(None)))
    np.testing.assert_allclose(ground, _cornerMax(hill, geom, hill.terrainHeight_m), atol=0.05)


def testCellMax_supersamplesCoarseGrids(hill: SyntheticScene):
    # 40 m cells over a 5 m source -> 8x8 sub-lattice catches the hill top inside a cell.
    geom = _geometry(hill, 0.0, 400.0, 40.0)
    model = GroundModel.load(Pack.open(hill.packDir), geom)
    ground = model.cellMaxOrthometric(ChunkColumns(geom, slice(None), slice(None)))
    assert np.nanmax(ground) == pytest.approx(40.0, abs=0.1)  # hill peak 10 + 30


def testChunked_equalsWhole(plane: SyntheticScene):
    geom = _geometry(plane, 30.0, 800.0, 20.0)
    model = GroundModel.load(Pack.open(plane.packDir), geom)
    whole = model.cellMaxOrthometric(ChunkColumns(geom, slice(None), slice(None)))
    parts = np.full(geom.shape, np.nan)
    for rows, cols in geom.iterChunks(7, 11):
        parts[rows, cols] = model.cellMaxOrthometric(ChunkColumns(geom, rows, cols))
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


def testSmoothing_zeroLeavesWindowUntouched(plane: SyntheticScene):
    geom = _geometry(plane, 0.0, 400.0, 20.0)
    raw = GroundModel.load(Pack.open(plane.packDir), geom)
    same = GroundModel.load(Pack.open(plane.packDir), geom, smoothing_m=0.0)
    np.testing.assert_array_equal(same.window.values, raw.window.values)
    assert same.nativeResolution_m == raw.nativeResolution_m


@pytest.mark.parametrize("heading_deg", [0.0, 30.0])
def testSmoothing_preservesPlane(plane: SyntheticScene, heading_deg: float):
    geom = _geometry(plane, heading_deg, 400.0, 20.0)
    model = GroundModel.load(Pack.open(plane.packDir), geom, smoothing_m=90.0)
    ground = model.cellMaxOrthometric(ChunkColumns(geom, slice(None), slice(None)))
    np.testing.assert_allclose(ground, _cornerMax(plane, geom, plane.terrainHeight_m), atol=2e-3)


def testSmoothing_hillMatchesConvolvedGaussian(hill: SyntheticScene):
    # Gaussian hill (sigma 150 m, 30 m tall) * Gaussian filter (FWHM 90 m) is a
    # wider Gaussian with the same volume: peak 30 * 150^2 / (150^2 + sf^2).
    geom = _geometry(hill, 0.0, 400.0, 20.0)
    model = GroundModel.load(Pack.open(hill.packDir), geom, smoothing_m=90.0)
    sf = 90.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    lat, lon = np.array([hill.frame.lat0_deg]), np.array([hill.frame.lon0_deg])
    expected = 10.0 + 30.0 * 150.0**2 / (150.0**2 + sf**2)
    assert float(model.orthometricAt(lat, lon)[0]) == pytest.approx(expected, abs=0.1)
    assert model.nativeResolution_m == 90.0  # effective resolution after smoothing


def testSmoothWindow_ignoresAndKeepsNodata():
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    from map3d.layers.terrain import smoothWindow
    from map3d.pack.reader import RasterWindow

    values = np.full((41, 41), 5.0)
    values[20, 20] = np.nan  # a hole: stays a hole, neighbours unaffected by it
    values[:, :3] = np.nan  # a no-data edge: does not drag the edge down
    window = RasterWindow(values, Affine(10.0, 0, 0, 0, -10.0, 410.0), CRS.from_epsg(3414), "x")
    out = smoothWindow(window, 60.0)
    assert np.isnan(out.values[20, 20]) and np.isnan(out.values[:, :3]).all()
    np.testing.assert_allclose(out.values[np.isfinite(out.values)], 5.0)
    assert out.transform == window.transform and out.sourceId == "x"


def testSmoothWindow_geographicPixelsUseMetres():
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    from map3d.layers.terrain import smoothWindow
    from map3d.pack.reader import RasterWindow

    spike = np.zeros((61, 61))
    spike[30, 30] = 1.0
    px_deg = 10.0 / 111_320.0  # ~10 m pixels at the equator
    geo = RasterWindow(spike, Affine(px_deg, 0, 103.8, 0, -px_deg, 1.3), CRS.from_epsg(4326), "g")
    proj = RasterWindow(spike, Affine(10.0, 0, 0, 0, -10.0, 0), CRS.from_epsg(3414), "p")
    np.testing.assert_allclose(
        smoothWindow(geo, 60.0).values, smoothWindow(proj, 60.0).values, atol=1e-3
    )
    # unit volume is conserved, and the FWHM is ~60 m = 6 pixels
    out = smoothWindow(proj, 60.0).values
    assert out.sum() == pytest.approx(1.0)
    assert out[30, 33] == pytest.approx(0.5 * out[30, 30], rel=0.05)


def testSmoothWindow_rejectsNegative():
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    from map3d.layers.terrain import smoothWindow
    from map3d.pack.reader import RasterWindow

    window = RasterWindow(np.zeros((3, 3)), Affine(1.0, 0, 0, 0, -1.0, 0), CRS.from_epsg(3414), "x")
    with pytest.raises(ValueError, match="smoothing"):
        smoothWindow(window, -1.0)
