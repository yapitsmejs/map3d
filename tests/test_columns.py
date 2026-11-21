"""Tests for ``map3d.layers.columns`` (per-chunk node column geometry)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from map3d.gridGeometry import GridGeometry
from map3d.layers.columns import ChunkColumns, cellMax
from map3d.request import SceneRequest


def _geom(heading_deg: float = 0.0, swath_m: float = 7000.0, s: float = 500.0) -> GridGeometry:
    h = math.radians(heading_deg)
    axes = [[math.cos(h), -math.sin(h), 0.0], [math.sin(h), math.cos(h), 0.0], [0.0, 0.0, 1.0]]
    req = SceneRequest((1.2834, 103.8607), s, axes, swath_m, swath_m)
    return GridGeometry.fromRequest(req)


@pytest.mark.parametrize("height_m", [-50.0, 0.0, 120.0, 600.0])
def testToEnuUp_linearInHeightMatchesExactSolver(height_m: float):
    geom = _geom(30.0)
    cols = ChunkColumns(geom, slice(None), slice(None))
    e_m, n_m = geom.nodesEnu()
    _, _, uRef = geom.frame.columnPointAtHeight(e_m, n_m, height_m)
    np.testing.assert_allclose(cols.toEnuUp(np.full(e_m.shape, height_m)), uRef, atol=1e-6)


def testLatticeLatLon_cornersOfCells():
    geom = _geom(0.0, swath_m=2000.0, s=500.0)  # 4 x 4 cells
    cols = ChunkColumns(geom, slice(1, 3), slice(0, 2))
    lat, lon = cols.latticeLatLon(1)
    assert lat.shape == (3, 3)
    # Corner (row 1, col 0) of the grid: gx = -1000, gy = -500.
    latRef, lonRef, _ = geom.frame.columnPointAtHeight(-1000.0, -500.0, 0.0)
    assert lat[0, 0] == pytest.approx(float(latRef), abs=1e-12)
    assert lon[0, 0] == pytest.approx(float(lonRef), abs=1e-12)


def testLatticeLatLon_supersampled():
    geom = _geom(0.0, swath_m=2000.0, s=500.0)
    lat, _ = ChunkColumns(geom, slice(0, 2), slice(0, 3)).latticeLatLon(4)
    assert lat.shape == (2 * 4 + 1, 3 * 4 + 1)


def testCellMax_poolsEachCellsSubLattice():
    values = np.arange(5 * 7, dtype=np.float64).reshape(5, 7)  # ss=2 -> 2 x 3 cells
    out = cellMax(values, 2)
    expected = np.array(
        [
            [values[0:3, 0:3].max(), values[0:3, 2:5].max(), values[0:3, 4:7].max()],
            [values[2:5, 0:3].max(), values[2:5, 2:5].max(), values[2:5, 4:7].max()],
        ]
    )
    np.testing.assert_array_equal(out, expected)


def testCellMax_ignoresNanUnlessAllNan():
    values = np.array([[np.nan, 1.0, np.nan], [np.nan, np.nan, np.nan]])
    np.testing.assert_array_equal(cellMax(values, 1), [[1.0, 1.0]])
    allNan = np.full((2, 2), np.nan)
    assert np.isnan(cellMax(allNan, 1)).all()


def testCellMax_rejectsMismatchedShape():
    with pytest.raises(ValueError, match="lattice"):
        cellMax(np.zeros((4, 4)), 2)


def testSupersampleFactor():
    from map3d.layers.columns import supersampleFactor

    assert supersampleFactor(0.5, 30.0) == 1
    assert supersampleFactor(10.0, 1.0) == 8  # capped
    assert supersampleFactor(10.0, 4.0) == 3
    assert supersampleFactor(5.0, 0.0) == 1  # vector / unknown


def testInterpolatedFootprint_matchesExactSolverOverDesignSwath():
    # 5 km swath, 0.5 m-ish sampling over a corner chunk far from the center.
    geom = _geom(17.0, swath_m=5000.0, s=0.5)
    rows, cols = slice(9000, 9400), slice(9300, 9900)
    chunk = ChunkColumns(geom, rows, cols)
    lat, lon = chunk.nodeLatLon()
    assert lat.shape == (400, 600)
    e_m, n_m = geom.nodesEnu(rows, cols)
    latRef, lonRef, uRef = geom.frame.columnPointAtHeight(e_m, n_m, 0.0)
    np.testing.assert_allclose(lat, latRef, atol=1e-10, rtol=0)  # ~11 um
    np.testing.assert_allclose(lon, lonRef, atol=1e-10, rtol=0)
    for h in (0.0, 250.0):
        _, _, uExact = geom.frame.columnPointAtHeight(e_m, n_m, h)
        np.testing.assert_allclose(chunk.toEnuUp(np.full(e_m.shape, h)), uExact, atol=1e-6, rtol=0)


def testMapLattice_interpolatesSmoothFunctionOfPosition():
    geom = _geom(0.0, swath_m=4000.0, s=2.0)
    chunk = ChunkColumns(geom, slice(100, 400), slice(200, 700))
    lat, lon = chunk.latticeLatLon(3)

    def fn(la, lo):
        return la * 1000.0, np.sin(np.radians(lo)) * 1e5

    a, b = chunk.mapLattice(3, fn)
    np.testing.assert_allclose(a, lat * 1000.0, atol=1e-7)
    np.testing.assert_allclose(b, np.sin(np.radians(lon)) * 1e5, atol=1e-6)
    na, nb = chunk.mapNodes(fn)
    nlat, nlon = chunk.nodeLatLon()
    np.testing.assert_allclose(na, nlat * 1000.0, atol=1e-7)
    assert nb.shape == (300, 500)


def testCoarseLattice_handlesSingleNodeChunks():
    geom = _geom(0.0, swath_m=2000.0, s=500.0)
    lat, lon = ChunkColumns(geom, slice(1, 2), slice(2, 3)).nodeLatLon()
    e_m, n_m = geom.nodesEnu(slice(1, 2), slice(2, 3))
    latRef, lonRef, _ = geom.frame.columnPointAtHeight(e_m, n_m, 0.0)
    np.testing.assert_allclose(lat, latRef, atol=1e-12)
