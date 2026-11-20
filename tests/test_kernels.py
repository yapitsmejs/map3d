"""Tests for ``map3d.layers._kernels``: each numba kernel vs a NumPy reference.

Every kernel is exercised both compiled and via ``.py_func`` (the pure-Python
original), so coverage measures the kernel bodies honestly.
"""

from __future__ import annotations

import numpy as np
import pytest

from map3d.layers import _kernels as k


def _refBilinear(values: np.ndarray, col: np.ndarray, row: np.ndarray) -> np.ndarray:
    nR, nC = values.shape
    out = np.full(col.shape, np.nan)
    fr, fc = row - 0.5, col - 0.5
    with np.errstate(invalid="ignore"):
        i0, j0 = np.floor(fr), np.floor(fc)
        ok = (
            np.isfinite(fr)
            & np.isfinite(fc)
            & (i0 >= 0)
            & (j0 >= 0)
            & (i0 + 1 < nR)
            & (j0 + 1 < nC)
        )
    ii, jj = i0[ok].astype(int), j0[ok].astype(int)
    ti, tj = fr[ok] - ii, fc[ok] - jj
    out[ok] = (
        (1 - ti) * (1 - tj) * values[ii, jj]
        + (1 - ti) * tj * values[ii, jj + 1]
        + ti * (1 - tj) * values[ii + 1, jj]
        + ti * tj * values[ii + 1, jj + 1]
    )
    return out


@pytest.fixture
def raster() -> np.ndarray:
    values = np.random.default_rng(0).normal(size=(12, 15))
    values[3, 4] = np.nan
    return values


@pytest.mark.parametrize("compiled", [True, False])
def testBilinear_matchesReferenceIncludingEdgesAndNan(raster: np.ndarray, compiled: bool):
    rng = np.random.default_rng(1)
    col = rng.uniform(-1.0, 17.0, 400)
    row = rng.uniform(-1.0, 14.0, 400)
    col[:3] = [np.nan, 5.0, 4.9]
    row[:3] = [2.0, np.nan, 3.9]
    fn = k.bilinearSample if compiled else k.bilinearSample.py_func
    out = np.empty(400)
    fn(raster, col, row, out)
    np.testing.assert_allclose(out, _refBilinear(raster, col, row), equal_nan=True)


@pytest.mark.parametrize("compiled", [True, False])
def testNearest_matchesReference(raster: np.ndarray, compiled: bool):
    col = np.array([0.2, 14.9, 15.0, -0.1, np.nan, 7.5])
    row = np.array([0.0, 11.99, 3.0, 3.0, 2.0, 3.5])
    fn = k.nearestSample if compiled else k.nearestSample.py_func
    out = np.empty(6)
    fn(raster, col, row, out)
    expected = [raster[0, 0], raster[11, 14], np.nan, np.nan, np.nan, raster[3, 7]]
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("compiled", [True, False])
def testInterpSeparable_matchesBroadcastFormula(compiled: bool):
    rng = np.random.default_rng(2)
    field = rng.normal(size=(5, 7))
    ix = rng.integers(0, 6, 9)
    tx = rng.uniform(0, 1, 9)
    iy = rng.integers(0, 4, 4)
    ty = rng.uniform(0, 1, 4)
    fn = k.interpSeparable if compiled else k.interpSeparable.py_func
    out = np.empty((4, 9))
    fn(field, ix, tx, iy, ty, out)
    txb, tyb = tx[None, :], ty[:, None]
    top = (1 - txb) * field[np.ix_(iy, ix)] + txb * field[np.ix_(iy, ix + 1)]
    bottom = (1 - txb) * field[np.ix_(iy + 1, ix)] + txb * field[np.ix_(iy + 1, ix + 1)]
    np.testing.assert_allclose(out, (1 - tyb) * top + tyb * bottom)


@pytest.mark.parametrize("compiled", [True, False])
def testCellMaxPool_ignoresNanAndAllNanGivesNan(compiled: bool):
    lattice = np.arange(5 * 7, dtype=np.float64).reshape(5, 7)
    lattice[0:3, 0:3] = np.nan  # whole first cell (ss=2) NaN
    lattice[4, 6] = np.nan
    fn = k.cellMaxPool if compiled else k.cellMaxPool.py_func
    out = np.empty((2, 3))
    fn(lattice, 2, out)
    windows = np.lib.stride_tricks.sliding_window_view(lattice, (3, 3))[::2, ::2]
    with np.errstate(invalid="ignore"), pytest.warns(RuntimeWarning):
        expected = np.nanmax(windows, axis=(-2, -1))
    np.testing.assert_array_equal(out, expected)
    assert np.isnan(out[0, 0])


@pytest.mark.parametrize("compiled", [True, False])
def testPackCrowns_greedyNonOverlappingAndCovering(compiled: bool):
    fn = k.packCrowns if compiled else k.packCrowns.py_func
    rows, cols = np.mgrid[0:12, 0:12]
    rows, cols = rows.ravel().astype(np.int64), cols.ravel().astype(np.int64)
    radius = np.full(rows.size, 2.5)
    accepted = np.zeros(rows.size, dtype=np.bool_)
    fn(rows, cols, radius, radius * 2.0, 12, 12, accepted)  # crowns twice as wide in columns
    assert accepted[0]  # the first (highest-priority) candidate always wins
    ar, ac = rows[accepted].astype(float), cols[accepted].astype(float)
    # No accepted crown centre lies inside an earlier crown's ellipse ...
    for i in range(ar.size):
        for j in range(i):
            assert ((ar[i] - ar[j]) / 2.5) ** 2 + ((ac[i] - ac[j]) / 5.0) ** 2 > 1.0
    # ... and every candidate lies inside some accepted crown (the canopy is covered).
    d = ((rows[:, None] - ar[None]) / 2.5) ** 2 + ((cols[:, None] - ac[None]) / 5.0) ** 2
    assert (d.min(axis=1) <= 1.0).all()


@pytest.mark.parametrize("compiled", [True, False])
def testPackCrowns_priorityOrderDecides(compiled: bool):
    fn = k.packCrowns if compiled else k.packCrowns.py_func
    rows = np.array([5, 5, 5], dtype=np.int64)
    cols = np.array([6, 5, 9], dtype=np.int64)  # the second is within the first's crown
    accepted = np.zeros(3, dtype=np.bool_)
    fn(rows, cols, np.full(3, 2.0), np.full(3, 2.0), 10, 12, accepted)
    assert accepted.tolist() == [True, False, True]


@pytest.mark.parametrize("compiled", [True, False])
def testGrowRegions_nearestSeedThroughCanopy(compiled: bool):
    fn = k.growRegions if compiled else k.growRegions.py_func
    mask = np.zeros((9, 22), dtype=np.bool_)
    mask[1:8, 1:9] = True  # canopy block A
    mask[1:8, 11:19] = True  # canopy block B, separated by a gap
    mask[4, 9] = True  # a one-pixel bridge from A into the gap (not reaching B)
    labels = np.full(mask.shape, -1, dtype=np.int32)
    seedRows = np.array([4, 4, 0], dtype=np.int64)
    seedCols = np.array([2, 12, 21], dtype=np.int64)  # third seed is away from any canopy
    fn(seedRows, seedCols, mask, 1.0, 1.0, 7.5, labels)
    assert labels[4, 2] == 0 and labels[4, 12] == 1 and labels[0, 21] == 2  # seeds own their pixel
    assert labels[4, 9] == 0  # the bridge belongs to A's seed (B is not reachable through it)
    assert (labels[1:8, 1:9] == 0).all() and (labels[1:8, 11:19] == 1).all()
    assert labels[0, 0] == -1 and labels[4, 10] == -1  # no canopy, no label
    counts = np.bincount(labels[labels >= 0], minlength=3)
    assert counts.tolist() == [57, 56, 1]


@pytest.mark.parametrize("compiled", [True, False])
def testGrowRegions_stopsAtReach(compiled: bool):
    fn = k.growRegions if compiled else k.growRegions.py_func
    mask = np.ones((1, 20), dtype=np.bool_)
    labels = np.full(mask.shape, -1, dtype=np.int32)
    fn(np.array([0], np.int64), np.array([0], np.int64), mask, 1.0, 1.0, 5.0, labels)
    assert labels[0].tolist() == [0] * 6 + [-1] * 14


@pytest.mark.parametrize("compiled", [True, False])
def testGrowRegions_splitsBetweenSeedsByDistance(compiled: bool):
    fn = k.growRegions if compiled else k.growRegions.py_func
    mask = np.ones((1, 11), dtype=np.bool_)
    labels = np.full(mask.shape, -1, dtype=np.int32)
    fn(np.array([0, 0], np.int64), np.array([0, 10], np.int64), mask, 1.0, 2.0, 100.0, labels)
    assert labels[0].tolist() == [0] * 5 + [labels[0, 5]] + [1] * 5
