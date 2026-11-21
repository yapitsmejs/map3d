"""Tests for ``map3d.mesh.martini`` (RTIN terrain simplification)."""

from __future__ import annotations

import numpy as np
import pytest

from map3d.mesh import martini


def _hill(size: int) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    c = (size - 1) / 2
    return 30.0 * np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2 * (size / 6) ** 2)) + 0.05 * x


def _meshHeightsAtGrid(terrain: np.ndarray, verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Linear interpolation of the mesh at every grid point (barycentric)."""
    out = np.full(terrain.shape, np.nan)
    z = terrain[verts[:, 1], verts[:, 0]]
    for a, b, c in tris:
        (x0, y0), (x1, y1), (x2, y2) = verts[a], verts[b], verts[c]
        xs = np.arange(min(x0, x1, x2), max(x0, x1, x2) + 1)
        ys = np.arange(min(y0, y1, y2), max(y0, y1, y2) + 1)
        gx, gy = np.meshgrid(xs, ys)
        det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        l0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / det
        l1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / det
        l2 = 1 - l0 - l1
        inside = (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
        out[gy[inside], gx[inside]] = (l0 * z[a] + l1 * z[b] + l2 * z[c])[inside]
    return out


def _area(verts: np.ndarray, tris: np.ndarray) -> float:
    p = verts[tris].astype(np.float64)
    u, v = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    return float(np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]).sum() / 2)


@pytest.mark.parametrize("compiled", [True, False])
def testFlatTerrain_isTwoTriangles(compiled: bool):
    terrain = np.full((9, 9), 5.0)
    errors = (martini.rtinErrors if compiled else martini.rtinErrors.py_func)(terrain)
    verts, tris = (martini.rtinMesh if compiled else martini.rtinMesh.py_func)(errors, 0.0)
    assert tris.shape == (2, 3)
    assert sorted(map(tuple, verts.tolist())) == [(0, 0), (0, 8), (8, 0), (8, 8)]


@pytest.mark.parametrize(("size", "compiled"), [(17, False), (65, True)])
@pytest.mark.parametrize("maxError", [0.0, 0.5, 3.0])
def testMesh_tilesSquareAndHonoursMaxError(size: int, compiled: bool, maxError: float):
    terrain = _hill(size)
    errors = (martini.rtinErrors if compiled else martini.rtinErrors.py_func)(terrain)
    verts, tris = (martini.rtinMesh if compiled else martini.rtinMesh.py_func)(errors, maxError)
    assert _area(verts, tris) == pytest.approx((size - 1) ** 2)
    approx = _meshHeightsAtGrid(terrain, verts, tris)
    assert np.isfinite(approx).all()
    # Martini bounds the error at bintree split points; interior samples may exceed it
    # slightly (the algorithm's known behaviour), but never at maxError = 0.
    bound = 1e-9 if maxError == 0 else 1.25 * maxError
    assert np.abs(approx - terrain).max() <= bound


def testMesh_coarserToleranceGivesFewerTriangles():
    errors = martini.rtinErrors(_hill(129))
    counts = [martini.rtinMesh(errors, e)[1].shape[0] for e in (0.0, 0.2, 1.0, 5.0)]
    assert counts == sorted(counts, reverse=True) and counts[0] > counts[-1]


def testRtin_rejectsBadGridSize():
    with pytest.raises(ValueError, match="2\\^k \\+ 1"):
        martini.checkGridSize(10)
    assert martini.checkGridSize(257) == 257
