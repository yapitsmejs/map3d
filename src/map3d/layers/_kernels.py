"""Numba kernels for the per-node hot loops (M6).

All kernels are ``nogil`` so scene chunks can run in parallel threads, and
write into caller-provided output arrays. Inputs are 1-D (flattened) unless
noted; NaN handling mirrors the NumPy implementations they replace.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from numba import njit

__all__ = ["bilinearSample", "cellMaxPool", "interpSeparable", "nearestSample"]

FloatArray = npt.NDArray[np.float64]


@njit(cache=True, nogil=True)
def bilinearSample(values: FloatArray, col: FloatArray, row: FloatArray, out: FloatArray) -> None:
    """Bilinear sample between pixel centres at fractional pixel-corner coords.

    NaN where the position is NaN, any neighbour is outside, or (by
    propagation) any neighbour is NaN.
    """
    nR, nC = values.shape
    for p in range(col.size):
        fr = row[p] - 0.5
        fc = col[p] - 0.5
        if not (fr == fr and fc == fc):
            out[p] = np.nan
            continue
        i0f = math.floor(fr)
        j0f = math.floor(fc)
        if i0f < 0 or j0f < 0 or i0f + 1 >= nR or j0f + 1 >= nC:
            out[p] = np.nan
            continue
        i0 = int(i0f)
        j0 = int(j0f)
        ti = fr - i0f
        tj = fc - j0f
        out[p] = (1.0 - ti) * ((1.0 - tj) * values[i0, j0] + tj * values[i0, j0 + 1]) + ti * (
            (1.0 - tj) * values[i0 + 1, j0] + tj * values[i0 + 1, j0 + 1]
        )


@njit(cache=True, nogil=True)
def nearestSample(values: FloatArray, col: FloatArray, row: FloatArray, out: FloatArray) -> None:
    """Value of the pixel containing each fractional pixel-corner position (NaN outside)."""
    nR, nC = values.shape
    for p in range(col.size):
        r = row[p]
        c = col[p]
        if not (r == r and c == c):
            out[p] = np.nan
            continue
        i = math.floor(r)
        j = math.floor(c)
        if i < 0 or j < 0 or i >= nR or j >= nC:
            out[p] = np.nan
        else:
            out[p] = values[int(i), int(j)]


@njit(cache=True, nogil=True)
def interpSeparable(
    field: FloatArray,
    ix: npt.NDArray[np.int64],
    tx: FloatArray,
    iy: npt.NDArray[np.int64],
    ty: FloatArray,
    out: FloatArray,
) -> None:
    """Bilinear interpolation of a 2-D field to the outer product of 1-D brackets."""
    for a in range(iy.size):
        y0 = iy[a]
        fy = ty[a]
        for b in range(ix.size):
            x0 = ix[b]
            fx = tx[b]
            top = (1.0 - fx) * field[y0, x0] + fx * field[y0, x0 + 1]
            bottom = (1.0 - fx) * field[y0 + 1, x0] + fx * field[y0 + 1, x0 + 1]
            out[a, b] = (1.0 - fy) * top + fy * bottom


@njit(cache=True, nogil=True)
def cellMaxPool(lattice: FloatArray, ss: int, out: FloatArray) -> None:
    """Per-cell max over each ``(ss+1) x (ss+1)`` corner-inclusive sub-lattice, ignoring NaN."""
    nr, nc = out.shape
    for i in range(nr):
        for j in range(nc):
            best = -np.inf
            found = False
            for a in range(i * ss, i * ss + ss + 1):
                for b in range(j * ss, j * ss + ss + 1):
                    v = lattice[a, b]
                    if v == v:
                        found = True
                        if v > best:
                            best = v
            out[i, j] = best if found else np.nan
