"""Right-triangulated irregular network (RTIN) terrain simplification.

A numba port of the Martini algorithm (Agafonkin, after Evans et al. 2001,
"Right-triangulated irregular networks"):

- :func:`rtinErrors` computes, for every bintree split point of a
  ``(2^k + 1)^2`` heightfield, the worst approximation error in its subtree.
- :func:`rtinMesh` extracts the coarsest mesh whose error at every split point
  is within ``maxError``. Vertices are integer grid coordinates ``(x, y)``;
  triangles index into them.

As in Martini, the tolerance is enforced at bintree split points (hypotenuse
midpoints). Interior samples can deviate slightly more (observed within
1.25x); ``maxError = 0`` reproduces the heightfield exactly.

Traversal uses an explicit stack (no recursion) so both run as ``nogil``
kernels.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit

__all__ = ["checkGridSize", "rtinErrors", "rtinMesh"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int32]


def checkGridSize(size: int) -> int:
    """Validate that a heightfield side is ``2^k + 1``.

    Raises:
        ValueError: Otherwise.
    """
    tile = size - 1
    if tile < 1 or tile & (tile - 1):
        raise ValueError(f"RTIN grid size must be 2^k + 1, got {size}")
    return size


@njit(cache=True, nogil=True)
def rtinErrors(terrain: FloatArray) -> FloatArray:
    """Per-vertex subtree errors of a ``(2^k+1) x (2^k+1)`` heightfield (``terrain[y, x]``)."""
    size = terrain.shape[0]
    tile = size - 1
    numTriangles = tile * tile * 2 - 2
    numParents = numTriangles - tile * tile
    coords = np.zeros((numTriangles, 4), dtype=np.int64)
    for i in range(numTriangles):
        node = i + 2
        ax = ay = bx = by = cx = cy = 0
        if node & 1:
            bx = by = cx = tile
        else:
            ax = ay = cy = tile
        node >>= 1
        while node > 1:
            mx = (ax + bx) >> 1
            my = (ay + by) >> 1
            if node & 1:
                bx, by = ax, ay
                ax, ay = cx, cy
            else:
                ax, ay = bx, by
                bx, by = cx, cy
            cx, cy = mx, my
            node >>= 1
        coords[i, 0] = ax
        coords[i, 1] = ay
        coords[i, 2] = bx
        coords[i, 3] = by
    errors = np.zeros((size, size), dtype=np.float64)
    for i in range(numTriangles - 1, -1, -1):
        ax, ay, bx, by = coords[i, 0], coords[i, 1], coords[i, 2], coords[i, 3]
        mx = (ax + bx) >> 1
        my = (ay + by) >> 1
        cx = mx + my - ay
        cy = my + ax - mx
        interpolated = 0.5 * (terrain[ay, ax] + terrain[by, bx])
        err = abs(interpolated - terrain[my, mx])
        if err > errors[my, mx]:
            errors[my, mx] = err
        if i < numParents:
            left = errors[(ay + cy) >> 1, (ax + cx) >> 1]
            right = errors[(by + cy) >> 1, (bx + cx) >> 1]
            if left > errors[my, mx]:
                errors[my, mx] = left
            if right > errors[my, mx]:
                errors[my, mx] = right
    return errors


@njit(cache=True, nogil=True)
def rtinMesh(errors: FloatArray, maxError: float) -> tuple[IntArray, IntArray]:
    """Extract the coarsest RTIN mesh within ``maxError``.

    Returns:
        ``(vertices, triangles)``: vertices ``(N, 2)`` as grid ``(x, y)``,
        triangles ``(M, 3)`` indexing them.
    """
    size = errors.shape[0]
    tile = size - 1
    index = np.full((size, size), -1, dtype=np.int64)
    capacity = 2 * size * size
    stack = np.zeros((capacity, 6), dtype=np.int64)
    tris = np.zeros((capacity, 3), dtype=np.int64)
    verts = np.zeros((size * size, 2), dtype=np.int64)
    nVerts = 0
    nTris = 0
    top = 0
    stack[0] = (0, 0, tile, tile, tile, 0)
    stack[1] = (tile, tile, 0, 0, 0, tile)
    top = 2
    while top > 0:
        top -= 1
        ax, ay, bx, by, cx, cy = stack[top]
        mx = (ax + bx) >> 1
        my = (ay + by) >> 1
        if abs(ax - cx) + abs(ay - cy) > 1 and errors[my, mx] > maxError:
            stack[top] = (cx, cy, ax, ay, mx, my)
            stack[top + 1] = (bx, by, cx, cy, mx, my)
            top += 2
            continue
        for k in range(3):
            if k == 0:
                px, py = ax, ay
            elif k == 1:
                px, py = bx, by
            else:
                px, py = cx, cy
            if index[py, px] < 0:
                index[py, px] = nVerts
                verts[nVerts, 0] = px
                verts[nVerts, 1] = py
                nVerts += 1
            tris[nTris, k] = index[py, px]
        nTris += 1
    return verts[:nVerts].astype(np.int32), tris[:nTris].astype(np.int32)
