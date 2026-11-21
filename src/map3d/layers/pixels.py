"""Lon/lat geometries -> grid pixel coordinates ``(col, row)``.

Pixel space is defined in :mod:`map3d.gridGeometry`. It is isotropic (1 unit =
one grid spacing), so buffers by metres are exact.
Points are converted at a given ellipsoidal height, so raised features (roofs,
decks) sit exactly in the node columns they occupy.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import Any

import numpy as np
import numpy.typing as npt
import shapely
from rasterio.features import rasterize
from rasterio.transform import Affine

from map3d.gridGeometry import GridGeometry

__all__ = ["rasterizePixels", "toPixelGeometries"]

_RASTERIZE_LOCK = threading.Lock()


def toPixelGeometries(
    geoms: npt.NDArray[np.object_], heightEll_m: npt.ArrayLike, geometry: GridGeometry
) -> npt.NDArray[np.object_]:
    """Convert lon/lat geometries to grid pixel coordinates.

    Args:
        geoms: Shapely geometries in lon/lat.
        heightEll_m: Ellipsoidal height per geometry (scalar or one per geometry);
            NaN is treated as 0.
        geometry: Scene grid geometry.

    Returns:
        New geometries in pixel coordinates (input untouched).
    """
    if geoms.size == 0:
        return geoms
    heights = np.broadcast_to(np.asarray(heightEll_m, dtype=np.float64), geoms.shape)
    coords, owner = shapely.get_coordinates(geoms, return_index=True)
    e_m, n_m, u_m = geometry.frame.geodeticToEnu(
        coords[:, 1], coords[:, 0], np.nan_to_num(heights[owner])
    )
    col, row = geometry.enuToPixel(e_m, n_m, u_m)
    return np.asarray(
        shapely.set_coordinates(geoms.copy(), np.column_stack([col, row])), dtype=object
    )


def rasterizePixels(
    shapes: Iterable[tuple[Any, float]],
    outShape: tuple[int, int],
    row0: int,
    col0: int,
    allTouched: bool,
    dtype: str,
    fill: float = 0,
) -> npt.NDArray[Any]:
    """Burn pixel-space shapes into a chunk whose top-left node is ``(row0, col0)``.

    Later shapes overwrite earlier ones, so pass them in ascending priority.

    Args:
        shapes: ``(geometry, value)`` pairs in grid pixel space.
        outShape: Chunk shape ``(rows, cols)``.
        row0: Chunk's first row.
        col0: Chunk's first column.
        allTouched: Burn every touched cell (conservative), not just centres.
        dtype: Output dtype name.
        fill: Background value.

    Returns:
        The burned chunk.
    """
    # rasterio's in-memory dataset wraps its open in warnings.catch_warnings(),
    # which is not thread-safe: concurrent calls leak warnings and can leave the
    # process-wide warning filters altered. Chunks run in threads, so serialize.
    with _RASTERIZE_LOCK:
        burned = rasterize(
            shapes,
            out_shape=outShape,
            transform=Affine.translation(col0, row0),
            fill=fill,
            all_touched=allTouched,
            dtype=dtype,
        )
    return np.asarray(burned)
