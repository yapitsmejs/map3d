"""Lon/lat geometries -> grid pixel coordinates ``(col, row)``, and burning them.

Pixel space is defined in :mod:`map3d.gridGeometry`. It is isotropic (1 unit =
one grid spacing), so buffers by metres are exact. Points are converted at a
given ellipsoidal height, so raised features (roofs, decks) sit exactly in the
node columns they occupy.
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
from map3d.layers.columns import ChunkColumns

__all__ = ["burnChunk", "chunkBox", "rasterizePixels", "toPixelGeometries"]

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


def chunkBox(columns: ChunkColumns) -> shapely.Polygon:
    """Pixel-space box covering a chunk's cells."""
    r0, c0 = columns.rows.start, columns.cols.start
    nr, nc = columns.shape
    return shapely.box(c0, r0, c0 + nc, r0 + nr)


def burnChunk(
    shapes: npt.NDArray[np.object_],
    values: npt.NDArray[Any],
    columns: ChunkColumns,
    allTouched: bool,
    dtype: str,
    priority: npt.NDArray[Any] | None = None,
) -> npt.NDArray[Any]:
    """Burn the pixel-space shapes that touch a chunk; the highest priority wins.

    Args:
        shapes: Geometries in grid pixel space.
        values: Value burned for each shape.
        columns: The chunk.
        allTouched: Burn every touched cell (conservative), not just centres.
        dtype: Output dtype name.
        priority: Draw order key per shape (ascending, stable); defaults to ``values``.

    Returns:
        The burned chunk, 0 where no shape lands.
    """
    hits = np.flatnonzero(shapely.intersects(shapes, chunkBox(columns)))
    if hits.size == 0:
        return np.zeros(columns.shape, dtype=dtype)
    key = values if priority is None else priority
    order = hits[np.argsort(key[hits], kind="stable")]
    return rasterizePixels(
        zip(shapes[order], values[order].tolist(), strict=True),
        columns.shape,
        columns.rows.start,
        columns.cols.start,
        allTouched=allTouched,
        dtype=dtype,
    )


def rasterizePixels(
    shapes: Iterable[tuple[Any, float]],
    outShape: tuple[int, int],
    row0: int,
    col0: int,
    allTouched: bool,
    dtype: str,
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

    Returns:
        The burned chunk (0 where nothing is burned).
    """
    # rasterio's in-memory dataset wraps its open in warnings.catch_warnings(),
    # which is not thread-safe: concurrent calls leak warnings and can leave the
    # process-wide warning filters altered. Chunks run in threads, so serialize.
    with _RASTERIZE_LOCK:
        burned = rasterize(
            shapes,
            out_shape=outShape,
            transform=Affine.translation(col0, row0),
            fill=0,
            all_touched=allTouched,
            dtype=dtype,
        )
    return np.asarray(burned)
