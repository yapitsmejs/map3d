"""Lon/lat geometries -> grid pixel coordinates ``(col, row)``.

Pixel space is the grid frame scaled by ``1 / gridSpacing_m`` and shifted so
that cell ``(row i, col j)`` spans ``[j, j+1] x [i, i+1]``, with row 0 at -y.
It is isotropic (1 unit = one grid spacing), so buffers by metres are exact.
Points are converted at a given ellipsoidal height, so raised features (roofs,
decks) sit exactly in the node columns they occupy.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import shapely

from map3d.gridGeometry import GridGeometry

__all__ = ["toPixelGeometries"]


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
    axes = geometry.gridAxes_ENU
    gx_m = axes[0, 0] * e_m + axes[0, 1] * n_m + axes[0, 2] * u_m
    gy_m = axes[1, 0] * e_m + axes[1, 1] * n_m + axes[1, 2] * u_m
    s = geometry.gridSpacing_m
    col = (gx_m + 0.5 * geometry.nCols * s) / s
    row = (gy_m + 0.5 * geometry.nRows * s) / s
    return np.asarray(
        shapely.set_coordinates(geoms.copy(), np.column_stack([col, row])), dtype=object
    )
