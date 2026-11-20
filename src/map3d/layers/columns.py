"""Per-chunk node-column geometry and cell-max pooling (SPEC sections 5.3-5.4).

A node column is the line through the node's ENU ``(e, n)`` parallel to the
scene-center Up axis. :class:`ChunkColumns` solves each column's ground
footprint once and converts ellipsoidal heights to ENU Up with the per-column
linear relation ``u = u0 + k*h``. ``k = 1 / (Up0 . n_local)``, where ``n_local``
is the ellipsoid normal at the footprint. This is exact to well below a
millimetre over the design swath; second-order terms are about 1e-8 m.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import numpy.typing as npt

from map3d.gridGeometry import GridGeometry

__all__ = ["ChunkColumns", "cellMax", "supersampleFactor"]

FloatArray = npt.NDArray[np.float64]

MAX_SUPERSAMPLE = 8


def supersampleFactor(gridSpacing_m: float, nativeResolution_m: float) -> int:
    """Sub-lattice subdivisions per cell needed to see a source's detail.

    Args:
        gridSpacing_m: Grid node spacing.
        nativeResolution_m: Source resolution (0 = vector/unknown).

    Returns:
        Subdivisions per cell side, in ``[1, MAX_SUPERSAMPLE]``.
    """
    if nativeResolution_m <= 0:
        return 1
    return int(min(MAX_SUPERSAMPLE, max(1, math.ceil(gridSpacing_m / nativeResolution_m))))


def cellMax(latticeValues: FloatArray, ss: int) -> FloatArray:
    """Max-pool lattice samples into cells, ignoring NaN.

    Args:
        latticeValues: Samples on a sub-lattice of shape ``(rows*ss + 1, cols*ss + 1)``
            that includes every cell corner.
        ss: Subdivisions per cell side.

    Returns:
        Per-cell maximum, shape ``(rows, cols)``; NaN where all samples are NaN.

    Raises:
        ValueError: If the lattice shape is not ``k*ss + 1`` on both axes.
    """
    nr, nc = latticeValues.shape
    if (nr - 1) % ss or (nc - 1) % ss or nr < 2 or nc < 2:
        raise ValueError(f"lattice shape {latticeValues.shape} is not (k*{ss}+1, m*{ss}+1)")
    windows = np.lib.stride_tricks.sliding_window_view(latticeValues, (ss + 1, ss + 1))
    pooled = windows[::ss, ::ss]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN cells
        result: FloatArray = np.nanmax(pooled, axis=(-2, -1))
    return result


class ChunkColumns:
    """Column geometry for a rectangular chunk of grid nodes."""

    def __init__(self, geometry: GridGeometry, rows: slice, cols: slice) -> None:
        """Bind a chunk.

        Args:
            geometry: The grid geometry.
            rows: Row slice of the chunk.
            cols: Column slice of the chunk.
        """
        self.geometry = geometry
        self.rows = slice(*rows.indices(geometry.nRows)[:2])
        self.cols = slice(*cols.indices(geometry.nCols)[:2])
        self._footprint: tuple[FloatArray, FloatArray, FloatArray] | None = None
        self._upFactor: FloatArray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """Chunk shape ``(rows, cols)``."""
        return (self.rows.stop - self.rows.start, self.cols.stop - self.cols.start)

    def nodeLatLon(self) -> tuple[FloatArray, FloatArray]:
        """Geodetic position of each node column's footprint on the ellipsoid."""
        lat, lon, _ = self._solveFootprint()
        return lat, lon

    def toEnuUp(self, heightEll_m: npt.ArrayLike) -> FloatArray:
        """ENU Up coordinate of the column points with given ellipsoidal heights.

        Args:
            heightEll_m: Height above the WGS84 ellipsoid, chunk-shaped.

        Returns:
            Up coordinate in the scene-center ENU frame.
        """
        _, _, u0 = self._solveFootprint()
        return u0 + self._factor() * np.asarray(heightEll_m, dtype=np.float64)

    def latticeLatLon(self, ss: int) -> tuple[FloatArray, FloatArray]:
        """Footprints of a sub-lattice including all cell corners of the chunk.

        Args:
            ss: Subdivisions per cell side.

        Returns:
            ``(lat_deg, lon_deg)`` of shape ``(rows*ss + 1, cols*ss + 1)``.
        """
        g = self.geometry
        step_m = g.gridSpacing_m / ss
        nr, nc = self.shape
        gx_m: FloatArray = (
            self.cols.start * ss + np.arange(nc * ss + 1, dtype=np.float64)
        ) * step_m - 0.5 * g.nCols * g.gridSpacing_m
        gy_m: FloatArray = (
            self.rows.start * ss + np.arange(nr * ss + 1, dtype=np.float64)
        ) * step_m - 0.5 * g.nRows * g.gridSpacing_m
        e_m, n_m = _gridToEnu(g, gx_m, gy_m)
        lat, lon, _ = g.frame.columnPointAtHeight(e_m, n_m, 0.0)
        return lat, lon

    def _solveFootprint(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        if self._footprint is None:
            e_m, n_m = self.geometry.nodesEnu(self.rows, self.cols)
            self._footprint = self.geometry.frame.columnPointAtHeight(e_m, n_m, 0.0)
        return self._footprint

    def _factor(self) -> FloatArray:
        if self._upFactor is None:
            lat, lon, _ = self._solveFootprint()
            lat_rad, lon_rad = np.radians(lat), np.radians(lon)
            up0 = self.geometry.frame.rotation[2]
            dot = (
                up0[0] * np.cos(lat_rad) * np.cos(lon_rad)
                + up0[1] * np.cos(lat_rad) * np.sin(lon_rad)
                + up0[2] * np.sin(lat_rad)
            )
            self._upFactor = 1.0 / dot
        return self._upFactor


def _gridToEnu(
    g: GridGeometry, gx_m: FloatArray, gy_m: FloatArray
) -> tuple[FloatArray, FloatArray]:
    axes = g.gridAxes_ENU
    e_m = gy_m[:, None] * axes[1, 0] + gx_m[None, :] * axes[0, 0]
    n_m = gy_m[:, None] * axes[1, 1] + gx_m[None, :] * axes[0, 1]
    return e_m, n_m
