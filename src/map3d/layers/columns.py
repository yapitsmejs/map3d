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
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from map3d.frames import WGS84_A_m
from map3d.gridGeometry import GridGeometry
from map3d.layers import _kernels

__all__ = ["COARSE_STEP_m", "ChunkColumns", "SmoothMap", "cellMax", "supersampleFactor"]

FloatArray = npt.NDArray[np.float64]
SmoothMap = Callable[[FloatArray, FloatArray], tuple[FloatArray, ...]]

MAX_SUPERSAMPLE = 8
COARSE_STEP_m = 50.0  # exact-solve lattice spacing; interpolation error << 1 mm


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
    out = np.empty(((nr - 1) // ss, (nc - 1) // ss), dtype=np.float64)
    _kernels.cellMaxPool(np.ascontiguousarray(latticeValues, dtype=np.float64), ss, out)
    return out


class ChunkColumns:
    """Column geometry for a rectangular chunk of grid nodes.

    Footprints are solved exactly (:meth:`LocalFrame.columnPointAtHeight`) on a
    coarse lattice of at most :data:`COARSE_STEP_m` spacing covering the chunk,
    then interpolated bilinearly to nodes or sub-lattices. The footprint map is
    extremely smooth (curvature ~1/R), so this is exact to well under a
    millimetre. The ENU Up offset is nearly quadratic, so its analytic part
    ``(e^2 + n^2) / 2a`` is removed before interpolating and added back after.
    """

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
        self._coarse: _CoarseField | None = None
        self._nodeLatLon: tuple[FloatArray, FloatArray] | None = None
        self._nodeUp: tuple[FloatArray, FloatArray] | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """Chunk shape ``(rows, cols)``."""
        return (self.rows.stop - self.rows.start, self.cols.stop - self.cols.start)

    def nodeLatLon(self) -> tuple[FloatArray, FloatArray]:
        """Geodetic position of each node column's footprint on the ellipsoid."""
        if self._nodeLatLon is None:
            gx, gy = self._nodeAxes()
            field = self._field()
            self._nodeLatLon = (field.interp(field.lat, gx, gy), field.interp(field.lon, gx, gy))
        return self._nodeLatLon

    def toEnuUp(self, heightEll_m: npt.ArrayLike) -> FloatArray:
        """ENU Up coordinate of the column points with given ellipsoidal heights.

        Args:
            heightEll_m: Height above the WGS84 ellipsoid, chunk-shaped (leading
                axes broadcast, e.g. ``(K, rows, cols)``).

        Returns:
            Up coordinate in the scene-center ENU frame.
        """
        if self._nodeUp is None:
            gx, gy = self._nodeAxes()
            field = self._field()
            e_m, n_m = _gridToEnu(self.geometry, gx, gy)
            u0 = field.interp(field.upResidual, gx, gy) - (e_m**2 + n_m**2) / (2.0 * WGS84_A_m)
            self._nodeUp = (u0, field.interp(field.upFactor, gx, gy))
        u0, factor = self._nodeUp
        result: FloatArray = u0 + factor * np.asarray(heightEll_m, dtype=np.float64)
        return result

    def latticeLatLon(self, ss: int) -> tuple[FloatArray, FloatArray]:
        """Footprints of a sub-lattice including all cell corners of the chunk.

        Args:
            ss: Subdivisions per cell side.

        Returns:
            ``(lat_deg, lon_deg)`` of shape ``(rows*ss + 1, cols*ss + 1)``.
        """
        gx, gy = self._latticeAxes(ss)
        field = self._field()
        return field.interp(field.lat, gx, gy), field.interp(field.lon, gx, gy)

    def mapLattice(self, ss: int, fn: SmoothMap) -> tuple[FloatArray, ...]:
        """Evaluate a smooth function of footprint position on a sub-lattice.

        ``fn(lat, lon)`` is evaluated exactly on the coarse lattice only, then
        interpolated. Use it for e.g. lon/lat -> raster pixel coordinates.

        Args:
            ss: Subdivisions per cell side.
            fn: Smooth function returning one or more arrays.

        Returns:
            Interpolated outputs, shape ``(rows*ss + 1, cols*ss + 1)`` each.
        """
        gx, gy = self._latticeAxes(ss)
        return self._mapAt(fn, gx, gy)

    def mapNodes(self, fn: SmoothMap) -> tuple[FloatArray, ...]:
        """Like :meth:`mapLattice`, at node positions (chunk-shaped)."""
        gx, gy = self._nodeAxes()
        return self._mapAt(fn, gx, gy)

    def _mapAt(self, fn: SmoothMap, gx: FloatArray, gy: FloatArray) -> tuple[FloatArray, ...]:
        field = self._field()
        outputs = fn(field.lat, field.lon)
        return tuple(field.interp(np.asarray(v, dtype=np.float64), gx, gy) for v in outputs)

    def _nodeAxes(self) -> tuple[FloatArray, FloatArray]:
        g = self.geometry
        return g.gridAxisX_m()[self.cols], g.gridAxisY_m()[self.rows]

    def _latticeAxes(self, ss: int) -> tuple[FloatArray, FloatArray]:
        g = self.geometry
        step_m = g.gridSpacing_m / ss
        nr, nc = self.shape
        halfX_m = 0.5 * g.nCols * g.gridSpacing_m
        halfY_m = 0.5 * g.nRows * g.gridSpacing_m
        gx: FloatArray = (
            self.cols.start * ss + np.arange(nc * ss + 1, dtype=np.float64)
        ) * step_m - halfX_m
        gy: FloatArray = (
            self.rows.start * ss + np.arange(nr * ss + 1, dtype=np.float64)
        ) * step_m - halfY_m
        return gx, gy

    def _field(self) -> _CoarseField:
        if self._coarse is None:
            gx, gy = self._latticeAxes(1)  # cell edges bound every node and sub-lattice
            self._coarse = _CoarseField.solve(self.geometry, gx[0], gx[-1], gy[0], gy[-1])
        return self._coarse


class _CoarseField:
    """Exact footprints on a coarse regular lattice in grid coordinates."""

    def __init__(
        self,
        gx: FloatArray,
        gy: FloatArray,
        lat: FloatArray,
        lon: FloatArray,
        upResidual: FloatArray,
        upFactor: FloatArray,
    ) -> None:
        self.gx, self.gy = gx, gy
        self.lat, self.lon, self.upResidual, self.upFactor = lat, lon, upResidual, upFactor

    @classmethod
    def solve(cls, g: GridGeometry, x0: float, x1: float, y0: float, y1: float) -> _CoarseField:
        gx = _subset(_globalAxis(g.nCols, g.gridSpacing_m), x0, x1)
        gy = _subset(_globalAxis(g.nRows, g.gridSpacing_m), y0, y1)
        e_m, n_m = _gridToEnu(g, gx, gy)
        lat, lon, u0 = g.frame.columnPointAtHeight(e_m, n_m, 0.0)
        # dU/dh along a column = 1 / (Up0 . n_local): smooth, so interpolated too.
        lat_rad, lon_rad = np.radians(lat), np.radians(lon)
        up0 = g.frame.rotation[2]
        dot = (
            up0[0] * np.cos(lat_rad) * np.cos(lon_rad)
            + up0[1] * np.cos(lat_rad) * np.sin(lon_rad)
            + up0[2] * np.sin(lat_rad)
        )
        return cls(gx, gy, lat, lon, u0 + (e_m**2 + n_m**2) / (2.0 * WGS84_A_m), 1.0 / dot)

    def interp(self, values: FloatArray, gx: FloatArray, gy: FloatArray) -> FloatArray:
        """Separable bilinear interpolation of a coarse field to ``gy x gx`` targets."""
        ix, tx = _bracket(self.gx, gx)
        iy, ty = _bracket(self.gy, gy)
        out = np.empty((iy.size, ix.size), dtype=np.float64)
        _kernels.interpSeparable(
            np.ascontiguousarray(values, dtype=np.float64), ix, tx, iy, ty, out
        )
        return out


def _bracket(axis: FloatArray, targets: FloatArray) -> tuple[npt.NDArray[np.int64], FloatArray]:
    idx = np.clip(np.searchsorted(axis, targets, side="right") - 1, 0, axis.size - 2).astype(
        np.int64
    )
    frac: FloatArray = (targets - axis[idx]) / (axis[idx + 1] - axis[idx])
    return idx, frac


def _gridToEnu(
    g: GridGeometry, gx_m: FloatArray, gy_m: FloatArray
) -> tuple[FloatArray, FloatArray]:
    axes = g.gridAxes_ENU
    e_m = gy_m[:, None] * axes[1, 0] + gx_m[None, :] * axes[0, 0]
    n_m = gy_m[:, None] * axes[1, 1] + gx_m[None, :] * axes[0, 1]
    return e_m, n_m


def _globalAxis(count: int, spacing_m: float) -> FloatArray:
    """Coarse lattice along one grid axis, with step <= ``COARSE_STEP_m``.

    Points sit every ``m`` cell edges (fine grids) or every ``1/q`` of a cell
    (grids coarser than the step), plus the far edge. The lattice is fixed for
    the whole grid, so every chunk interpolates from the same exact points and
    chunked builds are bitwise identical to unchunked ones.
    """
    if spacing_m <= COARSE_STEP_m:
        m = int(COARSE_STEP_m // spacing_m)
        edges = np.arange(0, count + 1, m, dtype=np.float64)
        if edges[-1] != count:
            edges = np.append(edges, float(count))
    else:
        q = math.ceil(spacing_m / COARSE_STEP_m)
        edges = np.arange(0, count * q + 1, dtype=np.float64) / q
    result: FloatArray = edges * spacing_m - 0.5 * count * spacing_m
    return result


def _subset(axis: FloatArray, lo: float, hi: float) -> FloatArray:
    i0 = max(0, int(np.searchsorted(axis, lo, side="right")) - 1)
    i1 = min(axis.size - 1, int(np.searchsorted(axis, hi, side="left")))
    return axis[i0 : max(i1, i0 + 1) + 1]
