"""Grid node lattice, its georeferencing, and chunk tiling (SPEC section 5.2).

Nodes are cell-centered. Arrays are indexed ``[row, col]`` with row 0 at -y and
column 0 at -x of the grid frame. Because v1 grids have z = Up, every node maps
to a horizontal ENU position ``(e, n)`` that defines its vertical column.

Three horizontal coordinate systems are used, converted by :class:`GridGeometry`:

- **ENU** ``(e, n)``: scene-center frame, metres.
- **Grid** ``(gx, gy)``: the grid frame (origin at the grid centre), metres.
- **Pixel** ``(col, row)``: the grid frame scaled by ``1 / gridSpacing_m`` and
  shifted so cell ``(row i, col j)`` spans ``[j, j+1] x [i, i+1]``; node
  ``(i, j)`` sits at pixel ``(j + 0.5, i + 0.5)``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pyproj
from rasterio.transform import Affine

from map3d.frames import LocalFrame
from map3d.request import SceneRequest

__all__ = ["GridGeometry"]

FloatArray = npt.NDArray[np.float64]

_FOOTPRINT_SAMPLES_PER_EDGE = 33


@dataclass(frozen=True, eq=False)
class GridGeometry:
    """Geometry of a scene grid.

    Attributes:
        frame: Scene-center ENU frame.
        gridAxes_ENU: Rows are the grid x, y, z axes in ENU (read-only).
        gridSpacing_m: Node spacing.
        nRows: Number of rows (along grid y).
        nCols: Number of columns (along grid x).
    """

    frame: LocalFrame
    gridAxes_ENU: FloatArray
    gridSpacing_m: float
    nRows: int
    nCols: int

    @classmethod
    def fromRequest(cls, request: SceneRequest) -> GridGeometry:
        """Build the grid geometry for a validated request.

        Args:
            request: The scene request.

        Returns:
            The corresponding grid geometry.
        """
        lat_deg, lon_deg = request.sceneCenter_deg
        return cls(
            frame=LocalFrame(lat_deg, lon_deg),
            gridAxes_ENU=request.gridAxes_ENU,
            gridSpacing_m=request.gridSpacing_m,
            nRows=request.nRows,
            nCols=request.nCols,
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Array shape ``(nRows, nCols)``."""
        return (self.nRows, self.nCols)

    def gridAxisX_m(self) -> FloatArray:
        """Grid-frame x coordinate of each column's nodes, shape ``(nCols,)``."""
        return _cellCenters(self.nCols, self.gridSpacing_m)

    def gridAxisY_m(self) -> FloatArray:
        """Grid-frame y coordinate of each row's nodes, shape ``(nRows,)``."""
        return _cellCenters(self.nRows, self.gridSpacing_m)

    def nodesEnu(
        self, rows: slice | None = None, cols: slice | None = None
    ) -> tuple[FloatArray, FloatArray]:
        """Horizontal ENU position of nodes (the node columns).

        Args:
            rows: Optional row slice (a chunk); defaults to all rows.
            cols: Optional column slice; defaults to all columns.

        Returns:
            ``(e_m, n_m)``, each of shape ``(len(rows), len(cols))``.
        """
        gx_m = self.gridAxisX_m()[cols if cols is not None else slice(None)]
        gy_m = self.gridAxisY_m()[rows if rows is not None else slice(None)]
        return self.gridToEnu(gx_m[None, :], gy_m[:, None])

    def gridToEnu(self, gx_m: FloatArray, gy_m: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Grid-frame horizontal coordinates to ENU ``(e, n)`` (arrays broadcast).

        Args:
            gx_m: Grid x, in meters.
            gy_m: Grid y, in meters.

        Returns:
            ``(e_m, n_m)``.
        """
        a = self.gridAxes_ENU
        return gx_m * a[0, 0] + gy_m * a[1, 0], gx_m * a[0, 1] + gy_m * a[1, 1]

    def enuToGrid(
        self, e_m: FloatArray, n_m: FloatArray, u_m: FloatArray | None = None
    ) -> tuple[FloatArray, FloatArray]:
        """ENU coordinates to grid-frame horizontal ``(gx, gy)`` (arrays broadcast).

        Args:
            e_m: East, in meters.
            n_m: North, in meters.
            u_m: Optional Up, in meters (omitted = 0).

        Returns:
            ``(gx_m, gy_m)``.
        """
        a = self.gridAxes_ENU
        gx_m = a[0, 0] * e_m + a[0, 1] * n_m
        gy_m = a[1, 0] * e_m + a[1, 1] * n_m
        if u_m is not None:
            gx_m = gx_m + a[0, 2] * u_m
            gy_m = gy_m + a[1, 2] * u_m
        return gx_m, gy_m

    def gridToPixel(self, gx_m: FloatArray, gy_m: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Grid-frame coordinates to pixel ``(col, row)``.

        Args:
            gx_m: Grid x, in meters.
            gy_m: Grid y, in meters.

        Returns:
            ``(col, row)`` in pixel units.
        """
        s = self.gridSpacing_m
        return (gx_m + 0.5 * self.nCols * s) / s, (gy_m + 0.5 * self.nRows * s) / s

    def pixelToGrid(self, col: FloatArray, row: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Pixel ``(col, row)`` to grid-frame coordinates (inverse of :meth:`gridToPixel`).

        Args:
            col: Pixel column coordinate.
            row: Pixel row coordinate.

        Returns:
            ``(gx_m, gy_m)``.
        """
        s = self.gridSpacing_m
        return (col - 0.5 * self.nCols) * s, (row - 0.5 * self.nRows) * s

    def enuToPixel(
        self, e_m: FloatArray, n_m: FloatArray, u_m: FloatArray | None = None
    ) -> tuple[FloatArray, FloatArray]:
        """ENU coordinates to pixel ``(col, row)``; see :meth:`enuToGrid`.

        Args:
            e_m: East, in meters.
            n_m: North, in meters.
            u_m: Optional Up, in meters (omitted = 0).

        Returns:
            ``(col, row)`` in pixel units.
        """
        return self.gridToPixel(*self.enuToGrid(e_m, n_m, u_m))

    def pixelToEnu(self, col: FloatArray, row: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Pixel ``(col, row)`` to ENU ``(e, n)``: the exact inverse of :meth:`enuToPixel`.

        Solves the 2x2 horizontal system instead of transposing it (as
        :meth:`gridToEnu` does), so it also inverts axes that are orthonormal only
        within the request tolerance.

        Args:
            col: Pixel column coordinate.
            row: Pixel row coordinate.

        Returns:
            ``(e_m, n_m)``.
        """
        gx_m, gy_m = self.pixelToGrid(col, row)
        a = self.gridAxes_ENU
        det = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
        return (a[1, 1] * gx_m - a[0, 1] * gy_m) / det, (a[0, 0] * gy_m - a[1, 0] * gx_m) / det

    def affine(self) -> Affine:
        """Affine map from pixel-corner ``(col, row)`` to ortho-CRS ``(x, y)``.

        Pixel centers ``(j + 0.5, i + 0.5)`` land exactly on the node ENU
        positions, so with :meth:`orthoCrs` this georeferences the grid exactly.

        Returns:
            The rasterio/GDAL affine transform.
        """
        s = self.gridSpacing_m
        axes = self.gridAxes_ENU
        halfX_m = 0.5 * self.nCols * s
        halfY_m = 0.5 * self.nRows * s
        return Affine(
            s * axes[0, 0],
            s * axes[1, 0],
            -halfX_m * axes[0, 0] - halfY_m * axes[1, 0],
            s * axes[0, 1],
            s * axes[1, 1],
            -halfX_m * axes[0, 1] - halfY_m * axes[1, 1],
        )

    def orthoCrs(self) -> pyproj.CRS:
        """Ellipsoidal orthographic CRS centered on the scene center."""
        return self.frame.orthoCrs()

    def footprintBounds_deg(self, margin_m: float = 0.0) -> tuple[float, float, float, float]:
        """Lon/lat bounding box of the swath rectangle's footprint on the ellipsoid.

        The rectangle covers all cells (not just node centers), optionally grown
        by ``margin_m`` on every side; its perimeter is sampled densely because
        straight grid edges are slightly curved in lon/lat.

        Args:
            margin_m: Extra distance added on every side, in meters.

        Returns:
            ``(minLon, minLat, maxLon, maxLat)`` in degrees.

        Raises:
            ValueError: If ``margin_m`` is negative.
        """
        if margin_m < 0:
            raise ValueError(f"margin_m must be non-negative, got {margin_m}")
        halfX_m = 0.5 * self.nCols * self.gridSpacing_m + margin_m
        halfY_m = 0.5 * self.nRows * self.gridSpacing_m + margin_m
        t = np.linspace(-1.0, 1.0, _FOOTPRINT_SAMPLES_PER_EDGE)
        ones = np.ones_like(t)
        gx_m = np.concatenate([t * halfX_m, ones * halfX_m, -t * halfX_m, -ones * halfX_m])
        gy_m = np.concatenate([-ones * halfY_m, t * halfY_m, ones * halfY_m, -t * halfY_m])
        e_m, n_m = self.gridToEnu(gx_m, gy_m)
        lat_deg, lon_deg, _ = self.frame.columnPointAtHeight(e_m, n_m, 0.0)
        return (
            float(lon_deg.min()),
            float(lat_deg.min()),
            float(lon_deg.max()),
            float(lat_deg.max()),
        )

    def iterChunks(self, chunkRows: int, chunkCols: int) -> Iterator[tuple[slice, slice]]:
        """Tile the grid into row-major chunks.

        Args:
            chunkRows: Maximum rows per chunk.
            chunkCols: Maximum columns per chunk.

        Yields:
            ``(rows, cols)`` slices covering every node exactly once.

        Raises:
            ValueError: If either chunk size is not positive.
        """
        if chunkRows <= 0 or chunkCols <= 0:
            raise ValueError(f"chunk sizes must be positive, got ({chunkRows}, {chunkCols})")
        for row0 in range(0, self.nRows, chunkRows):
            for col0 in range(0, self.nCols, chunkCols):
                yield (
                    slice(row0, min(row0 + chunkRows, self.nRows)),
                    slice(col0, min(col0 + chunkCols, self.nCols)),
                )


def _cellCenters(count: int, spacing_m: float) -> FloatArray:
    return (np.arange(count, dtype=np.float64) + 0.5) * spacing_m - 0.5 * count * spacing_m
