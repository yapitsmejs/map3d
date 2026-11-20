"""Vegetation canopy layer: height above ground (SPEC section 6.1).

- **Raster canopy** (e.g. Meta/WRI, metres above ground) is cell-max pooled
  over a corner-inclusive sub-lattice, like terrain.
- **Tree points** (``trees`` role) are drawn as crown discs of radius
  :func:`treeRadius_m`, and the taller value wins.
- **Coverage:** 0 means no canopy where a source covers the node; NaN means
  no canopy source at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import shapely
from rasterio.features import rasterize
from rasterio.transform import Affine

from map3d.gridGeometry import GridGeometry
from map3d.layers.columns import ChunkColumns, cellMax, supersampleFactor
from map3d.layers.pixels import toPixelGeometries
from map3d.pack.reader import Pack, RasterWindow

__all__ = ["CanopyModel", "treeRadius_m"]

FloatArray = npt.NDArray[np.float64]

_MIN_CROWN_RADIUS_m = 2.0
_CROWN_RADIUS_PER_HEIGHT = 0.25
_MAX_VALID_m = 150.0  # uint8 canopy rasters may use 255 as a fill value


def treeRadius_m(height_m: float | FloatArray) -> FloatArray:
    """Crown radius assumed for a tree of a given height."""
    return np.maximum(
        _MIN_CROWN_RADIUS_m, _CROWN_RADIUS_PER_HEIGHT * np.asarray(height_m, dtype=np.float64)
    )


@dataclass(frozen=True, eq=False)
class CanopyModel:
    """Canopy sources for a scene.

    Attributes:
        sourceId: Raster canopy source used, if any.
        nativeResolution_m: Its resolution (0 if none).
        window: Raster data over the scene, if any.
        treeDiscs: Crown discs in grid pixel space.
        treeHeights_m: Height of each disc.
        treeSourceId: Tree-point source used, if any.
    """

    sourceId: str | None
    nativeResolution_m: float
    window: RasterWindow | None
    treeDiscs: npt.NDArray[np.object_]
    treeHeights_m: FloatArray
    treeSourceId: str | None
    gridSpacing_m: float

    @classmethod
    def load(cls, pack: Pack, geometry: GridGeometry) -> CanopyModel:
        """Read the canopy raster and tree points covering the scene.

        Args:
            pack: Opened pack.
            geometry: Scene grid geometry.

        Returns:
            The canopy model (possibly with no sources).
        """
        bounds = geometry.footprintBounds_deg(margin_m=geometry.gridSpacing_m)
        raster = pack.primarySource("canopyHeight")
        window = pack.readRaster(raster.sourceId, bounds) if raster is not None else None
        trees = pack.primarySource("trees")
        discs = np.array([], dtype=object)
        heights = np.array([], dtype=np.float64)
        if trees is not None:
            selection = pack.readVectors(trees.sourceId, bounds, columns=["height_m"])
            heights = np.array(
                [
                    np.nan if v is None else float(v)
                    for v in selection.table.column("height_m").to_pylist()
                ]
            )
            keep = np.isfinite(heights) & (heights > 0)
            points = toPixelGeometries(selection.geometries[keep], 0.0, geometry)
            heights = heights[keep]
            discs = np.asarray(
                shapely.buffer(points, treeRadius_m(heights) / geometry.gridSpacing_m), dtype=object
            )
        return cls(
            sourceId=None if raster is None else raster.sourceId,
            nativeResolution_m=0.0 if raster is None else raster.nativeResolution_m,
            window=window,
            treeDiscs=discs,
            treeHeights_m=heights,
            treeSourceId=None if trees is None else trees.sourceId,
            gridSpacing_m=geometry.gridSpacing_m,
        )

    def cellMax(self, columns: ChunkColumns) -> FloatArray:
        """Cell-max canopy height above ground for a chunk.

        Args:
            columns: Chunk column geometry.

        Returns:
            Chunk-shaped heights: 0 = no canopy, NaN = no canopy source.
        """
        shape = columns.shape
        if self.window is not None:
            ss = supersampleFactor(self.gridSpacing_m, self.nativeResolution_m)
            lat, lon = columns.latticeLatLon(ss)
            values = self.window.sample(lat, lon)
            values[(values < 0) | (values > _MAX_VALID_m)] = np.nan
            canopy = cellMax(values, ss)
        elif self.treeSourceId is not None:
            canopy = np.zeros(shape)
        else:
            return np.full(shape, np.nan)
        if self.treeDiscs.size:
            canopy = np.fmax(canopy, self._rasterizeTrees(columns))
        return canopy

    def _rasterizeTrees(self, columns: ChunkColumns) -> FloatArray:
        r0, c0 = columns.rows.start, columns.cols.start
        nr, nc = columns.shape
        box = shapely.box(c0, r0, c0 + nc, r0 + nr)
        hits = np.flatnonzero(shapely.intersects(self.treeDiscs, box))
        if hits.size == 0:
            return np.zeros((nr, nc))
        order = hits[np.argsort(self.treeHeights_m[hits], kind="stable")]
        burned = rasterize(
            ((self.treeDiscs[i], float(self.treeHeights_m[i])) for i in order),
            out_shape=(nr, nc),
            transform=Affine.translation(c0, r0),
            fill=0.0,
            all_touched=True,
            dtype="float64",
        )
        return np.asarray(burned, dtype=np.float64)
