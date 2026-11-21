"""Land-cover / material class layer (SPEC section 6.4).

ESA WorldCover codes, extended with vector-derived classes. This module gives
the *ground* class: land cover (nearest pixel at the node footprint) with
ground-level roads and rails drawn on top, with their width. The scene then
applies the full precedence building > deck > road/rail > land cover.

Ground transport is drawn with ``all_touched=False`` (cell centres only), so
coarse grids don't inflate roads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import shapely

from map3d.gridGeometry import GridGeometry
from map3d.layers._vectors import objectColumn
from map3d.layers.columns import ChunkColumns
from map3d.layers.decks import corridorWidths_m
from map3d.layers.pixels import burnChunk, toPixelGeometries
from map3d.pack.reader import Pack, RasterWindow
from map3d.pack.registry import sourcesForRole

__all__ = [
    "CLASS_BUILDING",
    "CLASS_DECK",
    "CLASS_RAIL",
    "CLASS_ROAD",
    "LandCoverModel",
    "classTable",
    "remapClasses",
]

CLASS_BUILDING = 200
CLASS_ROAD = 210
CLASS_DECK = 220
CLASS_RAIL = 230

_NAMES = {
    0: "unknown",
    10: "tree",
    20: "shrub",
    30: "grass",
    40: "cropland",
    50: "builtUp",
    60: "bare",
    70: "snowIce",
    80: "water",
    90: "wetland",
    95: "mangrove",
    100: "mossLichen",
    CLASS_BUILDING: "building",
    CLASS_ROAD: "road",
    CLASS_DECK: "deck",
    CLASS_RAIL: "rail",
}
# Impact Observatory 10 m LULC -> WorldCover. 10 = clouds -> unknown.
_REMAP = {"impactObservatoryLulc": {1: 80, 2: 10, 4: 90, 5: 40, 7: 50, 8: 60, 9: 70, 10: 0, 11: 30}}


def classTable() -> dict[str, str]:
    """Code (as string, JSON-friendly) -> class name, for ``scene.json``."""
    return {str(code): name for code, name in _NAMES.items()}


def remapClasses(values: npt.NDArray[np.generic], sourceId: str) -> npt.NDArray[np.uint8]:
    """Map a source's class codes to WorldCover codes; unknown codes become 0."""
    table = _REMAP.get(sourceId)
    raw = np.asarray(values)
    if table is None:
        return raw.astype(np.uint8)
    lut = np.zeros(256, dtype=np.uint8)
    for src, dst in table.items():
        lut[src] = dst
    return lut[np.clip(raw, 0, 255).astype(np.intp)]


@dataclass(frozen=True, eq=False)
class LandCoverModel:
    """Land cover raster plus ground-level transport, ready to classify chunks.

    Attributes:
        window: Land-cover raster over the scene, if any.
        transportShapes: Buffered ground-level road/rail footprints (pixel space).
        transportClasses: Class code of each footprint.
    """

    window: RasterWindow | None
    transportShapes: npt.NDArray[np.object_]
    transportClasses: npt.NDArray[np.uint8]

    @classmethod
    def load(cls, pack: Pack, geometry: GridGeometry) -> LandCoverModel:
        """Read land cover and ground-level transport covering the scene.

        Args:
            pack: Opened pack.
            geometry: Scene grid geometry.

        Returns:
            The model.
        """
        bounds = geometry.footprintBounds_deg(margin_m=geometry.gridSpacing_m)
        entry = pack.primarySource("landCover")
        window = pack.readRaster(entry.sourceId, bounds) if entry is not None else None
        shapes: list[object] = []
        classes: list[int] = []
        for sourceId in sourcesForRole("transport"):
            if sourceId not in pack.manifest.sources:
                continue
            selection = pack.readVectors(
                sourceId, bounds, columns=["kind", "isBridge", "level", "width_m"]
            )
            table = selection.table
            # Every road/rail corridor burns its class - bridges included, so a
            # viaduct's slab top paints as the road it carries. Footways (park
            # paths, pavements) leave the underlying land cover alone. Levels
            # are not filtered: OSM tags level -1 on ordinary roads that dip
            # under intersections, which must keep their surface paint.
            kinds = objectColumn(table, "kind")
            burn = kinds != "footway"
            if not burn.any():
                continue
            kinds = kinds[burn]
            widths = corridorWidths_m(kinds, objectColumn(table, "width_m")[burn])
            lines = toPixelGeometries(selection.geometries[burn], 0.0, geometry)
            shapes.extend(
                # round caps: a node just past a dead-end still burns as road
                shapely.buffer(lines, 0.5 * widths / geometry.gridSpacing_m, cap_style="round")
            )
            classes.extend(CLASS_RAIL if k == "rail" else CLASS_ROAD for k in kinds)
        return cls(
            window=window,
            transportShapes=np.array(shapes, dtype=object),
            transportClasses=np.array(classes, dtype=np.uint8),
        )

    def classes(self, columns: ChunkColumns) -> npt.NDArray[np.uint8]:
        """Ground class of each node in a chunk (0 = unknown).

        Args:
            columns: Chunk column geometry.

        Returns:
            Chunk-shaped uint8 class codes.
        """
        if self.window is not None:
            col, row = columns.mapNodes(self.window.pixelCoords)
            sampled = self.window.sampleAtPixels(col, row, method="nearest")
            out = remapClasses(np.nan_to_num(sampled, nan=0.0), self.window.sourceId)
        else:
            out = np.zeros(columns.shape, dtype=np.uint8)
        # rail (the higher code) is drawn last
        burned = burnChunk(
            self.transportShapes, self.transportClasses, columns, allTouched=False, dtype="uint8"
        )
        return np.where(burned > 0, burned, out).astype(np.uint8)
