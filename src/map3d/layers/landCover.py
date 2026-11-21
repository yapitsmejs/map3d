"""Land-cover / material class layer (SPEC section 6.4).

ESA WorldCover codes, extended with vector-derived classes. This module gives
the *ground* class: land cover (nearest pixel at the node footprint), then
land-cover polygons (e.g. mapped water) drawn crisply over it, then
ground-level roads and rails drawn on top, with their width. The scene then
applies the full precedence building > deck > road/rail > land cover.

Where a polygon source maps a class completely (``overtureWater`` maps all
water), raster pixels of that class outside its polygons are the raster's
coarse edge, not real water: they become built-up land. Grass polygons
(parks) leave the raster's tree cover in place.

Ground transport is drawn with ``all_touched=False`` (cell centres only), so
coarse grids don't inflate roads.
"""

from __future__ import annotations

import dataclasses
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
from map3d.pack.registry import getSource

__all__ = [
    "keepsRaster",
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
# Polygon sources that map a class completely: raster pixels of that class
# outside their polygons become the replacement class.
_COMPLETE = {"overtureWater": (80,)}
_REPLACEMENT_CLASS = 50  # built-up
# Polygon class -> raster classes it does not overwrite: a park's grass keeps
# the raster's tree cover (the wooded parts of a park).
_KEEP_RASTER = {30: (10,)}
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
        coverShapes: Land-cover polygons (pixel space), in drawing order.
        coverClasses: Class code of each polygon.
        replacedClasses: Raster classes the polygons map completely; raster
            pixels of these classes become built-up.
    """

    window: RasterWindow | None
    transportShapes: npt.NDArray[np.object_]
    transportClasses: npt.NDArray[np.uint8]
    coverShapes: npt.NDArray[np.object_] = dataclasses.field(
        default_factory=lambda: np.zeros(0, dtype=object)
    )
    coverClasses: npt.NDArray[np.uint8] = dataclasses.field(
        default_factory=lambda: np.zeros(0, dtype=np.uint8)
    )
    replacedClasses: tuple[int, ...] = ()

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
        for sourceId in pack.sourcesFor("transport"):
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
        coverShapes, coverClasses, replaced = _coverPolygons(pack, bounds, geometry)
        return cls(
            window=window,
            transportShapes=np.array(shapes, dtype=object),
            transportClasses=np.array(classes, dtype=np.uint8),
            coverShapes=coverShapes,
            coverClasses=coverClasses,
            replacedClasses=replaced,
        )

    def rasterClasses(self, values: npt.NDArray[np.generic]) -> npt.NDArray[np.uint8]:
        """Land-cover raster values as WorldCover codes, with completely mapped classes replaced.

        Args:
            values: Raw raster values (NaN = no data).

        Returns:
            Class codes (0 = unknown).
        """
        assert self.window is not None
        codes = remapClasses(np.nan_to_num(values, nan=0.0), self.window.sourceId)
        if self.replacedClasses:
            codes = np.where(
                np.isin(codes, self.replacedClasses), np.uint8(_REPLACEMENT_CLASS), codes
            ).astype(np.uint8)
        return codes

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
            out = self.rasterClasses(sampled)
        else:
            out = np.zeros(columns.shape, dtype=np.uint8)
        if self.coverShapes.size:
            cover = burnChunk(
                self.coverShapes,
                self.coverClasses,
                columns,
                allTouched=False,
                dtype="uint8",
                priority=np.arange(self.coverShapes.size),
            )
            out = np.where((cover > 0) & ~keepsRaster(cover, out), cover, out).astype(np.uint8)
        # rail (the higher code) is drawn last
        burned = burnChunk(
            self.transportShapes, self.transportClasses, columns, allTouched=False, dtype="uint8"
        )
        return np.where(burned > 0, burned, out).astype(np.uint8)


def _coverPolygons(
    pack: Pack, bounds: tuple[float, float, float, float], geometry: GridGeometry
) -> tuple[npt.NDArray[np.object_], npt.NDArray[np.uint8], tuple[int, ...]]:
    """Land-cover polygons in drawing order: lowest-priority source first, large before small.

    Returns:
        ``(pixel-space polygons, class codes, completely mapped classes)``.
    """
    shapes: list[npt.NDArray[np.object_]] = []
    classes: list[npt.NDArray[np.uint8]] = []
    replaced: set[int] = set()
    for sourceId in reversed(pack.sourcesFor("landCoverPolygons")):
        selection = pack.readVectors(sourceId, bounds, columns=["landClass"])
        codes = np.array(selection.table.column("landClass").to_pylist(), dtype=np.uint8)
        order = np.argsort(-shapely.area(selection.geometries), kind="stable")
        shapes.append(toPixelGeometries(selection.geometries[order], 0.0, geometry))
        classes.append(codes[order])
        replaced.update(_COMPLETE.get(getSource(sourceId).sourceId, ()))
    if not shapes:
        return np.zeros(0, dtype=object), np.zeros(0, dtype=np.uint8), ()
    return np.concatenate(shapes), np.concatenate(classes), tuple(sorted(replaced))


def keepsRaster(
    cover: npt.NDArray[np.uint8], raster: npt.NDArray[np.uint8]
) -> npt.NDArray[np.bool_]:
    """Where a land-cover polygon's class leaves the raster's class in place.

    Args:
        cover: Polygon classes (0 = none).
        raster: Raster classes at the same places.

    Returns:
        True where the raster class wins (e.g. tree cover inside a grassy park).
    """
    keep = np.zeros(np.shape(cover), dtype=bool)
    for polygonClass, rasterClasses in _KEEP_RASTER.items():
        keep |= (cover == polygonClass) & np.isin(raster, rasterClasses)
    return keep
