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
from map3d.layers.columns import ChunkColumns
from map3d.layers.pixels import rasterizePixels, toPixelGeometries
from map3d.pack.reader import Pack, RasterWindow
from map3d.pack.registry import sourcesForRole

__all__ = [
    "CLASS_BUILDING",
    "CLASS_DECK",
    "CLASS_RAIL",
    "CLASS_ROAD",
    "DEFAULT_WIDTH_m",
    "LandCoverModel",
    "classTable",
    "remapClasses",
]

CLASS_BUILDING = 200
CLASS_ROAD = 210
CLASS_DECK = 220
CLASS_RAIL = 230
DEFAULT_WIDTH_m = {"road": 7.0, "rail": 4.0}

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
        sourceId: Land-cover source used, if any.
        window: Raster data over the scene, if any.
        transportShapes: Buffered ground-level road/rail footprints (pixel space).
        transportClasses: Class code of each footprint.
    """

    sourceId: str | None
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
            selection = pack.readVectors(sourceId, bounds, columns=["kind", "isBridge", "width_m"])
            table = selection.table
            # Footways (park paths, pavements) leave the underlying land cover alone.
            ground = np.array(
                [
                    not bool(bridge) and kind != "footway"
                    for bridge, kind in zip(
                        table.column("isBridge").to_pylist(),
                        table.column("kind").to_pylist(),
                        strict=True,
                    )
                ],
                dtype=bool,
            )
            if not ground.any():
                continue
            kinds = np.array(table.column("kind").to_pylist(), dtype=object)[ground]
            widths = np.array(
                [
                    DEFAULT_WIDTH_m.get(str(k), DEFAULT_WIDTH_m["road"]) if w is None else float(w)
                    for k, w in zip(
                        kinds,
                        np.array(table.column("width_m").to_pylist(), dtype=object)[ground],
                        strict=True,
                    )
                ]
            )
            lines = toPixelGeometries(selection.geometries[ground], 0.0, geometry)
            shapes.extend(
                shapely.buffer(lines, 0.5 * widths / geometry.gridSpacing_m, cap_style="flat")
            )
            classes.extend(CLASS_RAIL if k == "rail" else CLASS_ROAD for k in kinds)
        return cls(
            sourceId=None if entry is None else entry.sourceId,
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
        nr, nc = columns.shape
        if self.window is not None and self.sourceId is not None:
            col, row = columns.mapNodes(self.window.pixelCoords)
            sampled = self.window.sampleAtPixels(col, row, method="nearest")
            out = remapClasses(np.nan_to_num(sampled, nan=0.0), self.sourceId)
        else:
            out = np.zeros((nr, nc), dtype=np.uint8)
        if self.transportShapes.size:
            r0, c0 = columns.rows.start, columns.cols.start
            box = shapely.box(c0, r0, c0 + nc, r0 + nr)
            hits = np.flatnonzero(shapely.intersects(self.transportShapes, box))
            if hits.size:
                order = hits[
                    np.argsort(self.transportClasses[hits], kind="stable")
                ]  # rail drawn last
                burned = rasterizePixels(
                    ((self.transportShapes[i], int(self.transportClasses[i])) for i in order),
                    (nr, nc),
                    r0,
                    c0,
                    allTouched=False,
                    dtype="uint8",
                    fill=0,
                )
                out = np.where(burned > 0, burned, out).astype(np.uint8)
        return out
