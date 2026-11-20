"""Elevated structures (bridges, flyovers, viaducts) as stacked deck levels (SPEC section 8).

Open data has no deck elevations, so heights are **estimated** (provenance
``DECK_ESTIMATED``):

- **Footprints:**
  - Bridge pieces from the transport role are buffered by their width (or a
    per-kind default) in grid-pixel space.
  - Elevated-structure outlines (e.g. LTA flyovers) are added unless more
    than half their area is already covered by a bridge footprint.
- **Height:** deck top = ground interpolated linearly between the bridge's
  two ends (projected along its centerline) + level x per-level height (road
  7 m, rail viaduct 10 m, footbridge 6 m). Levels <= 0 or missing count as 1. Outline-only
  decks use the median ground on their outline. The deck underside is
  :data:`DECK_THICKNESS_m` below the top.
- **Per node:** decks are sorted by top height. Tops within
  :data:`MERGE_TOLERANCE_m` merge into one level, so overlapping carriageways
  and piece joints don't create phantom levels. The lowest ``K`` levels are
  stored, and the highest deck is always reported for the surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import shapely
from rasterio.features import rasterize
from rasterio.transform import Affine

from map3d.gridGeometry import GridGeometry
from map3d.layers.columns import ChunkColumns
from map3d.layers.pixels import toPixelGeometries
from map3d.layers.terrain import GroundModel
from map3d.pack.reader import Pack
from map3d.pack.registry import sourcesForRole

__all__ = [
    "DECK_THICKNESS_m",
    "DEFAULT_DECK_WIDTH_m",
    "MERGE_TOLERANCE_m",
    "RAIL_LEVEL_HEIGHT_m",
    "ROAD_LEVEL_HEIGHT_m",
    "DeckChunk",
    "DeckSet",
    "loadDecks",
    "rasterizeDecks",
]

FloatArray = npt.NDArray[np.float64]

ROAD_LEVEL_HEIGHT_m = 7.0
RAIL_LEVEL_HEIGHT_m = 10.0
FOOTBRIDGE_LEVEL_HEIGHT_m = 6.0
DECK_THICKNESS_m = 1.5
MERGE_TOLERANCE_m = 3.0
DEFAULT_DECK_WIDTH_m = {"road": 8.0, "rail": 8.0, "footway": 3.0}
LEVEL_HEIGHT_m = {
    "road": ROAD_LEVEL_HEIGHT_m,
    "rail": RAIL_LEVEL_HEIGHT_m,
    "footway": FOOTBRIDGE_LEVEL_HEIGHT_m,
}
DUPLICATE_OVERLAP = 0.5
_OUTLINE_STEP_deg = 5e-5


@dataclass(frozen=True, eq=False)
class DeckSet:
    """Deck features placed in the scene. Deck ``i`` is referenced as ``i + 1``.

    Attributes:
        pixelShapes: Deck footprints in grid pixel space.
        centerlines: Pixel-space centerline, or ``None`` for outline-only decks.
        groundStart_m: Orthometric ground at the centerline start (or outline median).
        groundEnd_m: Orthometric ground at the centerline end (or outline median).
        levelHeight_m: Deck top above the interpolated ground.
        levels: Nominal level (>= 1).
        sourceIds: Source of each deck.
        sourceFeatureIds: Feature id within its source.
    """

    pixelShapes: npt.NDArray[np.object_]
    centerlines: npt.NDArray[np.object_]
    groundStart_m: FloatArray
    groundEnd_m: FloatArray
    levelHeight_m: FloatArray
    levels: npt.NDArray[np.int64]
    sourceIds: npt.NDArray[np.str_]
    sourceFeatureIds: npt.NDArray[np.str_]

    def __len__(self) -> int:
        return int(self.levels.size)

    def featureTable(self, firstFeatureId: int) -> dict[str, npt.NDArray[Any]]:
        """Columns of the scene ``features`` table for these decks.

        Args:
            firstFeatureId: Feature id of deck 0 (after the buildings).

        Returns:
            Column name to array.
        """
        n = len(self)
        return {
            "featureId": np.arange(firstFeatureId, firstFeatureId + n, dtype=np.int64),
            "kind": np.array(["deck"] * n, dtype=np.str_),
            "sourceDataset": self.sourceIds.astype(np.str_),
            "sourceFeatureId": self.sourceFeatureIds.astype(np.str_),
            "height_m": self.levelHeight_m.copy(),
            "heightSource": np.array(["DECK_ESTIMATED"] * n, dtype=np.str_),
            "deckLevel": self.levels.copy(),
        }


@dataclass(frozen=True, eq=False)
class DeckChunk:
    """Deck rasterization of one chunk.

    Attributes:
        tops: Orthometric top of levels ``0..K-1`` (lowest first), NaN where absent.
        bottoms: Orthometric underside of those levels.
        maxTop: Highest deck top at each node (any level), NaN where none.
        maxDeck: Deck index + 1 of that highest deck; 0 where none.
        overflowNodes: Nodes with more than ``K`` levels.
    """

    tops: FloatArray
    bottoms: FloatArray
    maxTop: FloatArray
    maxDeck: npt.NDArray[np.int64]
    overflowNodes: int


def loadDecks(pack: Pack, geometry: GridGeometry, ground: GroundModel) -> DeckSet:
    """Collect and place every deck touching the scene.

    Args:
        pack: Opened pack.
        geometry: Scene grid geometry.
        ground: Ground model (deck heights are relative to it).

    Returns:
        The decks (possibly empty).
    """
    bounds = geometry.footprintBounds_deg(margin_m=geometry.gridSpacing_m)
    s = geometry.gridSpacing_m
    shapes: list[Any] = []
    lines: list[Any] = []
    g0: list[float] = []
    g1: list[float] = []
    heights: list[float] = []
    levels: list[int] = []
    sources: list[str] = []
    ids: list[str] = []
    for sourceId in sourcesForRole("transport"):
        if sourceId not in pack.manifest.sources:
            continue
        sel = pack.readVectors(
            sourceId, bounds, columns=["sourceFeatureId", "kind", "isBridge", "level", "width_m"]
        )
        t = sel.table
        bridge = np.array([bool(v) for v in t.column("isBridge").to_pylist()], dtype=bool)
        if not bridge.any():
            continue
        geoms = sel.geometries[bridge]
        kinds = [str(k) for k in np.array(t.column("kind").to_pylist(), dtype=object)[bridge]]
        widths = [
            DEFAULT_DECK_WIDTH_m.get(k, DEFAULT_DECK_WIDTH_m["road"]) if w is None else float(w)
            for k, w in zip(
                kinds, np.array(t.column("width_m").to_pylist(), dtype=object)[bridge], strict=True
            )
        ]
        lvls = [
            max(1, int(v)) if v is not None else 1
            for v in np.array(t.column("level").to_pylist(), dtype=object)[bridge]
        ]
        pixelLines = toPixelGeometries(geoms, 0.0, geometry)
        start = shapely.get_point(geoms, 0)
        end = shapely.get_point(geoms, -1)
        g0 += list(ground.orthometricAt(shapely.get_y(start), shapely.get_x(start)))
        g1 += list(ground.orthometricAt(shapely.get_y(end), shapely.get_x(end)))
        shapes += list(shapely.buffer(pixelLines, 0.5 * np.array(widths) / s, cap_style="flat"))
        lines += list(pixelLines)
        heights += [
            lv * LEVEL_HEIGHT_m.get(k, ROAD_LEVEL_HEIGHT_m)
            for lv, k in zip(lvls, kinds, strict=True)
        ]
        levels += lvls
        sources += [sourceId] * len(kinds)
        ids += [
            str(v) for v in np.array(t.column("sourceFeatureId").to_pylist(), dtype=object)[bridge]
        ]

    bridgeShapes = np.array(shapes, dtype=object)
    for sourceId in sourcesForRole("elevatedStructures"):
        if sourceId not in pack.manifest.sources:
            continue
        sel = pack.readVectors(sourceId, bounds, columns=["sourceFeatureId"])
        pixel = toPixelGeometries(shapely.make_valid(sel.geometries), 0.0, geometry)
        keep = shapely.area(pixel) > 0
        if bridgeShapes.size:
            keep &= ~_coveredBy(pixel, bridgeShapes)
        idx = np.flatnonzero(keep)
        median = _outlineMedianGround(sel.geometries[idx], ground)
        shapes += list(pixel[idx])
        lines += [None] * idx.size
        g0 += list(median)
        g1 += list(median)
        heights += [ROAD_LEVEL_HEIGHT_m] * idx.size
        levels += [1] * idx.size
        sources += [sourceId] * idx.size
        ids += [
            str(v)
            for v in np.array(sel.table.column("sourceFeatureId").to_pylist(), dtype=object)[idx]
        ]

    allShapes = np.array(shapes, dtype=object)
    gs, ge = np.array(g0, dtype=np.float64), np.array(g1, dtype=np.float64)
    gridBox = shapely.box(0.0, 0.0, geometry.nCols, geometry.nRows)
    keep = (
        np.isfinite(gs) & np.isfinite(ge) & shapely.intersects(allShapes, gridBox)
        if allShapes.size
        else np.zeros(0, dtype=bool)
    )
    return DeckSet(
        pixelShapes=allShapes[keep],
        centerlines=np.array(lines, dtype=object)[keep] if lines else np.array([], dtype=object),
        groundStart_m=gs[keep],
        groundEnd_m=ge[keep],
        levelHeight_m=np.array(heights, dtype=np.float64)[keep],
        levels=np.array(levels, dtype=np.int64)[keep],
        sourceIds=np.array(sources, dtype=np.str_)[keep],
        sourceFeatureIds=np.array(ids, dtype=np.str_)[keep],
    )


def rasterizeDecks(
    decks: DeckSet, geometry: GridGeometry, columns: ChunkColumns, maxLevels: int
) -> DeckChunk:
    """Rasterize decks into up to ``maxLevels`` stacked levels for a chunk.

    Args:
        decks: Placed decks.
        geometry: Scene grid geometry.
        columns: Chunk column geometry.
        maxLevels: ``K``, the number of levels to store.

    Returns:
        The chunk's deck levels.
    """
    nr, nc = columns.shape
    r0, c0 = columns.rows.start, columns.cols.start
    tops = np.full((maxLevels, nr, nc), np.nan)
    bottoms = np.full((maxLevels, nr, nc), np.nan)
    maxTop = np.full((nr, nc), np.nan)
    maxDeck = np.zeros((nr, nc), dtype=np.int64)
    if len(decks) == 0:
        return DeckChunk(tops, bottoms, maxTop, maxDeck, 0)
    hits = np.flatnonzero(
        shapely.intersects(decks.pixelShapes, shapely.box(c0, r0, c0 + nc, r0 + nr))
    )
    flats: list[npt.NDArray[np.int64]] = []
    values: list[FloatArray] = []
    owners: list[npt.NDArray[np.int64]] = []
    for i in hits:
        mask = rasterize(
            [(decks.pixelShapes[i], 1)],
            out_shape=(nr, nc),
            transform=Affine.translation(c0, r0),
            fill=0,
            all_touched=True,
            dtype="uint8",
        )
        rr, cc = np.nonzero(mask)
        if rr.size == 0:
            continue
        line = decks.centerlines[i]
        if line is None:
            groundH = np.full(rr.size, decks.groundStart_m[i])
        else:
            frac = shapely.line_locate_point(
                line, shapely.points(c0 + cc + 0.5, r0 + rr + 0.5), normalized=True
            )
            groundH = decks.groundStart_m[i] + frac * (
                decks.groundEnd_m[i] - decks.groundStart_m[i]
            )
        flats.append(rr * nc + cc)
        values.append(groundH + decks.levelHeight_m[i])
        owners.append(np.full(rr.size, i + 1, dtype=np.int64))
    if not flats:
        return DeckChunk(tops, bottoms, maxTop, maxDeck, 0)
    flat = np.concatenate(flats)
    top = np.concatenate(values)
    owner = np.concatenate(owners)
    order = np.lexsort((top, flat))
    flat, top, owner = flat[order], top[order], owner[order]
    isFirst = np.ones(flat.size, dtype=bool)
    isFirst[1:] = flat[1:] != flat[:-1]
    newLevel = isFirst.copy()
    newLevel[1:] |= (top[1:] - top[:-1]) >= MERGE_TOLERANCE_m
    groupStart = np.maximum.accumulate(np.where(isFirst, np.arange(flat.size), 0))
    levelCum = np.cumsum(newLevel)
    level = levelCum - levelCum[groupStart]
    isLast = np.ones(flat.size, dtype=bool)
    isLast[:-1] = flat[1:] != flat[:-1]
    maxTop.ravel()[flat[isLast]] = top[isLast]
    maxDeck.ravel()[flat[isLast]] = owner[isLast]
    stored = level < maxLevels
    for k in range(maxLevels):
        sel = stored & (level == k)
        np.fmax.at(tops[k].ravel(), flat[sel], top[sel])
        np.fmin.at(bottoms[k].ravel(), flat[sel], top[sel] - DECK_THICKNESS_m)
    overflow = int(np.unique(flat[level >= maxLevels]).size)
    return DeckChunk(tops, bottoms, maxTop, maxDeck, overflow)


def _coveredBy(
    candidates: npt.NDArray[np.object_], cover: npt.NDArray[np.object_]
) -> npt.NDArray[np.bool_]:
    tree = shapely.STRtree(cover)
    candIdx, covIdx = tree.query(candidates, predicate="intersects")
    overlap = np.zeros(candidates.size)
    if candIdx.size:
        np.add.at(
            overlap, candIdx, shapely.area(shapely.intersection(candidates[candIdx], cover[covIdx]))
        )
    covered: npt.NDArray[np.bool_] = overlap > DUPLICATE_OVERLAP * shapely.area(candidates)
    return covered


def _outlineMedianGround(geoms: npt.NDArray[np.object_], ground: GroundModel) -> FloatArray:
    result = np.full(geoms.size, np.nan)
    if geoms.size == 0:
        return result
    outline = shapely.segmentize(shapely.boundary(geoms), _OUTLINE_STEP_deg)
    coords, owner = shapely.get_coordinates(outline, return_index=True)
    values = ground.orthometricAt(coords[:, 1], coords[:, 0])
    for i in range(geoms.size):
        mine = values[(owner == i) & np.isfinite(values)]
        if mine.size:
            result[i] = float(np.median(mine))
    return result
