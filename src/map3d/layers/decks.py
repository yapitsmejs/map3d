"""Elevated structures (bridges, flyovers, viaducts) as stacked deck levels (SPEC section 8).

Open data has no deck elevations, so heights are **estimated** (provenance
``DECK_ESTIMATED``):

- **Footprints:**
  - Bridge pieces from the transport role are buffered by their width (or a
    per-kind default) in grid-pixel space.
  - Elevated-structure outlines (e.g. LTA flyovers) are added unless more
    than half their area is already covered by a bridge footprint.
    At-grade carriageways (non-bridge, level 0 transport) are punched out of
    them first — except where a bridge passes overhead — so ground-level
    roads inside an interchange outline stay at grade.
- **Height:** deck top = ground sampled along the bridge centerline + level x
  per-level height (road 7 m, rail viaduct 10 m, footbridge 6 m). Levels <= 0
  or missing count as 1. Free ends taper to :data:`END_RESIDUAL_m` above the
  ground at :data:`END_TAPER_GRADE`, over at most half the piece; ends joined
  to another bridge piece or an elevated way (non-bridge, level >= 1) stay at
  full height. See :func:`deckTopAboveGround_m`. Outline-only decks use the
  median ground on their outline. The deck underside is
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

from map3d.gridGeometry import GridGeometry
from map3d.layers._levels import LevelEntries, stackLevels
from map3d.layers._vectors import mostlyCovered, objectColumn, outlineGround, textColumn
from map3d.layers.columns import ChunkColumns
from map3d.layers.pixels import chunkBox, rasterizePixels, toPixelGeometries
from map3d.layers.terrain import GroundModel
from map3d.pack.reader import Pack

__all__ = [
    "DECK_THICKNESS_m",
    "DEFAULT_DECK_WIDTH_m",
    "END_RESIDUAL_m",
    "END_TAPER_GRADE",
    "MERGE_TOLERANCE_m",
    "RAIL_LEVEL_HEIGHT_m",
    "ROAD_LEVEL_HEIGHT_m",
    "DeckChunk",
    "DeckSet",
    "corridorWidths_m",
    "deckTopAboveGround_m",
    "loadDecks",
    "rasterizeDecks",
]

FloatArray = npt.NDArray[np.float64]

ROAD_LEVEL_HEIGHT_m = 7.0
RAIL_LEVEL_HEIGHT_m = 10.0
FOOTBRIDGE_LEVEL_HEIGHT_m = 6.0
DECK_THICKNESS_m = 1.5
MERGE_TOLERANCE_m = 3.0
END_TAPER_GRADE = 0.08  # free deck ends descend to grade at this slope
END_RESIDUAL_m = 0.5  # taper target: the slab stays this far above the ground
DEFAULT_DECK_WIDTH_m = {"road": 8.0, "rail": 8.0, "footway": 3.0}
LEVEL_HEIGHT_m = {
    "road": ROAD_LEVEL_HEIGHT_m,
    "rail": RAIL_LEVEL_HEIGHT_m,
    "footway": FOOTBRIDGE_LEVEL_HEIGHT_m,
}
_JOIN_TOLERANCE_px = 1e-6  # bridge-piece ends closer than this share a vertex


@dataclass(frozen=True, eq=False)
class DeckSet:
    """Deck features placed in the scene. Deck ``i`` is referenced as ``i + 1``.

    Attributes:
        pixelShapes: Deck footprints in grid pixel space.
        centerlines: Pixel-space centerline, or ``None`` for outline-only decks.
        groundStart_m: Orthometric ground at the centerline start (or outline median).
        groundEnd_m: Orthometric ground at the centerline end (or outline median).
        levelHeight_m: Full deck top above the local ground (before end tapers).
        taperStart_m: End-taper length at the start, in meters (0 = no taper).
        taperEnd_m: End-taper length at the end, in meters (0 = no taper).
        levels: Nominal level (>= 1).
        sourceIds: Source of each deck.
        sourceFeatureIds: Feature id within its source.
    """

    pixelShapes: npt.NDArray[np.object_]
    centerlines: npt.NDArray[np.object_]
    groundStart_m: FloatArray
    groundEnd_m: FloatArray
    levelHeight_m: FloatArray
    taperStart_m: FloatArray
    taperEnd_m: FloatArray
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
            "minHeight_m": np.maximum(self.levelHeight_m - DECK_THICKNESS_m, 0.0),
            "roofShape": np.array(["flat"] * n, dtype=np.str_),
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


def deckTopAboveGround_m(
    frac: npt.ArrayLike,
    length_m: float,
    levelHeight_m: float,
    taperStart_m: float,
    taperEnd_m: float,
) -> FloatArray:
    """Deck top height above the local ground along its centerline, with its end tapers.

    The mesh slabs and the grid layers both use this profile, so they agree exactly.

    Args:
        frac: Position(s) along the centerline, normalized to ``[0, 1]``.
        length_m: Centerline length, in meters.
        levelHeight_m: Deck top above ground at full height.
        taperStart_m: Taper length at the start, in meters (0 = no taper).
        taperEnd_m: Taper length at the end, in meters (0 = no taper).

    Returns:
        Height(s) above the local ground, in meters.
    """
    f = np.asarray(frac, dtype=np.float64)
    d = f * length_m
    ramp = np.ones_like(d)
    if taperStart_m > 0.0:
        ramp = np.minimum(ramp, d / taperStart_m)
    if taperEnd_m > 0.0:
        ramp = np.minimum(ramp, (length_m - d) / taperEnd_m)
    np.clip(ramp, 0.0, 1.0, out=ramp)
    return END_RESIDUAL_m + (levelHeight_m - END_RESIDUAL_m) * ramp


def corridorWidths_m(kinds: npt.NDArray[np.object_], widths: npt.NDArray[np.object_]) -> FloatArray:
    """Transport corridor widths: the mapped width, else the default for the kind.

    Decks, the land-cover road burn and the canopy carve all use these widths,
    so they agree on where a road is.

    Args:
        kinds: Transport kind per segment (``"road"``, ``"rail"``, ``"footway"``, ...).
        widths: Mapped width in metres, or ``None``.

    Returns:
        Width per segment, in metres (unknown kinds default to a road).
    """
    return np.array(
        [
            DEFAULT_DECK_WIDTH_m.get(k, DEFAULT_DECK_WIDTH_m["road"]) if w is None else float(w)
            for k, w in zip(kinds, widths, strict=True)
        ],
        dtype=np.float64,
    )


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
    gradeLines: list[Any] = []
    gradeHalfWidth_px: list[float] = []
    elevatedWayJoins: list[Any] = []
    for sourceId in pack.sourcesFor("transport"):
        sel = pack.readVectors(
            sourceId, bounds, columns=["sourceFeatureId", "kind", "isBridge", "level", "width_m"]
        )
        t = sel.table
        bridge = np.array([bool(v) for v in t.column("isBridge").to_pylist()], dtype=bool)
        kindsAll = objectColumn(t, "kind")
        widthsAll = corridorWidths_m(kindsAll, objectColumn(t, "width_m"))
        levelsAll = objectColumn(t, "level")
        grade = ~bridge & np.array([v is not None and int(v) == 0 for v in levelsAll], dtype=bool)
        if grade.any():
            gradeLines += list(toPixelGeometries(sel.geometries[grade], 0.0, geometry))
            gradeHalfWidth_px += list(0.5 * widthsAll[grade] / s)
        elevatedWay = ~bridge & np.array(
            [v is not None and int(v) >= 1 for v in levelsAll], dtype=bool
        )
        if elevatedWay.any():
            wayLines = toPixelGeometries(sel.geometries[elevatedWay], 0.0, geometry)
            elevatedWayJoins += list(shapely.get_point(wayLines, 0)) + list(
                shapely.get_point(wayLines, -1)
            )
        if not bridge.any():
            continue
        geoms = sel.geometries[bridge]
        kinds = kindsAll[bridge]
        lvls = [max(1, int(v)) if v is not None else 1 for v in levelsAll[bridge]]
        pixelLines = toPixelGeometries(geoms, 0.0, geometry)
        start = shapely.get_point(geoms, 0)
        end = shapely.get_point(geoms, -1)
        shapes += list(shapely.buffer(pixelLines, 0.5 * widthsAll[bridge] / s, cap_style="flat"))
        lines += list(pixelLines)
        g0 += ground.orthometricAt(shapely.get_y(start), shapely.get_x(start)).tolist()
        g1 += ground.orthometricAt(shapely.get_y(end), shapely.get_x(end)).tolist()
        heights += [
            lv * LEVEL_HEIGHT_m.get(k, ROAD_LEVEL_HEIGHT_m)
            for lv, k in zip(lvls, kinds, strict=True)
        ]
        levels += lvls
        sources += [sourceId] * geoms.size
        ids += list(textColumn(t, "sourceFeatureId")[bridge])

    bridgeShapes = np.array(shapes, dtype=object)
    outlineShapes: list[Any] = []
    outlineSources: list[str] = []
    outlineIds: list[str] = []
    for sourceId in pack.sourcesFor("elevatedStructures"):
        sel = pack.readVectors(sourceId, bounds, columns=["sourceFeatureId"])
        pixel = toPixelGeometries(shapely.make_valid(sel.geometries), 0.0, geometry)
        outlineShapes += list(pixel)
        outlineSources += [sourceId] * pixel.size
        outlineIds += list(textColumn(sel.table, "sourceFeatureId"))

    if outlineShapes:
        candidates = np.array(outlineShapes, dtype=object)
        candidates = _punchAtGradeCorridors(candidates, gradeLines, gradeHalfWidth_px, bridgeShapes)
        keep = shapely.area(candidates) > 0
        if bridgeShapes.size:
            keep &= ~mostlyCovered(candidates, bridgeShapes)
        idx = np.flatnonzero(keep)
        median = _outlineMedianGround(_pixelToLonLat(candidates[idx], geometry), ground)
        shapes += list(candidates[idx])
        lines += [None] * idx.size
        g0 += list(median)
        g1 += list(median)
        heights += [ROAD_LEVEL_HEIGHT_m] * idx.size
        levels += [1] * idx.size
        sources += [outlineSources[i] for i in idx]
        ids += [outlineIds[i] for i in idx]

    allShapes = np.array(shapes, dtype=object)
    gs, ge = np.array(g0, dtype=np.float64), np.array(g1, dtype=np.float64)
    gridBox = shapely.box(0.0, 0.0, geometry.nCols, geometry.nRows)
    keep = np.isfinite(gs) & np.isfinite(ge) & shapely.intersects(allShapes, gridBox)
    centerlines = np.array(lines, dtype=object)[keep]
    levelHeight_m = np.array(heights, dtype=np.float64)[keep]
    taperStart, taperEnd = _endTapers(centerlines, levelHeight_m, s, elevatedWayJoins)
    return DeckSet(
        pixelShapes=allShapes[keep],
        centerlines=centerlines,
        groundStart_m=gs[keep],
        groundEnd_m=ge[keep],
        levelHeight_m=levelHeight_m,
        taperStart_m=taperStart,
        taperEnd_m=taperEnd,
        levels=np.array(levels, dtype=np.int64)[keep],
        sourceIds=np.array(sources, dtype=np.str_)[keep],
        sourceFeatureIds=np.array(ids, dtype=np.str_)[keep],
    )


def _endTapers(
    centerlines: npt.NDArray[np.object_],
    levelHeight_m: FloatArray,
    gridSpacing_m: float,
    elevatedWayJoins: list[Any],
) -> tuple[FloatArray, FloatArray]:
    """Taper length at each deck end, per the rule in the module docstring.

    Args:
        centerlines: Pixel-space centerlines (``None`` for outline-only decks).
        levelHeight_m: Full deck-top height above ground per deck.
        gridSpacing_m: Grid spacing, to convert pixel lengths to metres.
        elevatedWayJoins: Pixel points where elevated ways end.

    Returns:
        ``(taperStart_m, taperEnd_m)`` per deck; 0 for outline-only decks.
    """
    taperStart = np.zeros(centerlines.size)
    taperEnd = np.zeros(centerlines.size)
    isCenter = np.array([line is not None for line in centerlines], dtype=bool)
    if not isCenter.any():
        return taperStart, taperEnd
    center = centerlines[isCenter]
    full = np.maximum(levelHeight_m[isCenter] - END_RESIDUAL_m, 0.0) / END_TAPER_GRADE
    full = np.minimum(full, 0.5 * shapely.length(center) * gridSpacing_m)
    starts = shapely.get_point(center, 0)
    ends = shapely.get_point(center, -1)
    points = np.concatenate([starts, ends] + ([elevatedWayJoins] if elevatedWayJoins else []))
    coords = np.column_stack([shapely.get_x(points), shapely.get_y(points)])
    n = center.size
    # An end joined to another elevated piece or way keeps its full height.
    match = np.abs(coords[:, None, :] - coords[None, :, :]).max(axis=2) < _JOIN_TOLERANCE_px
    np.fill_diagonal(match, False)  # an end never joins itself
    joined = match.any(axis=1)
    taperStart[isCenter] = np.where(joined[:n], 0.0, full)
    taperEnd[isCenter] = np.where(joined[n : 2 * n], 0.0, full)
    return taperStart, taperEnd


def rasterizeDecks(
    decks: DeckSet,
    columns: ChunkColumns,
    ground: GroundModel,
    maxLevels: int,
    extraLevels: LevelEntries | None = None,
) -> DeckChunk:
    """Rasterize decks into up to ``maxLevels`` stacked levels for a chunk.

    Args:
        decks: Placed decks.
        columns: Chunk column geometry.
        ground: Ground model; centerline decks follow the ground it samples
            at each node (plus :func:`deckTopAboveGround_m`).
        maxLevels: ``K``, the number of levels to store.
        extraLevels: Other elevated surfaces stacked with the decks (raised
            building parts); they don't count towards ``maxTop`` / ``maxDeck``.

    Returns:
        The chunk's elevated levels.
    """
    nr, nc = columns.shape
    maxTop = np.full((nr, nc), np.nan)
    maxDeck = np.zeros((nr, nc), dtype=np.int64)
    extra = LevelEntries.empty() if extraLevels is None else extraLevels
    flat, top, owner = _deckEntries(decks, columns, ground)
    if flat.size:
        order = np.lexsort((top, flat))
        flat, top, owner = flat[order], top[order], owner[order]
        isLast = np.ones(flat.size, dtype=bool)
        isLast[:-1] = flat[1:] != flat[:-1]
        maxTop.ravel()[flat[isLast]] = top[isLast]
        maxDeck.ravel()[flat[isLast]] = owner[isLast]
    entries = LevelEntries.concatenate([LevelEntries(flat, top, top - DECK_THICKNESS_m), extra])
    tops, bottoms, overflow = stackLevels(entries, (nr, nc), maxLevels, MERGE_TOLERANCE_m)
    return DeckChunk(tops, bottoms, maxTop, maxDeck, overflow)


def _deckEntries(
    decks: DeckSet, columns: ChunkColumns, ground: GroundModel
) -> tuple[npt.NDArray[np.int64], FloatArray, npt.NDArray[np.int64]]:
    """Deck tops at the chunk nodes they cover: ``(flat node, top, deck index + 1)``."""
    nr, nc = columns.shape
    r0, c0 = columns.rows.start, columns.cols.start
    none = (np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0, dtype=np.int64))
    if len(decks) == 0:
        return none
    hits = np.flatnonzero(shapely.intersects(decks.pixelShapes, chunkBox(columns)))
    nodeLat, nodeLon = columns.nodeLatLon()
    nodeGround = ground.orthometricAt(nodeLat, nodeLon)
    flats: list[npt.NDArray[np.int64]] = []
    values: list[FloatArray] = []
    owners: list[npt.NDArray[np.int64]] = []
    for i in hits:
        mask = rasterizePixels(
            [(decks.pixelShapes[i], 1)],
            (nr, nc),
            r0,
            c0,
            allTouched=True,
            dtype="uint8",
        )
        rr, cc = np.nonzero(mask)
        if rr.size == 0:
            continue
        line = decks.centerlines[i]
        if line is None:
            top = np.full(rr.size, decks.groundStart_m[i] + decks.levelHeight_m[i])
        else:
            frac = shapely.line_locate_point(
                line, shapely.points(c0 + cc + 0.5, r0 + rr + 0.5), normalized=True
            )
            length_m = shapely.length(line) * columns.geometry.gridSpacing_m
            top = nodeGround[rr, cc] + deckTopAboveGround_m(
                frac, length_m, decks.levelHeight_m[i], decks.taperStart_m[i], decks.taperEnd_m[i]
            )
        flats.append(rr * nc + cc)
        values.append(top)
        owners.append(np.full(rr.size, i + 1, dtype=np.int64))
    if not flats:
        return none
    return np.concatenate(flats), np.concatenate(values), np.concatenate(owners)


def _punchAtGradeCorridors(
    outlines: npt.NDArray[np.object_],
    gradeLines: list[Any],
    gradeHalfWidth_px: list[float],
    bridges: npt.NDArray[np.object_],
) -> npt.NDArray[np.object_]:
    """Subtract at-grade carriageways from elevated-structure outlines.

    Interchange outlines cover the whole structure footprint, including
    ground-level carriageways; a flat estimated deck over the whole polygon
    would raise them. Their buffered footprints are punched out instead —
    except where a bridge passes overhead, so the viaduct stays continuous
    across the undercrossing.

    Args:
        outlines: Outline polygons in grid pixel space.
        gradeLines: Non-bridge, level 0 transport centerlines in pixel space.
        gradeHalfWidth_px: Half of each at-grade segment's width, in pixels.
        bridges: Bridge footprints in pixel space (kept over undercrossings).

    Returns:
        The outlines with at-grade corridors removed (input untouched).
    """
    result = outlines.copy()
    if outlines.size == 0 or not gradeLines:
        return result
    corridors = shapely.buffer(
        np.array(gradeLines, dtype=object), np.asarray(gradeHalfWidth_px), cap_style="flat"
    )
    tree = shapely.STRtree(corridors)
    hitOutline, hitCorridor = tree.query(outlines, predicate="intersects")
    if hitOutline.size == 0:
        return result
    bridgeTree = shapely.STRtree(bridges) if bridges.size else None
    for oi in np.unique(hitOutline):
        hole = shapely.union_all(
            shapely.intersection(outlines[oi], corridors[hitCorridor[hitOutline == oi]])
        )
        if bridgeTree is not None:
            above = bridgeTree.query(outlines[oi], predicate="intersects")
            if np.asarray(above).size:
                hole = hole.difference(shapely.union_all(bridges[np.asarray(above)]))
        result[oi] = outlines[oi].difference(hole)
    return result


def _pixelToLonLat(
    geoms: npt.NDArray[np.object_], geometry: GridGeometry
) -> npt.NDArray[np.object_]:
    def toLonLat(xy: FloatArray) -> FloatArray:
        e, n = geometry.pixelToEnu(xy[:, 0], xy[:, 1])
        lat, lon, _ = geometry.frame.enuToGeodetic(e, n, np.zeros_like(e))
        return np.column_stack([lon, lat])

    return np.asarray(shapely.transform(geoms, toLonLat), dtype=object)


def _outlineMedianGround(geoms: npt.NDArray[np.object_], ground: GroundModel) -> FloatArray:
    result = np.full(geoms.size, np.nan)
    values, owner = outlineGround(geoms, ground)
    for i in range(geoms.size):
        mine = values[(owner == i) & np.isfinite(values)]
        if mine.size:
            result[i] = float(np.median(mine))
    return result
