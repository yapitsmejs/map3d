"""Building fusion, height precedence and conservative rasterization (SPEC section 8).

1. **Fusion:** every building source in the pack is read in priority order
   (authoritative first). A feature is dropped as a duplicate when more than
   half of its area overlaps features already accepted from higher-priority
   sources. Overlaps within one source are kept (building parts).
2. **Height precedence:** authoritative height, then authoritative floors x
   storey height, then explicit height, then the building-height raster (median
   inside the footprint), then floors x storey height, then a default.
3. **Placement:** the base is the lowest ground under the footprint outline and
   the roof is base + height. The footprint is converted to grid pixel
   coordinates at roof height, so the roof sits exactly in its node columns.
4. **Rasterization:** any cell touched by a footprint gets the building
   (``all_touched``); where footprints overlap, the tallest roof wins.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pyproj
import shapely
from rasterio.features import rasterize

from map3d.gridGeometry import GridGeometry
from map3d.layers.groundFix import GroundCorrection
from map3d.layers.pixels import rasterizePixels, toPixelGeometries
from map3d.layers.provenance import HeightSource
from map3d.layers.terrain import GroundModel
from map3d.pack.reader import Pack, RasterWindow
from map3d.pack.registry import getSource, sourcesForRole

__all__ = [
    "DEFAULT_BUILDING_HEIGHT_m",
    "DUPLICATE_OVERLAP",
    "STOREY_HEIGHT_m",
    "BuildingSet",
    "loadBuildings",
    "rasterizeBuildings",
]

FloatArray = npt.NDArray[np.float64]

STOREY_HEIGHT_m = 3.0
DEFAULT_BUILDING_HEIGHT_m = 9.0
DUPLICATE_OVERLAP = 0.5
_OUTLINE_STEP_deg = 5e-5  # ~5.5 m: outline densification for the base-height search


@dataclass(frozen=True, eq=False)
class BuildingSet:
    """Fused buildings placed in the scene. Feature ``i`` has ``featureId = i + 1``.

    Attributes:
        geometries: Footprints in lon/lat.
        sourceIds: Source of each building.
        sourceFeatureIds: Feature id within its source.
        height_m: Resolved height above the base.
        heightSource: :class:`HeightSource` code of the resolved height.
        nativeResolution_m: Resolution of the height source (0 for vector attributes).
        baseOrthometric_m: Lowest ground under the footprint (MSL).
        roofOrthometric_m: Roof height (MSL).
        pixelGeometries: Footprints in grid pixel coordinates ``(col, row)`` at roof height.
        groundCorrection: Ground cap applied under buildings, if any.
    """

    geometries: npt.NDArray[np.object_]
    sourceIds: npt.NDArray[np.str_]
    sourceFeatureIds: npt.NDArray[np.str_]
    height_m: FloatArray
    heightSource: npt.NDArray[np.uint8]
    nativeResolution_m: FloatArray
    baseOrthometric_m: FloatArray
    roofOrthometric_m: FloatArray
    pixelGeometries: npt.NDArray[np.object_]
    groundCorrection: GroundCorrection | None = None

    def __len__(self) -> int:
        return int(self.height_m.size)

    def featureTable(self) -> dict[str, npt.NDArray[Any]]:
        """Columns of the scene ``features`` table for these buildings.

        Returns:
            Column name to array.
        """
        n = len(self)
        return {
            "featureId": np.arange(1, n + 1, dtype=np.int64),
            "kind": np.array(["building"] * n, dtype=np.str_),
            "sourceDataset": self.sourceIds.astype(np.str_),
            "sourceFeatureId": self.sourceFeatureIds.astype(np.str_),
            "height_m": self.height_m.copy(),
            "heightSource": np.array(
                [HeightSource(c).name for c in self.heightSource], dtype=np.str_
            ),
            "deckLevel": np.full(n, -1, dtype=np.int64),
        }


def loadBuildings(
    pack: Pack, geometry: GridGeometry, ground: GroundModel, correctGround: bool = True
) -> BuildingSet:
    """Fuse, resolve and place every building that touches the scene.

    Args:
        pack: Opened data pack.
        geometry: Scene grid geometry.
        ground: Ground model (for base heights and the geoid).
        correctGround: Remove building residue from the ground under footprints
            (:mod:`map3d.layers.groundFix`) before placing bases.

    Returns:
        The placed buildings (possibly empty).
    """
    bounds = geometry.footprintBounds_deg(margin_m=geometry.gridSpacing_m)
    geoms, sourceIds, featureIds, heights, floors, authoritative = _fuse(pack, bounds)
    rasterHeight, rasterRes = _rasterHeights(pack, bounds, geoms)
    height, code, nativeRes = _resolveHeights(
        heights, floors, authoritative, rasterHeight, rasterRes
    )
    correction = GroundCorrection.build(geometry, ground, geoms) if correctGround else None
    if correction is not None:
        ground = dataclasses.replace(ground, correction=correction)
    base = _baseHeights(geoms, ground)
    roof = base + height
    centroids = shapely.centroid(geoms)
    roofEll = roof + ground.undulation_m(shapely.get_y(centroids), shapely.get_x(centroids))
    pixel = toPixelGeometries(geoms, roofEll, geometry)
    gridBox = shapely.box(0.0, 0.0, geometry.nCols, geometry.nRows)
    keep = np.isfinite(base) & shapely.intersects(pixel, gridBox)
    return BuildingSet(
        geometries=geoms[keep],
        sourceIds=sourceIds[keep],
        sourceFeatureIds=featureIds[keep],
        height_m=height[keep],
        heightSource=code[keep],
        nativeResolution_m=nativeRes[keep],
        baseOrthometric_m=base[keep],
        roofOrthometric_m=roof[keep],
        pixelGeometries=pixel[keep],
        groundCorrection=correction,
    )


def rasterizeBuildings(
    buildings: BuildingSet, geometry: GridGeometry, rows: slice, cols: slice
) -> npt.NDArray[np.int64]:
    """Feature ids (``index + 1``, 0 = none) of the tallest building touching each cell.

    Args:
        buildings: Placed buildings.
        geometry: Scene grid geometry.
        rows: Row slice of the chunk.
        cols: Column slice of the chunk.

    Returns:
        Chunk-shaped int64 array.
    """
    r0, r1 = rows.indices(geometry.nRows)[:2]
    c0, c1 = cols.indices(geometry.nCols)[:2]
    out = np.zeros((r1 - r0, c1 - c0), dtype=np.int64)
    if len(buildings) == 0:
        return out
    chunkBox = shapely.box(c0, r0, c1, r1)
    hits = np.flatnonzero(shapely.intersects(buildings.pixelGeometries, chunkBox))
    if hits.size == 0:
        return out
    order = hits[np.argsort(buildings.roofOrthometric_m[hits], kind="stable")]
    shapes = ((buildings.pixelGeometries[i], int(i) + 1) for i in order)
    burned = rasterizePixels(
        shapes,
        out.shape,
        r0,
        c0,
        allTouched=True,
        dtype="int32",
        fill=0,
    )
    result: npt.NDArray[np.int64] = np.asarray(burned, dtype=np.int64)
    return result


def _fuse(
    pack: Pack, bounds: tuple[float, float, float, float]
) -> tuple[
    npt.NDArray[np.object_],
    npt.NDArray[np.str_],
    npt.NDArray[np.str_],
    FloatArray,
    FloatArray,
    npt.NDArray[np.bool_],
]:
    geoms: list[Any] = []
    sourceIds: list[str] = []
    featureIds: list[str] = []
    heights: list[float] = []
    floors: list[float] = []
    authoritative: list[bool] = []
    for sourceId in sourcesForRole("buildings"):
        if sourceId not in pack.manifest.sources:
            continue
        selection = pack.readVectors(
            sourceId, bounds, columns=["sourceFeatureId", "height_m", "numFloors"]
        )
        candidates = shapely.make_valid(selection.geometries)
        keep = shapely.area(candidates) > 0
        if geoms:
            keep &= ~_duplicates(candidates, np.array(geoms, dtype=object))
        idx = np.flatnonzero(keep)
        table = selection.table
        geoms.extend(candidates[idx])
        sourceIds.extend([sourceId] * idx.size)
        featureIds.extend(
            str(v)
            for v in np.asarray(table.column("sourceFeatureId").to_pylist(), dtype=object)[idx]
        )
        heights.extend(_floatColumn(table, "height_m")[idx])
        floors.extend(_floatColumn(table, "numFloors")[idx])
        authoritative.extend([getSource(sourceId).authoritative] * idx.size)
    return (
        np.array(geoms, dtype=object),
        np.array(sourceIds, dtype=np.str_),
        np.array(featureIds, dtype=np.str_),
        np.array(heights, dtype=np.float64),
        np.array(floors, dtype=np.float64),
        np.array(authoritative, dtype=bool),
    )


def _floatColumn(table: Any, name: str) -> FloatArray:
    return np.array(
        [np.nan if v is None else float(v) for v in table.column(name).to_pylist()],
        dtype=np.float64,
    )


def _duplicates(
    candidates: npt.NDArray[np.object_], accepted: npt.NDArray[np.object_]
) -> npt.NDArray[np.bool_]:
    tree = shapely.STRtree(accepted)
    candIdx, accIdx = tree.query(candidates, predicate="intersects")
    overlap = np.zeros(candidates.size)
    if candIdx.size:
        areas = shapely.area(shapely.intersection(candidates[candIdx], accepted[accIdx]))
        np.add.at(overlap, candIdx, areas)
    isDuplicate: npt.NDArray[np.bool_] = overlap > DUPLICATE_OVERLAP * shapely.area(candidates)
    return isDuplicate


def _rasterHeights(
    pack: Pack, bounds: tuple[float, float, float, float], geoms: npt.NDArray[np.object_]
) -> tuple[FloatArray, float]:
    entry = pack.primarySource("buildingHeightRaster")
    result = np.full(geoms.size, np.nan)
    if entry is None or geoms.size == 0:
        return result, 0.0
    window = pack.readRaster(entry.sourceId, bounds)
    return _medianInFootprints(window, geoms), entry.nativeResolution_m


def _medianInFootprints(window: RasterWindow, geoms: npt.NDArray[np.object_]) -> FloatArray:
    crs = pyproj.CRS.from_wkt(window.crs.to_wkt())
    toCrs = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    projected = shapely.transform(
        geoms, lambda xy: np.column_stack(toCrs.transform(xy[:, 0], xy[:, 1]))
    )
    ids = rasterize(
        ((g, i + 1) for i, g in enumerate(projected)),
        out_shape=window.values.shape,
        transform=window.transform,
        fill=0,
        all_touched=False,
        dtype="int32",
    ).ravel()
    values = window.values.ravel()
    valid = (ids > 0) & np.isfinite(values) & (values > 0)
    medians = np.full(geoms.size, np.nan)
    if valid.any():
        order = np.lexsort((values[valid], ids[valid]))
        sortedIds = ids[valid][order]
        sortedValues = values[valid][order]
        uniq, start, count = np.unique(sortedIds, return_index=True, return_counts=True)
        medians[uniq - 1] = 0.5 * (
            sortedValues[start + (count - 1) // 2] + sortedValues[start + count // 2]
        )
    # Footprints smaller than a pixel: sample at a representative point.
    missing = np.flatnonzero(~np.isfinite(medians))
    if missing.size:
        points = shapely.point_on_surface(geoms[missing])
        sampled = window.sample(shapely.get_y(points), shapely.get_x(points))
        medians[missing] = np.where(sampled > 0, sampled, np.nan)
    return medians


def _resolveHeights(
    heights: FloatArray,
    floors: FloatArray,
    authoritative: npt.NDArray[np.bool_],
    rasterHeight: FloatArray,
    rasterRes: float,
) -> tuple[FloatArray, npt.NDArray[np.uint8], FloatArray]:
    hasHeight = np.isfinite(heights) & (heights > 0)
    hasFloors = np.isfinite(floors) & (floors > 0)
    hasRaster = np.isfinite(rasterHeight) & (rasterHeight > 0)
    rules = [
        (authoritative & hasHeight, heights, HeightSource.AUTHORITATIVE_HEIGHT, 0.0),
        (
            authoritative & hasFloors,
            floors * STOREY_HEIGHT_m,
            HeightSource.AUTHORITATIVE_FLOORS,
            0.0,
        ),
        (hasHeight, heights, HeightSource.EXPLICIT_HEIGHT, 0.0),
        (hasRaster, rasterHeight, HeightSource.BUILDING_HEIGHT_RASTER, rasterRes),
        (hasFloors, floors * STOREY_HEIGHT_m, HeightSource.FLOORS, 0.0),
    ]
    height = np.full(heights.size, DEFAULT_BUILDING_HEIGHT_m)
    code = np.full(heights.size, HeightSource.DEFAULT_HEIGHT, dtype=np.uint8)
    nativeRes = np.zeros(heights.size)
    for condition, value, source, res in reversed(rules):
        height = np.where(condition, value, height)
        code = np.where(condition, np.uint8(source), code).astype(np.uint8)
        nativeRes = np.where(condition, res, nativeRes)
    return height, code, nativeRes


def _baseHeights(geoms: npt.NDArray[np.object_], ground: GroundModel) -> FloatArray:
    base = np.full(geoms.size, np.inf)
    if geoms.size == 0:
        return base
    outline = shapely.segmentize(shapely.boundary(geoms), _OUTLINE_STEP_deg)
    coords, owner = shapely.get_coordinates(outline, return_index=True)
    values = ground.orthometricAt(coords[:, 1], coords[:, 0])
    np.fmin.at(base, owner, np.where(np.isfinite(values), values, np.inf))
    return np.where(np.isfinite(base), base, np.nan)
