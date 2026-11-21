"""Building fusion, height precedence and conservative rasterization (SPEC section 8).

1. **Fusion:** every building source in the pack is read in priority order
   (authoritative first). A feature is dropped as a duplicate when more than
   half of its area overlaps features already accepted from higher-priority
   sources. Overlaps within one source are kept (building parts). A dropped
   duplicate that is the same building as the accepted feature it overlaps
   most (each covers at least :data:`INHERIT_OVERLAP` of the other) hands it
   the appearance attributes it lacks: roof shape (with its height, direction
   and orientation), colours and materials. Heights and min heights stay with
   the accepted, higher-priority source.
2. **Height precedence:** authoritative height, then authoritative floors x
   storey height, then explicit height, then the building-height raster (median
   inside the footprint), then floors x storey height, then a default.
3. **Placement:** heights are measured from the building's **anchor**, the
   median ground under its footprint (outline and interior): on a slope, a
   height above the local ground (as a DSM-derived raster gives it) belongs
   there, not at the lowest point. The top is anchor + height, lifted if
   needed so the roof clears the highest ground under the footprint by a
   storey (or the building's height, if lower), so no building is buried
   uphill. Walls reach down to the **base**, the lowest ground under the
   outline, so the downhill side stands on a plinth. A raised part
   (``minHeight_m``, else ``minFloor`` x storey height; ignored unless below
   the top) starts at anchor + min height; one with nothing beneath it whose
   lower levels are evidently unmapped is taken down to the ground
   (:func:`_groundUnsupported`).
   The footprint is converted to grid pixel coordinates at top height, so the
   roof sits exactly in its node columns.
4. **Roof shape:** mapped roof shapes become height fields
   (:mod:`map3d.layers.roofs`); the roof's highest point is the top.
5. **Rasterization:** any cell touched by a footprint gets the building
   (``all_touched``). Per node, building parts that stand on the ground or on
   each other (gaps under :data:`FLOATING_GAP_m`) form one solid column; where
   footprints overlap, the highest surface wins. Raised parts with open space
   beneath them are also reported as elevated levels (with the decks).
"""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pyproj
import shapely
from rasterio.features import rasterize

from map3d.gridGeometry import GridGeometry
from map3d.layers._levels import LevelEntries
from map3d.layers._vectors import (
    floatColumn,
    mostlyCovered,
    objectColumn,
    outlineGround,
    textColumn,
)
from map3d.layers.columns import ChunkColumns
from map3d.layers.groundFix import GroundCorrection
from map3d.layers.lofts import Loft, findLofts
from map3d.layers.pixels import burnChunk, chunkBox, rasterizePixels, toPixelGeometries
from map3d.layers.provenance import HeightSource
from map3d.layers.roofs import Roof, RoofShape, makeRoof, roofShapeCodes
from map3d.layers.terrain import GroundModel
from map3d.layers.wheels import Wheel, findWheels
from map3d.pack.reader import Pack, RasterWindow
from map3d.pack.registry import getSource

__all__ = [
    "DEFAULT_BUILDING_HEIGHT_m",
    "FLOATING_GAP_m",
    "UNSUPPORTED_MIN_THICKNESS_m",
    "INHERIT_OVERLAP",
    "MIN_RASTER_HEIGHT_m",
    "SMALL_DEFAULT_HEIGHT_m",
    "SMALL_FOOTPRINT_m2",
    "STOREY_HEIGHT_m",
    "BuildingChunk",
    "BuildingSet",
    "loadBuildings",
    "rasterizeBuildings",
]

FloatArray = npt.NDArray[np.float64]

STOREY_HEIGHT_m = 3.0
DEFAULT_BUILDING_HEIGHT_m = 9.0
SMALL_DEFAULT_HEIGHT_m = 3.0  # one storey: the default for sheds, kiosks, shelters
SMALL_FOOTPRINT_m2 = 50.0  # footprints under this get SMALL_DEFAULT_HEIGHT_m
MIN_RASTER_HEIGHT_m = 2.0  # raster heights below this are treated as missing
FLOATING_GAP_m = 2.0  # a raised part this close above what is under it rests on it
INHERIT_OVERLAP = 0.5  # mutual coverage for a dropped duplicate to hand over attributes
_NO_SURFACE_m = -1.0e4  # stands in for "nothing below" when stacking parts


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
        baseOrthometric_m: Lowest ground under the footprint outline (MSL): wall bottoms.
        roofOrthometric_m: Top, i.e. the roof's highest point (MSL).
        pixelGeometries: Footprints in grid pixel coordinates ``(col, row)`` at top height.
        minHeight_m: Bottom of a raised part above its base (0 = on the ground).
        roofShape: :class:`~map3d.layers.roofs.RoofShape` code (as mapped).
        roofs: Roof height field per building, ``None`` for flat roofs.
        facadeColor: Mapped facade colour (``#RRGGBB`` or a name), ``None`` if unmapped.
        roofColor: Mapped roof colour, ``None`` if unmapped.
        facadeMaterial: Mapped facade material (e.g. ``glass``), ``None`` if unmapped.
        roofMaterial: Mapped roof material (e.g. ``roof_tiles``), ``None`` if unmapped.
        anchorOrthometric_m: Median ground under the footprint, which heights
            are measured from (MSL); the base if not given.
        numFloors: Mapped storey count (NaN where unknown), for facade floor bands.
        buildingClass: Mapped building class (``office``, ``apartments``, ...), or ``None``.
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
    minHeight_m: FloatArray
    roofShape: npt.NDArray[np.uint8]
    roofs: tuple[Roof | None, ...]
    facadeColor: npt.NDArray[np.object_]
    roofColor: npt.NDArray[np.object_]
    facadeMaterial: npt.NDArray[np.object_]
    roofMaterial: npt.NDArray[np.object_]
    anchorOrthometric_m: FloatArray | None = None
    numFloors: FloatArray | None = None
    buildingClass: npt.NDArray[np.object_] | None = None
    groundCorrection: GroundCorrection | None = None

    def __len__(self) -> int:
        return int(self.height_m.size)

    @property
    def bottomOrthometric_m(self) -> FloatArray:
        """Underside of each building: its base, or anchor + min height for raised parts."""
        anchor = (
            self.baseOrthometric_m if self.anchorOrthometric_m is None else self.anchorOrthometric_m
        )
        result: FloatArray = np.where(
            self.minHeight_m > 0.0, anchor + self.minHeight_m, self.baseOrthometric_m
        )
        return result

    @property
    def floorHeight_m(self) -> FloatArray:
        """Storey height for facade floor bands: height / mapped storeys, else NaN."""
        floors = np.full(len(self), np.nan) if self.numFloors is None else self.numFloors
        with np.errstate(divide="ignore", invalid="ignore"):
            perFloor = self.height_m / floors
        result: FloatArray = np.where(
            np.isfinite(perFloor) & (floors > 0), np.clip(perFloor, 2.4, 6.0), np.nan
        )
        return result

    @functools.cached_property
    def lofts(self) -> list[Loft]:
        """Stacks of concentric round parts that widen upward (:mod:`map3d.layers.lofts`)."""
        return findLofts(self.geometries, self.bottomOrthometric_m, self.roofOrthometric_m)

    @functools.cached_property
    def wheels(self) -> list[Wheel]:
        """Observation wheels mapped as rows of spans (:mod:`map3d.layers.wheels`)."""
        return findWheels(self.geometries, self.bottomOrthometric_m, self.roofOrthometric_m)

    @functools.cached_property
    def wheelOf(self) -> dict[int, int]:
        """Building index -> index of the wheel it belongs to."""
        return {i: k for k, wheel in enumerate(self.wheels) for i in wheel.members}

    @functools.cached_property
    def loftOf(self) -> dict[int, int]:
        """Building index -> index of the loft it belongs to."""
        return {i: k for k, loft in enumerate(self.lofts) for i in (*loft.members, *loft.absorbed)}

    @property
    def eaveOrthometric_m(self) -> FloatArray:
        """Where walls meet the roof: the top minus the roof's rise."""
        rise = np.array([0.0 if r is None else r.rise_m for r in self.roofs])
        result: FloatArray = self.roofOrthometric_m - rise.reshape(self.roofOrthometric_m.shape)
        return result

    def topAt(self, index: int, lon_deg: npt.ArrayLike, lat_deg: npt.ArrayLike) -> FloatArray:
        """Orthometric roof surface of building ``index`` at geographic positions.

        Args:
            index: Building index.
            lon_deg: Longitudes.
            lat_deg: Latitudes.

        Returns:
            The roof height field (the flat top for flat roofs).
        """
        roof = self.roofs[index]
        top = float(self.roofOrthometric_m[index])
        if roof is None:
            return np.full(np.shape(lon_deg), top)
        result: FloatArray = top - roof.rise_m + roof.riseAt(lon_deg, lat_deg)
        return result

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
            "minHeight_m": self.minHeight_m.copy(),
            "roofShape": np.array(
                [RoofShape(c).name.lower() for c in self.roofShape], dtype=np.str_
            ),
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
            (:mod:`map3d.layers.groundFix`) before placing bases. Raised parts
            don't take part.

    Returns:
        The placed buildings (possibly empty).
    """
    bounds = geometry.footprintBounds_deg(margin_m=geometry.gridSpacing_m)
    fused = _fuse(pack, bounds)
    geoms = fused.geoms
    rasterHeight, rasterRes = _rasterHeights(pack, bounds, geoms)
    area_m2 = shapely.area(toPixelGeometries(geoms, 0.0, geometry)) * geometry.gridSpacing_m**2
    height, code, nativeRes = _resolveHeights(
        fused.heights, fused.floors, fused.authoritative, rasterHeight, rasterRes, area_m2
    )
    minHeight = _resolveMinHeights(fused.minHeights, fused.minFloors, height)
    minHeight = _groundUnsupported(geoms, minHeight, height)
    correction = (
        GroundCorrection.build(geometry, ground, geoms[minHeight <= 0.0]) if correctGround else None
    )
    if correction is not None:
        ground = dataclasses.replace(ground, correction=correction)
    base, anchor, highest = _footprintGround(geoms, ground)
    roof = anchor + height
    grounded = minHeight <= 0.0
    clearance = np.minimum(height, STOREY_HEIGHT_m)
    roof = np.where(grounded, np.fmax(roof, highest + clearance), roof)
    centroids = shapely.centroid(geoms)
    roofEll = roof + ground.undulation_m(shapely.get_y(centroids), shapely.get_x(centroids))
    pixel = toPixelGeometries(geoms, roofEll, geometry)
    gridBox = shapely.box(0.0, 0.0, geometry.nCols, geometry.nRows)
    keep = np.flatnonzero(np.isfinite(base) & shapely.intersects(pixel, gridBox))
    roofs = tuple(
        None
        if fused.roofShape[i] == RoofShape.FLAT
        else makeRoof(
            geoms[i],
            int(fused.roofShape[i]),
            float(roof[i] - anchor[i] - minHeight[i]),
            roofHeight_m=float(fused.roofHeights[i]),
            direction_deg=float(fused.roofDirections[i]),
            orientation=fused.roofOrientations[i],
        )
        for i in keep
    )
    return BuildingSet(
        geometries=geoms[keep],
        sourceIds=fused.sourceIds[keep],
        sourceFeatureIds=fused.featureIds[keep],
        height_m=height[keep],
        heightSource=code[keep],
        nativeResolution_m=nativeRes[keep],
        baseOrthometric_m=base[keep],
        roofOrthometric_m=roof[keep],
        anchorOrthometric_m=anchor[keep],
        numFloors=fused.floors[keep],
        buildingClass=fused.buildingClass[keep],
        pixelGeometries=pixel[keep],
        minHeight_m=minHeight[keep],
        roofShape=fused.roofShape[keep],
        roofs=roofs,
        facadeColor=fused.facadeColor[keep],
        roofColor=fused.roofColor[keep],
        facadeMaterial=fused.facadeMaterial[keep],
        roofMaterial=fused.roofMaterial[keep],
        groundCorrection=correction,
    )


@dataclass(frozen=True, eq=False)
class BuildingChunk:
    """Building rasterization of one chunk.

    Attributes:
        ids: Feature id (``index + 1``) of the highest building surface at each
            node; 0 where there is none.
        top: Orthometric height of that surface (NaN where there is none).
        levels: Raised parts with open space beneath them, as elevated levels.
    """

    ids: npt.NDArray[np.int64]
    top: FloatArray
    levels: LevelEntries


def rasterizeBuildings(
    buildings: BuildingSet, columns: ChunkColumns, groundH: FloatArray | None = None
) -> BuildingChunk:
    """Building surfaces and raised levels at a chunk's nodes.

    Parts on the ground are burned with the tallest top winning, and their roof
    height fields evaluated at the nodes. Each raised part is then stacked on
    what is under it: if its underside is within :data:`FLOATING_GAP_m` of the
    solid column (or the ground) it extends that column, otherwise it is an
    elevated level with open space beneath.

    Args:
        buildings: Placed buildings.
        columns: The chunk.
        groundH: Orthometric ground at the nodes; ``None`` treats every raised
            part over open ground as elevated.

    Returns:
        The chunk's building surfaces.
    """
    shape = columns.shape
    ids = np.zeros(shape, dtype=np.int64)
    top = np.full(shape, np.nan)
    if len(buildings) == 0:
        return BuildingChunk(ids, top, LevelEntries.empty())
    grounded = np.flatnonzero(buildings.minHeight_m <= 0.0)
    if grounded.size:
        burned = burnChunk(
            buildings.pixelGeometries[grounded],
            grounded + 1,
            columns,
            allTouched=True,
            dtype="int32",
            priority=buildings.roofOrthometric_m[grounded],
        )
        ids = np.asarray(burned, dtype=np.int64)
        top = _surfaceAt(buildings, columns, ids)
    raised = np.flatnonzero(buildings.minHeight_m > 0.0)
    raised = raised[shapely.intersects(buildings.pixelGeometries[raised], chunkBox(columns))]
    entries = _raisedEntries(buildings, columns, raised)
    if entries[0].size == 0:
        return BuildingChunk(ids, top, LevelEntries.empty())
    return _stackRaised(ids, top, groundH, *entries)


def _surfaceAt(
    buildings: BuildingSet, columns: ChunkColumns, ids: npt.NDArray[np.int64]
) -> FloatArray:
    """Roof surface of the building at each node (flat tops, or roof height fields)."""
    top = np.where(ids > 0, buildings.roofOrthometric_m[np.maximum(ids - 1, 0)], np.nan)
    shaped = [int(i) for i in np.unique(ids[ids > 0]) if buildings.roofs[i - 1] is not None]
    if shaped:
        lat, lon = columns.nodeLatLon()
        for i in shaped:
            cells = ids == i
            top[cells] = buildings.topAt(i - 1, lon[cells], lat[cells])
    return top


def _raisedEntries(
    buildings: BuildingSet, columns: ChunkColumns, indices: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.int64], FloatArray, FloatArray, npt.NDArray[np.int64]]:
    """``(flat node, top, underside, index + 1)`` of raised parts at the nodes they cover."""
    nr, nc = columns.shape
    flats: list[npt.NDArray[np.int64]] = [np.zeros(0, dtype=np.int64)]
    tops: list[FloatArray] = [np.zeros(0)]
    bottoms: list[FloatArray] = [np.zeros(0)]
    owners: list[npt.NDArray[np.int64]] = [np.zeros(0, dtype=np.int64)]
    for i in indices:
        mask = rasterizePixels(
            [(buildings.pixelGeometries[i], 1)],
            (nr, nc),
            columns.rows.start,
            columns.cols.start,
            allTouched=True,
            dtype="uint8",
        )
        rr, cc = np.nonzero(mask)
        lat, lon = columns.nodeLatLon()
        flats.append((rr * nc + cc).astype(np.int64))
        tops.append(buildings.topAt(int(i), lon[rr, cc], lat[rr, cc]))
        bottoms.append(np.full(rr.size, buildings.bottomOrthometric_m[i]))
        owners.append(np.full(rr.size, i + 1, dtype=np.int64))
    return (
        np.concatenate(flats),
        np.concatenate(tops),
        np.concatenate(bottoms),
        np.concatenate(owners),
    )


def _stackRaised(
    ids: npt.NDArray[np.int64],
    top: FloatArray,
    groundH: FloatArray | None,
    flat: npt.NDArray[np.int64],
    partTop: FloatArray,
    partBottom: FloatArray,
    owner: npt.NDArray[np.int64],
) -> BuildingChunk:
    """Merge raised parts into the solid columns they rest on; the rest become levels.

    Per node, the solid column (or the ground) and the raised parts are sorted
    by underside; a part whose underside is more than :data:`FLOATING_GAP_m`
    above everything before it starts a new group, i.e. an elevated level.
    """
    cells = np.unique(flat)
    solid = top.ravel()[cells]
    ground = np.full(cells.size, np.nan) if groundH is None else groundH.ravel()[cells]
    solid = np.where(np.isfinite(solid), solid, np.nan_to_num(ground, nan=_NO_SURFACE_m))
    # One baseline entry per node (its solid column, or the ground), then the parts.
    f = np.concatenate([cells, flat])
    lo = np.concatenate([np.full(cells.size, 2.0 * _NO_SURFACE_m), partBottom])
    hi = np.concatenate([solid, partTop])
    own = np.concatenate([ids.ravel()[cells], owner])
    order = np.lexsort((lo, f))
    f, lo, hi, own = f[order], lo[order], hi[order], own[order]
    isFirst = np.ones(f.size, dtype=bool)
    isFirst[1:] = f[1:] != f[:-1]
    offset = np.searchsorted(cells, f) * 1.0e6  # keeps the running max within each node
    reach = np.maximum.accumulate(hi + offset) - offset
    previous = np.concatenate([[-np.inf], reach[:-1]])
    newGroup = isFirst | (lo > previous + FLOATING_GAP_m)
    group = np.cumsum(newGroup) - 1
    starts = np.flatnonzero(newGroup)
    ends = np.append(starts[1:], f.size) - 1
    groupTop = np.maximum.reduceat(hi, starts)
    groupBottom = lo[starts]
    groupFlat = f[starts]
    groupOwner = own[np.lexsort((hi, group))[ends]]
    elevated = ~isFirst[starts]
    levels = LevelEntries(groupFlat[elevated], groupTop[elevated], groupBottom[elevated])
    # The highest surface at each node, and the building it belongs to.
    byNode = np.lexsort((groupTop, groupFlat))
    last = np.ones(byNode.size, dtype=bool)
    last[:-1] = groupFlat[byNode][1:] != groupFlat[byNode][:-1]
    pick = byNode[last]
    ids = ids.copy()
    top = top.copy()
    ids.ravel()[groupFlat[pick]] = groupOwner[pick]
    top.ravel()[groupFlat[pick]] = np.where(groupOwner[pick] > 0, groupTop[pick], np.nan)
    return BuildingChunk(ids, top, levels)


@dataclass(frozen=True, eq=False)
class _Fused:
    geoms: npt.NDArray[np.object_]
    sourceIds: npt.NDArray[np.str_]
    featureIds: npt.NDArray[np.str_]
    heights: FloatArray
    floors: FloatArray
    authoritative: npt.NDArray[np.bool_]
    minHeights: FloatArray
    minFloors: FloatArray
    roofShape: npt.NDArray[np.uint8]
    roofHeights: FloatArray
    roofDirections: FloatArray
    roofOrientations: npt.NDArray[np.object_]
    facadeColor: npt.NDArray[np.object_]
    roofColor: npt.NDArray[np.object_]
    facadeMaterial: npt.NDArray[np.object_]
    roofMaterial: npt.NDArray[np.object_]
    buildingClass: npt.NDArray[np.object_]


def _fuse(pack: Pack, bounds: tuple[float, float, float, float]) -> _Fused:
    geoms: list[Any] = []
    cols: dict[str, list[Any]] = {k: [] for k in _Fused.__dataclass_fields__ if k != "geoms"}
    for sourceId in pack.sourcesFor("buildings"):
        selection = pack.readVectors(sourceId, bounds)
        candidates = shapely.make_valid(selection.geometries)
        keep = shapely.area(candidates) > 0
        table = selection.table
        if geoms:
            accepted = np.array(geoms, dtype=object)
            duplicate = keep & mostlyCovered(candidates, accepted)
            keep &= ~duplicate
            _inherit(cols, accepted, candidates, np.flatnonzero(duplicate), table)
        idx = np.flatnonzero(keep)
        geoms.extend(candidates[idx])
        cols["sourceIds"].extend([sourceId] * idx.size)
        cols["featureIds"].extend(textColumn(table, "sourceFeatureId")[idx])
        cols["heights"].extend(floatColumn(table, "height_m")[idx])
        cols["floors"].extend(floatColumn(table, "numFloors")[idx])
        cols["authoritative"].extend([getSource(sourceId).authoritative] * idx.size)
        cols["minHeights"].extend(_optionalFloat(table, "minHeight_m")[idx])
        cols["minFloors"].extend(_optionalFloat(table, "minFloor")[idx])
        cols["roofShape"].extend(roofShapeCodes(_optionalObject(table, "roofShape"))[idx])
        cols["roofHeights"].extend(_optionalFloat(table, "roofHeight_m")[idx])
        cols["roofDirections"].extend(_optionalFloat(table, "roofDirection_deg")[idx])
        cols["roofOrientations"].extend(_optionalObject(table, "roofOrientation")[idx])
        cols["facadeColor"].extend(_optionalObject(table, "facadeColor")[idx])
        cols["roofColor"].extend(_optionalObject(table, "roofColor")[idx])
        cols["facadeMaterial"].extend(_optionalObject(table, "facadeMaterial")[idx])
        cols["roofMaterial"].extend(_optionalObject(table, "roofMaterial")[idx])
        cols["buildingClass"].extend(_optionalObject(table, "buildingClass")[idx])
    return _Fused(
        geoms=np.array(geoms, dtype=object),
        sourceIds=np.array(cols["sourceIds"], dtype=np.str_),
        featureIds=np.array(cols["featureIds"], dtype=np.str_),
        heights=np.array(cols["heights"], dtype=np.float64),
        floors=np.array(cols["floors"], dtype=np.float64),
        authoritative=np.array(cols["authoritative"], dtype=bool),
        minHeights=np.array(cols["minHeights"], dtype=np.float64),
        minFloors=np.array(cols["minFloors"], dtype=np.float64),
        roofShape=np.array(cols["roofShape"], dtype=np.uint8),
        roofHeights=np.array(cols["roofHeights"], dtype=np.float64),
        roofDirections=np.array(cols["roofDirections"], dtype=np.float64),
        roofOrientations=np.array(cols["roofOrientations"], dtype=object),
        facadeColor=np.array(cols["facadeColor"], dtype=object),
        roofColor=np.array(cols["roofColor"], dtype=object),
        facadeMaterial=np.array(cols["facadeMaterial"], dtype=object),
        roofMaterial=np.array(cols["roofMaterial"], dtype=object),
        buildingClass=np.array(cols["buildingClass"], dtype=object),
    )


# _Fused columns handed from a dropped duplicate to the feature that replaced it,
# as groups that travel together: (columns, the table column that holds the first).
_INHERITED = (
    (("roofShape", "roofHeights", "roofDirections", "roofOrientations"), "roofShape"),
    (("facadeColor",), "facadeColor"),
    (("roofColor",), "roofColor"),
    (("facadeMaterial",), "facadeMaterial"),
    (("roofMaterial",), "roofMaterial"),
)
_SOURCE_COLUMN = {
    "roofShape": "roofShape",
    "roofHeights": "roofHeight_m",
    "roofDirections": "roofDirection_deg",
    "roofOrientations": "roofOrientation",
    "facadeColor": "facadeColor",
    "roofColor": "roofColor",
    "facadeMaterial": "facadeMaterial",
    "roofMaterial": "roofMaterial",
}


def _inherit(
    cols: dict[str, list[Any]],
    accepted: npt.NDArray[np.object_],
    candidates: npt.NDArray[np.object_],
    dropped: npt.NDArray[np.int64],
    table: Any,
) -> None:
    """Hand appearance attributes from dropped duplicates to the features that replaced them."""
    if dropped.size == 0:
        return
    cand, acc = shapely.STRtree(accepted).query(candidates[dropped], predicate="intersects")
    if cand.size == 0:
        return
    overlap = shapely.area(shapely.intersection(candidates[dropped[cand]], accepted[acc]))
    order = np.lexsort((overlap, cand))
    last = np.append(cand[order][1:] != cand[order][:-1], True)
    best = order[last]  # each dropped feature's largest overlap
    k, target, area = dropped[cand[best]], acc[best], overlap[best]
    same = (area >= INHERIT_OVERLAP * shapely.area(candidates[k])) & (
        area >= INHERIT_OVERLAP * shapely.area(accepted[target])
    )
    values: dict[str, npt.NDArray[Any]] = {
        name: _optionalObject(table, column) for name, column in _SOURCE_COLUMN.items()
    }
    values["roofShape"] = roofShapeCodes(values["roofShape"])
    for columns, lead in _INHERITED:
        for i, j in zip(k[same], target[same], strict=True):
            if _present(values[lead][i]) and not _present(cols[lead][j]):
                for name in columns:
                    cols[name][j] = _converted(name, values[name][i])


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float | np.floating):
        return bool(np.isfinite(value))
    if isinstance(value, int | np.integer):
        return int(value) != RoofShape.FLAT
    return bool(str(value).strip())


def _converted(name: str, value: Any) -> Any:
    if name in ("roofHeights", "roofDirections"):
        return np.nan if value is None else float(value)
    return value


UNSUPPORTED_MIN_THICKNESS_m = 6.0  # unsupported raised parts this thick reach the ground
_SUPPORT_SHARE = 0.05  # a raised part with less than this much beneath it is unsupported
_MAX_ELONGATION = 4.0  # longer, thinner raised parts (link bridges) may span open space


def _groundUnsupported(
    geoms: npt.NDArray[np.object_], minHeight: FloatArray, height: FloatArray
) -> FloatArray:
    """Extend raised parts with nothing beneath them down to the ground.

    A raised part needs something under it: another part reaching up to within
    :data:`FLOATING_GAP_m` of its underside over at least 5 % of its footprint.
    Without that, the building's lower levels are simply unmapped (e.g. a tower
    part from 45 m whose podium is not mapped), so a part at least
    :data:`UNSUPPORTED_MIN_THICKNESS_m` thick that is not long and thin is taken
    down to the ground. Thin parts (canopies) and long thin parts (link bridges)
    stay raised; so do stacks drawn as lofts, which get their own trunk, and
    the spans of observation wheels.

    Heights here are relative to each part's own base, which is close enough to
    compare neighbouring parts.
    """
    raised = np.flatnonzero(minHeight > 0.0)
    if raised.size == 0:
        return minHeight
    lofted = {
        i for loft in findLofts(geoms, minHeight, height) for i in (*loft.members, *loft.absorbed)
    }
    lofted |= {i for wheel in findWheels(geoms, minHeight, height) for i in wheel.members}
    tree = shapely.STRtree(geoms)
    out = minHeight.copy()
    for i in raised:
        if i in lofted or height[i] - minHeight[i] < UNSUPPORTED_MIN_THICKNESS_m:
            continue
        if _isBridgeLike(geoms[i]):
            continue
        near = tree.query(geoms[i], predicate="intersects")
        below = near[
            (near != i)
            & (height[near] >= minHeight[i] - FLOATING_GAP_m)
            & (minHeight[near] < minHeight[i] - 0.5)
        ]
        if below.size:
            support = shapely.area(shapely.intersection(geoms[i], shapely.union_all(geoms[below])))
            if support >= _SUPPORT_SHARE * shapely.area(geoms[i]):
                continue
        out[i] = 0.0
    return out


_BRIDGE_MAX_WIDTH_m = 15.0


def _isBridgeLike(geom: Any) -> bool:
    """Long and narrow: more than 4x as long as wide, and under 15 m wide."""
    rect = shapely.get_coordinates(shapely.oriented_envelope(geom))
    if rect.shape[0] < 4:
        return True
    scale = np.array([np.cos(np.radians(rect[0, 1])), 1.0]) * 111_320.0
    sides = np.hypot(*(np.diff(rect[:3], axis=0) * scale).T)
    width = float(sides.min())
    return width < _BRIDGE_MAX_WIDTH_m and float(sides.max()) > _MAX_ELONGATION * max(width, 1e-9)


def _optionalFloat(table: Any, name: str) -> FloatArray:
    if name in table.column_names:
        return floatColumn(table, name)
    return np.full(table.num_rows, np.nan)


def _optionalObject(table: Any, name: str) -> npt.NDArray[np.object_]:
    if name in table.column_names:
        return objectColumn(table, name)
    return np.full(table.num_rows, None, dtype=object)


def _resolveMinHeights(
    minHeights: FloatArray, minFloors: FloatArray, height: FloatArray
) -> FloatArray:
    """Bottom of each part above its base: mapped min height, else min floor x storey height.

    A min height that doesn't leave the part at least a metre tall is ignored.
    """
    fromFloors = np.where(
        np.isfinite(minFloors) & (minFloors > 0), minFloors * STOREY_HEIGHT_m, 0.0
    )
    value = np.where(np.isfinite(minHeights) & (minHeights > 0), minHeights, fromFloors)
    result: FloatArray = np.where(value <= height - 1.0, value, 0.0)
    return result


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
    area_m2: FloatArray,
) -> tuple[FloatArray, npt.NDArray[np.uint8], FloatArray]:
    """Pick each building's height by source precedence.

    Raster heights under :data:`MIN_RASTER_HEIGHT_m` are treated as missing
    (the raster saw ground, not roof). Buildings with no usable source get
    :data:`SMALL_DEFAULT_HEIGHT_m` when the footprint is under
    :data:`SMALL_FOOTPRINT_m2`, else :data:`DEFAULT_BUILDING_HEIGHT_m`.
    """
    hasHeight = np.isfinite(heights) & (heights > 0)
    hasFloors = np.isfinite(floors) & (floors > 0)
    hasRaster = np.isfinite(rasterHeight) & (rasterHeight >= MIN_RASTER_HEIGHT_m)
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
    height = np.where(
        area_m2 < SMALL_FOOTPRINT_m2, SMALL_DEFAULT_HEIGHT_m, DEFAULT_BUILDING_HEIGHT_m
    )
    code = np.full(heights.size, HeightSource.DEFAULT_HEIGHT, dtype=np.uint8)
    nativeRes = np.zeros(heights.size)
    for condition, value, source, res in reversed(rules):
        height = np.where(condition, value, height)
        code = np.where(condition, np.uint8(source), code).astype(np.uint8)
        nativeRes = np.where(condition, res, nativeRes)
    return height, code, nativeRes


def _footprintGround(
    geoms: npt.NDArray[np.object_], ground: GroundModel
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``(base, anchor, highest)``: lowest outline ground, median and highest ground under each footprint.

    The median and maximum use the densified outline plus an interior grid of
    about :data:`_INTERIOR_STEP_deg` (at most :data:`_MAX_INTERIOR_SAMPLES` points).
    """
    n = geoms.size
    base = np.full(n, np.inf)
    values, owner = outlineGround(geoms, ground)
    finite = np.isfinite(values)
    np.fmin.at(base, owner, np.where(finite, values, np.inf))
    lat, lon, inOwner = _interiorPoints(geoms)
    inside = ground.orthometricAt(lat, lon) if lat.size else np.zeros(0)
    allValues = np.concatenate([values, inside])
    allOwner = np.concatenate([owner, inOwner])
    ok = np.isfinite(allValues)
    allValues, allOwner = allValues[ok], allOwner[ok]
    highest = np.full(n, -np.inf)
    np.fmax.at(highest, allOwner, allValues)
    anchor = np.full(n, np.nan)
    if allValues.size:
        order = np.lexsort((allValues, allOwner))
        sortedOwner, sortedValues = allOwner[order], allValues[order]
        owners, start, count = np.unique(sortedOwner, return_index=True, return_counts=True)
        anchor[owners] = 0.5 * (
            sortedValues[start + (count - 1) // 2] + sortedValues[start + count // 2]
        )
    base = np.where(np.isfinite(base), base, np.nan)
    anchor = np.fmax(np.where(np.isfinite(anchor), anchor, base), base)
    highest = np.where(np.isfinite(highest), highest, base)
    return base, anchor, highest


_INTERIOR_STEP_deg = 4.5e-5  # ~5 m
_MAX_INTERIOR_SAMPLES = 400


def _interiorPoints(
    geoms: npt.NDArray[np.object_],
) -> tuple[FloatArray, FloatArray, npt.NDArray[np.int64]]:
    """``(lat, lon, owner)`` of grid points inside each footprint."""
    lats: list[FloatArray] = [np.zeros(0)]
    lons: list[FloatArray] = [np.zeros(0)]
    owners: list[npt.NDArray[np.int64]] = [np.zeros(0, dtype=np.int64)]
    bounds = shapely.bounds(geoms)
    for i, poly in enumerate(geoms):
        x0, y0, x1, y1 = bounds[i]
        step = max(
            _INTERIOR_STEP_deg,
            float(np.sqrt((x1 - x0) * (y1 - y0) / _MAX_INTERIOR_SAMPLES)),
        )
        xs = np.arange(x0 + step / 2, x1, step)
        ys = np.arange(y0 + step / 2, y1, step)
        if xs.size == 0 or ys.size == 0:
            continue
        gx, gy = np.meshgrid(xs, ys)
        inside = shapely.contains_xy(poly, gx, gy)
        lons.append(gx[inside])
        lats.append(gy[inside])
        owners.append(np.full(int(inside.sum()), i, dtype=np.int64))
    return np.concatenate(lats), np.concatenate(lons), np.concatenate(owners)
