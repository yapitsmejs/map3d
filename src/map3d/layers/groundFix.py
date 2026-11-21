"""Remove building residue from the bare-earth ground model under buildings.

FABDEM removes buildings from the Copernicus 30 m DSM with machine learning.
For some large buildings (depots, warehouses) the removal fails, leaving a
building-shaped mound in the "ground". It spills about 10-15 m past the walls
at 30 m resolution, and can be taller than the building's own roof (e.g.
Loyang Bus Depot: ~20 m mound, 13 m building).

For each building footprint:

1. **Reference plane:** sample the ground on rings 20-45 m outside and fit a
   plane robustly, iteratively dropping points more than 1 m above the fit
   (neighbouring mounds, embankments).
2. **Trigger:** correct the building only if its interior ground clearly
   stands above that plane (90th-percentile excess > :data:`TRIGGER_m`).
3. **Zone:** start from the footprint buffered by :data:`ZONE_BUFFER_m`, then
   grow across connected ground standing more than :data:`GROW_EXCESS_m` above
   the plane, up to :data:`MAX_GROW_m` from the walls. Residue often spills
   further than the walls, or beyond the mapped outline.
4. **Correction:** inside the zone, ground = min(ground, plane), fading back to
   the raw ground over the last :data:`TAPER_m` of the reach. Ground is only
   ever lowered and stays continuous: zone edges sit where the residue fades out
   or within the taper.

Everything is in scene-centre ENU metres; heights are orthometric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import shapely
from rasterio.features import shapes
from rasterio.transform import Affine

from map3d.frames import LocalFrame
from map3d.gridGeometry import GridGeometry
from map3d.layers.pixels import chunkBox, rasterizePixels

if TYPE_CHECKING:
    from map3d.layers.columns import ChunkColumns
    from map3d.layers.terrain import GroundModel

__all__ = [
    "GROW_EXCESS_m",
    "MAX_GROW_m",
    "TAPER_m",
    "RING_INNER_m",
    "RING_OUTER_m",
    "TRIGGER_m",
    "ZONE_BUFFER_m",
    "GroundCorrection",
]

FloatArray = npt.NDArray[np.float64]

ZONE_BUFFER_m = 10.0
RING_INNER_m = 20.0
RING_OUTER_m = 45.0
TRIGGER_m = 1.0
GROW_EXCESS_m = 0.5
MAX_GROW_m = 60.0
TAPER_m = 20.0  # correction fades out over the last TAPER_m of the reach
_GROW_STEP_m = 2.5
_RING_STEP_m = 5.0
_INTERIOR_STEP_m = 5.0
_ABOVE_FIT_m = 1.0
_FIT_ITERATIONS = 3


@dataclass(frozen=True, eq=False)
class GroundCorrection:
    """Per-building ground planes that cap the ground model under buildings.

    Attributes:
        geometry: Scene grid geometry (its frame is the ENU frame used here).
        zones: Corrected zones (grown from footprint + buffer) as ENU polygons.
        footprints: The triggering buildings' footprints as ENU polygons (taper distance).
        pixelZones: The same zones in grid pixel space.
        coeffs: Plane ``H = a + b*e + c*n`` per zone, shape ``(Z, 3)``.
        maxLowering_m: Largest interior excess removed per zone.
    """

    geometry: GridGeometry
    zones: npt.NDArray[np.object_]
    footprints: npt.NDArray[np.object_]
    pixelZones: npt.NDArray[np.object_]
    coeffs: FloatArray
    maxLowering_m: FloatArray

    @classmethod
    def build(
        cls, geometry: GridGeometry, ground: GroundModel, footprints_deg: npt.NDArray[np.object_]
    ) -> GroundCorrection:
        """Detect residual mounds under building footprints and fit ground planes.

        Args:
            geometry: Scene grid geometry.
            ground: Uncorrected ground model.
            footprints_deg: Building footprints in lon/lat.

        Returns:
            The correction (possibly with no zones).
        """
        frame = geometry.frame
        empty = cls(
            geometry,
            np.array([], dtype=object),
            np.array([], dtype=object),
            np.array([], dtype=object),
            np.zeros((0, 3)),
            np.zeros(0),
        )
        if footprints_deg.size == 0:
            return empty
        enu = _toEnu(frame, footprints_deg)
        ringPts, ringOwner = _ringSamples(enu)
        ringH = _groundAtEnu(frame, ground, ringPts)
        inPts, inOwner = _interiorSamples(enu)
        inH = _groundAtEnu(frame, ground, inPts)

        zones: list[Any] = []
        coeffs: list[FloatArray] = []
        lowering: list[float] = []
        ringSlices = _groupSlices(ringOwner, enu.size)
        inSlices = _groupSlices(inOwner, enu.size)
        ringOrder = np.argsort(ringOwner, kind="stable")
        inOrder = np.argsort(inOwner, kind="stable")
        ringPts, ringH = ringPts[ringOrder], ringH[ringOrder]
        inPts, inH = inPts[inOrder], inH[inOrder]
        for i in range(enu.size):
            plane = _robustPlane(ringPts[ringSlices[i]], ringH[ringSlices[i]])
            mine = inSlices[i]
            if plane is None or mine.stop == mine.start:
                continue
            excess = inH[mine] - _evaluate(plane, inPts[mine])
            excess = excess[np.isfinite(excess)]
            if excess.size == 0 or np.percentile(excess, 90) <= TRIGGER_m:
                continue
            zones.append(enu[i])  # grown into the final zone below
            coeffs.append(plane)
            lowering.append(float(excess.max()))
        if not zones:
            return empty
        footprintArray = np.array(zones, dtype=object)
        zoneArray = _growZones(frame, ground, footprintArray, np.array(coeffs))
        return cls(
            geometry,
            zoneArray,
            footprintArray,
            shapely.transform(zoneArray, lambda xy: np.column_stack(geometry.enuToPixel(*xy.T))),
            np.array(coeffs),
            np.array(lowering),
        )

    def correctPoints(
        self, lat_deg: npt.ArrayLike, lon_deg: npt.ArrayLike, heights: FloatArray
    ) -> FloatArray:
        """Cap ground heights at points inside corrected zones (NaN stays NaN).

        Args:
            lat_deg: Latitude of each point.
            lon_deg: Longitude of each point.
            heights: Orthometric ground heights at the points.

        Returns:
            Corrected heights (a new array).
        """
        out = np.array(heights, dtype=np.float64, copy=True)
        if self.zones.size == 0:
            return out
        e, n, _ = self.geometry.frame.geodeticToEnu(lat_deg, lon_deg, 0.0)
        e, n = np.ravel(e), np.ravel(n)
        pointIdx, zoneIdx = shapely.STRtree(self.zones).query(
            shapely.points(e, n), predicate="intersects"
        )
        if pointIdx.size:
            c = self.coeffs[zoneIdx]
            plane = c[:, 0] + c[:, 1] * e[pointIdx] + c[:, 2] * n[pointIdx]
            dist = shapely.distance(
                self.footprints[zoneIdx], shapely.points(e[pointIdx], n[pointIdx])
            )
            capped = _taperedCap(out.reshape(-1)[pointIdx], plane, dist)
            with np.errstate(invalid="ignore"):  # NaN (no data) stays NaN by design
                np.minimum.at(out.reshape(-1), pointIdx, capped)
        return out

    def correctChunk(self, columns: ChunkColumns, groundH: FloatArray) -> FloatArray:
        """Cap a chunk of the ground layer (node centres inside zones).

        Args:
            columns: The chunk.
            groundH: Chunk-shaped orthometric ground.

        Returns:
            Corrected ground (a new array).
        """
        out = np.array(groundH, dtype=np.float64, copy=True)
        if self.zones.size == 0:
            return out
        hits = np.flatnonzero(shapely.intersects(self.pixelZones, chunkBox(columns)))
        if hits.size == 0:
            return out
        r0, c0 = columns.rows.start, columns.cols.start
        e, n = self.geometry.nodesEnu(columns.rows, columns.cols)
        for z in hits:
            mask = (
                rasterizePixels(
                    [(self.pixelZones[z], 1)], out.shape, r0, c0, allTouched=False, dtype="uint8"
                )
                > 0
            )
            if mask.any():
                a, b, c = self.coeffs[z]
                dist = shapely.distance(self.footprints[z], shapely.points(e[mask], n[mask]))
                capped = _taperedCap(out[mask], a + b * e[mask] + c * n[mask], dist)
                out[mask] = np.minimum(out[mask], capped)
        return out

    def summary(self) -> dict[str, Any]:
        """Statistics for ``scene.json``."""
        return {
            "buildingsCorrected": int(self.coeffs.shape[0]),
            "maxLowering_m": float(self.maxLowering_m.max()) if self.maxLowering_m.size else 0.0,
            "correctedArea_m2": float(shapely.area(shapely.union_all(self.zones)))
            if self.zones.size
            else 0.0,
            "method": (
                "ground under buildings capped by a robust plane fitted to ground "
                f"{RING_INNER_m:g}-{RING_OUTER_m:g} m outside the footprint, when the interior stands "
                f"> {TRIGGER_m:g} m above it; the zone (footprint + {ZONE_BUFFER_m:g} m) grows across "
                f"connected ground > {GROW_EXCESS_m:g} m above the plane, up to {MAX_GROW_m:g} m out"
            ),
        }


def _taperedCap(heights: FloatArray, plane: FloatArray, dist_m: FloatArray) -> FloatArray:
    """Step 4 of the module docstring: ``plane + w * (h - plane)`` above the plane."""
    w = np.clip((dist_m - (MAX_GROW_m - TAPER_m)) / TAPER_m, 0.0, 1.0)
    with np.errstate(invalid="ignore"):
        result: FloatArray = np.where(heights > plane, plane + w * (heights - plane), heights)
    return result


def _growZones(
    frame: LocalFrame, ground: GroundModel, footprints: npt.NDArray[np.object_], coeffs: FloatArray
) -> npt.NDArray[np.object_]:
    """Step 3 of the module docstring, on a :data:`_GROW_STEP_m` lattice (4-connected)."""
    grids: list[tuple[FloatArray, FloatArray]] = []
    pts: list[FloatArray] = []
    for poly in footprints:
        x0, y0, x1, y1 = poly.bounds
        xs = np.arange(x0 - MAX_GROW_m, x1 + MAX_GROW_m + _GROW_STEP_m, _GROW_STEP_m)
        ys = np.arange(y0 - MAX_GROW_m, y1 + MAX_GROW_m + _GROW_STEP_m, _GROW_STEP_m)
        gx, gy = np.meshgrid(xs, ys)
        grids.append((xs, ys))
        pts.append(np.column_stack([gx.ravel(), gy.ravel()]))
    allPts = np.concatenate(pts)
    allH = _groundAtEnu(frame, ground, allPts)
    zones: list[Any] = []
    offset = 0
    for poly, plane, (xs, ys), cellPts in zip(footprints, coeffs, grids, pts, strict=True):
        h = allH[offset : offset + cellPts.shape[0]].reshape(ys.size, xs.size)
        offset += cellPts.shape[0]
        gx, gy = cellPts[:, 0].reshape(h.shape), cellPts[:, 1].reshape(h.shape)
        seedPoly = shapely.buffer(poly, ZONE_BUFFER_m)
        seed = shapely.contains_xy(seedPoly, gx, gy)
        reach = shapely.contains_xy(shapely.buffer(poly, MAX_GROW_m), gx, gy)
        with np.errstate(invalid="ignore"):
            residue = (h - (plane[0] + plane[1] * gx + plane[2] * gy)) > GROW_EXCESS_m
        grown = _floodFill(seed, seed | (residue & reach))
        zone = seedPoly
        if (grown & ~seed).any():
            transform = Affine(
                _GROW_STEP_m,
                0.0,
                xs[0] - _GROW_STEP_m / 2,
                0.0,
                _GROW_STEP_m,
                ys[0] - _GROW_STEP_m / 2,
            )
            cells = [
                shapely.geometry.shape(g)
                for g, _ in shapes(grown.astype(np.uint8), mask=grown, transform=transform)
            ]
            zone = shapely.make_valid(shapely.union_all([seedPoly, *cells]))
        zones.append(zone)
    return np.array(zones, dtype=object)


def _floodFill(
    seed: npt.NDArray[np.bool_], allowed: npt.NDArray[np.bool_]
) -> npt.NDArray[np.bool_]:
    """Cells of ``allowed`` 4-connected to ``seed`` (iterative constrained dilation)."""
    grown = seed & allowed
    while True:
        nxt = grown.copy()
        nxt[1:, :] |= grown[:-1, :]
        nxt[:-1, :] |= grown[1:, :]
        nxt[:, 1:] |= grown[:, :-1]
        nxt[:, :-1] |= grown[:, 1:]
        nxt &= allowed
        if np.array_equal(nxt, grown):
            return grown
        grown = nxt


def _toEnu(frame: LocalFrame, geoms: npt.NDArray[np.object_]) -> npt.NDArray[np.object_]:
    def toEnu(xy: FloatArray) -> FloatArray:
        e, n, _ = frame.geodeticToEnu(xy[:, 1], xy[:, 0], 0.0)
        return np.column_stack([e, n])

    return np.asarray(shapely.make_valid(shapely.transform(geoms, toEnu)), dtype=object)


def _ringSamples(enu: npt.NDArray[np.object_]) -> tuple[FloatArray, npt.NDArray[np.int64]]:
    pts: list[FloatArray] = []
    owners: list[npt.NDArray[np.int64]] = []
    for d in np.linspace(RING_INNER_m, RING_OUTER_m, 3):
        rings = shapely.segmentize(shapely.boundary(shapely.buffer(enu, d)), _RING_STEP_m)
        coords, owner = shapely.get_coordinates(rings, return_index=True)
        pts.append(coords)
        owners.append(owner)
    return np.concatenate(pts), np.concatenate(owners)


def _interiorSamples(enu: npt.NDArray[np.object_]) -> tuple[FloatArray, npt.NDArray[np.int64]]:
    pts: list[FloatArray] = []
    owners: list[npt.NDArray[np.int64]] = []
    for i, poly in enumerate(enu):
        x0, y0, x1, y1 = poly.bounds
        xs = np.arange(x0 + _INTERIOR_STEP_m / 2, x1, _INTERIOR_STEP_m)
        ys = np.arange(y0 + _INTERIOR_STEP_m / 2, y1, _INTERIOR_STEP_m)
        if xs.size == 0 or ys.size == 0:
            c = shapely.point_on_surface(poly)
            xs, ys = np.array([c.x]), np.array([c.y])
        gx, gy = np.meshgrid(xs, ys)
        inside = shapely.contains_xy(poly, gx, gy)
        if not inside.any():
            c = shapely.point_on_surface(poly)
            gx, gy, inside = np.array([c.x]), np.array([c.y]), np.array([True])
        pts.append(np.column_stack([gx[inside], gy[inside]]))
        owners.append(np.full(int(inside.sum()), i, dtype=np.int64))
    return np.concatenate(pts), np.concatenate(owners)


def _groundAtEnu(frame: LocalFrame, ground: GroundModel, pts: FloatArray) -> FloatArray:
    lat, lon, _ = frame.columnPointAtHeight(pts[:, 0], pts[:, 1], 0.0)
    return ground.orthometricAt(lat, lon)


def _robustPlane(pts: FloatArray, h: FloatArray) -> FloatArray | None:
    ok = np.isfinite(h)
    pts, h = pts[ok], h[ok]
    if h.size < 3:
        return None
    keep = np.ones(h.size, dtype=bool)
    plane = None
    for _ in range(_FIT_ITERATIONS):
        if keep.sum() < 3:
            break
        design = np.column_stack([np.ones(int(keep.sum())), pts[keep]])
        plane, *_ = np.linalg.lstsq(design, h[keep], rcond=None)
        keep = (h - _evaluate(plane, pts)) <= _ABOVE_FIT_m
    return plane


def _groupSlices(owner: npt.NDArray[np.int64], count: int) -> list[slice]:
    """Slices of each owner's samples after a stable sort by owner (linear, not O(B*N))."""
    bounds = np.searchsorted(np.sort(owner, kind="stable"), np.arange(count + 1))
    return [slice(int(bounds[i]), int(bounds[i + 1])) for i in range(count)]


def _evaluate(plane: FloatArray, pts: FloatArray) -> FloatArray:
    result: FloatArray = plane[0] + plane[1] * pts[:, 0] + plane[2] * pts[:, 1]
    return result
