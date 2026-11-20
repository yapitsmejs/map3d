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
3. **Correction:** within the footprint buffered by :data:`ZONE_BUFFER_m`, set
   ground = min(ground, plane). Ground is only ever lowered.

Everything is in scene-centre ENU metres; heights are orthometric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import shapely

from map3d.frames import LocalFrame
from map3d.gridGeometry import GridGeometry
from map3d.layers.pixels import rasterizePixels

if TYPE_CHECKING:
    from map3d.layers.terrain import GroundModel

__all__ = ["RING_INNER_m", "RING_OUTER_m", "TRIGGER_m", "ZONE_BUFFER_m", "GroundCorrection"]

FloatArray = npt.NDArray[np.float64]

ZONE_BUFFER_m = 10.0
RING_INNER_m = 20.0
RING_OUTER_m = 45.0
TRIGGER_m = 1.0
_RING_STEP_m = 5.0
_INTERIOR_STEP_m = 5.0
_ABOVE_FIT_m = 1.0
_FIT_ITERATIONS = 3


@dataclass(frozen=True, eq=False)
class GroundCorrection:
    """Per-building ground planes that cap the ground model under buildings.

    Attributes:
        frame: Scene ENU frame.
        geometry: Scene grid geometry.
        zones: Corrected zones (footprint + buffer) as ENU polygons.
        pixelZones: The same zones in grid pixel space.
        coeffs: Plane ``H = a + b*e + c*n`` per zone, shape ``(Z, 3)``.
        buildingIndex: Index of each zone's building in the input array.
        maxLowering_m: Largest interior excess removed per zone.
    """

    frame: LocalFrame
    geometry: GridGeometry
    zones: npt.NDArray[np.object_]
    pixelZones: npt.NDArray[np.object_]
    coeffs: FloatArray
    buildingIndex: npt.NDArray[np.int64]
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
            frame,
            geometry,
            np.array([], dtype=object),
            np.array([], dtype=object),
            np.zeros((0, 3)),
            np.zeros(0, dtype=np.int64),
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
        index: list[int] = []
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
            zones.append(shapely.buffer(enu[i], ZONE_BUFFER_m))
            coeffs.append(plane)
            index.append(i)
            lowering.append(float(excess.max()))
        if not zones:
            return empty
        zoneArray = np.array(zones, dtype=object)
        return cls(
            frame,
            geometry,
            zoneArray,
            _enuToPixel(geometry, zoneArray),
            np.array(coeffs),
            np.array(index, dtype=np.int64),
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
        e, n, _ = self.frame.geodeticToEnu(lat_deg, lon_deg, 0.0)
        e, n = np.ravel(e), np.ravel(n)
        pointIdx, zoneIdx = shapely.STRtree(self.zones).query(
            shapely.points(e, n), predicate="intersects"
        )
        if pointIdx.size:
            c = self.coeffs[zoneIdx]
            plane = c[:, 0] + c[:, 1] * e[pointIdx] + c[:, 2] * n[pointIdx]
            with np.errstate(invalid="ignore"):  # NaN (no data) stays NaN by design
                np.minimum.at(out.reshape(-1), pointIdx, plane)
        return out

    def correctChunk(self, rows: slice, cols: slice, groundH: FloatArray) -> FloatArray:
        """Cap a chunk of the ground layer (node centres inside zones).

        Args:
            rows: Row slice of the chunk.
            cols: Column slice of the chunk.
            groundH: Chunk-shaped orthometric ground.

        Returns:
            Corrected ground (a new array).
        """
        out = np.array(groundH, dtype=np.float64, copy=True)
        if self.zones.size == 0:
            return out
        r0, r1 = rows.start, rows.start + out.shape[0]
        c0, c1 = cols.start, cols.start + out.shape[1]
        hits = np.flatnonzero(shapely.intersects(self.pixelZones, shapely.box(c0, r0, c1, r1)))
        if hits.size == 0:
            return out
        e, n = self.geometry.nodesEnu(slice(r0, r1), slice(c0, c1))
        for z in hits:
            mask = (
                rasterizePixels(
                    [(self.pixelZones[z], 1)], out.shape, r0, c0, allTouched=False, dtype="uint8"
                )
                > 0
            )
            if mask.any():
                a, b, c = self.coeffs[z]
                out[mask] = np.minimum(out[mask], a + b * e[mask] + c * n[mask])
        return out

    def summary(self) -> dict[str, Any]:
        """Statistics for ``scene.json``."""
        return {
            "buildingsCorrected": int(self.coeffs.shape[0]),
            "maxLowering_m": float(self.maxLowering_m.max()) if self.maxLowering_m.size else 0.0,
            "method": (
                "ground under buildings capped by a robust plane fitted to ground "
                f"{RING_INNER_m:g}-{RING_OUTER_m:g} m outside the footprint, within the footprint "
                f"+{ZONE_BUFFER_m:g} m, when the interior stands > {TRIGGER_m:g} m above it"
            ),
        }


def _toEnu(frame: LocalFrame, geoms: npt.NDArray[np.object_]) -> npt.NDArray[np.object_]:
    coords = shapely.get_coordinates(geoms)
    e, n, _ = frame.geodeticToEnu(coords[:, 1], coords[:, 0], 0.0)
    return np.asarray(
        shapely.make_valid(shapely.set_coordinates(geoms.copy(), np.column_stack([e, n]))),
        dtype=object,
    )


def _enuToPixel(geometry: GridGeometry, zones: npt.NDArray[np.object_]) -> npt.NDArray[np.object_]:
    coords = shapely.get_coordinates(zones)
    a = geometry.gridAxes_ENU
    gx = coords[:, 0] * a[0, 0] + coords[:, 1] * a[0, 1]
    gy = coords[:, 0] * a[1, 0] + coords[:, 1] * a[1, 1]
    s = geometry.gridSpacing_m
    col = (gx + 0.5 * geometry.nCols * s) / s
    row = (gy + 0.5 * geometry.nRows * s) / s
    return np.asarray(
        shapely.set_coordinates(zones.copy(), np.column_stack([col, row])), dtype=object
    )


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
