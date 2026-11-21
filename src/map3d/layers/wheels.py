"""Observation wheels mapped as rows of vertical spans (SPEC section 8).

OSM maps an observation wheel such as the Singapore Flyer, which Simple 3D
Buildings cannot describe, as a row of small building parts along the wheel's
plane, each spanning the wheel's full height at its position (e.g. 15-165 m at
the hub, 63-117 m near the rim). Extruded, they render as a solid windowed
disc. Such a row is recognised here and drawn as a wheel
(:func:`map3d.mesh.scenery.buildingParts`).

A wheel is a group of at least :data:`MIN_SPANS` small parts (under
:data:`MAX_SPAN_AREA_m2`, spanning at least :data:`MIN_SPAN_m`), each within :data:`LINK_DISTANCE_m` of the next,
whose centres lie on a straight line (RMS offset under 3 m) and whose bottoms
and tops, plotted against position along that line, fit one circle (RMS
misfit under 6 % of its radius) of radius at least :data:`MIN_RADIUS_m`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import shapely

__all__ = [
    "LINK_DISTANCE_m",
    "MAX_SPAN_AREA_m2",
    "MIN_RADIUS_m",
    "MIN_SPANS",
    "MIN_SPAN_m",
    "Wheel",
    "findWheels",
]

FloatArray = npt.NDArray[np.float64]

MIN_SPANS = 6
MIN_SPAN_m = 20.0  # spans shorter than this (sheds, kiosks) are not part of a wheel
MAX_SPAN_AREA_m2 = 150.0
LINK_DISTANCE_m = 25.0
MIN_RADIUS_m = 10.0
_MAX_LINE_RMS_m = 3.0
_MAX_CIRCLE_MISFIT = 0.06
_A_m = 6378137.0


@dataclass(frozen=True)
class Wheel:
    """One observation wheel.

    Attributes:
        members: Building indices of the spans it replaces.
        hub_deg: ``(lon, lat)`` under the hub.
        axis: Unit ``(east, north)`` vector along the wheel's plane.
        hubHeight_m: Hub height, in the same vertical reference as the spans.
        radius_m: Rim radius.
        width_m: Rim width across the plane (the spans' median width).
    """

    members: tuple[int, ...]
    hub_deg: tuple[float, float]
    axis: tuple[float, float]
    hubHeight_m: float
    radius_m: float
    width_m: float


def findWheels(
    geometries: npt.NDArray[np.object_], bottom_m: FloatArray, top_m: FloatArray
) -> list[Wheel]:
    """Find observation wheels mapped as rows of vertical spans.

    Args:
        geometries: Footprints in lon/lat.
        bottom_m: Underside of each part.
        top_m: Top of each part (same vertical reference).

    Returns:
        The wheels (each building in at most one).
    """
    if geometries.size == 0:
        return []
    lat0 = float(np.nanmean(shapely.get_y(shapely.centroid(geometries))))
    kx = math.radians(1.0) * _A_m * math.cos(math.radians(lat0))
    ky = math.radians(1.0) * _A_m
    area = shapely.area(geometries) * kx * ky
    small = np.flatnonzero(
        (area < MAX_SPAN_AREA_m2)
        & np.isfinite(bottom_m)
        & np.isfinite(top_m)
        & (top_m - bottom_m >= MIN_SPAN_m)
    )
    if small.size < MIN_SPANS:
        return []
    c = shapely.centroid(geometries[small])
    xy = np.column_stack([shapely.get_x(c) * kx, shapely.get_y(c) * ky])
    groups = _linkedGroups(xy, LINK_DISTANCE_m)
    wheels: list[Wheel] = []
    for g in groups:
        if g.size < MIN_SPANS:
            continue
        wheel = _fitWheel(small[g], xy[g], geometries, bottom_m, top_m, kx, ky)
        if wheel is not None:
            wheels.append(wheel)
    return wheels


def _linkedGroups(xy: FloatArray, distance: float) -> list[npt.NDArray[np.int64]]:
    """Connected groups of points closer than ``distance`` (single linkage)."""
    tree = shapely.STRtree(shapely.points(xy))
    a, b = tree.query(shapely.points(xy), predicate="dwithin", distance=distance)
    parent = np.arange(xy.shape[0])

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    for i, j in zip(a.tolist(), b.tolist(), strict=True):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(xy.shape[0])])
    return [np.flatnonzero(roots == r) for r in np.unique(roots)]


def _fitWheel(
    members: npt.NDArray[np.int64],
    xy: FloatArray,
    geometries: npt.NDArray[np.object_],
    bottom: FloatArray,
    top: FloatArray,
    kx: float,
    ky: float,
) -> Wheel | None:
    centre = xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy - centre)
    axis = vt[0]
    s = (xy - centre) @ axis
    offset = (xy - centre) @ vt[1]
    if float(np.sqrt(np.mean(offset**2))) > _MAX_LINE_RMS_m:
        return None
    # Circle through the span ends in the (s, z) plane: x^2 + z^2 + D x + E z + F = 0.
    px = np.concatenate([s, s])
    pz = np.concatenate([bottom[members], top[members]])
    design = np.column_stack([px, pz, np.ones_like(px)])
    (d, e, f), *_ = np.linalg.lstsq(design, -(px**2 + pz**2), rcond=None)
    cs, cz = -d / 2.0, -e / 2.0
    r2 = cs**2 + cz**2 - f
    if r2 <= MIN_RADIUS_m**2:
        return None
    radius = float(np.sqrt(r2))
    misfit = np.hypot(px - cs, pz - cz) - radius
    if float(np.sqrt(np.mean(misfit**2))) > _MAX_CIRCLE_MISFIT * radius:
        return None
    if np.ptp(s) < radius:  # the spans must cover most of the wheel's width
        return None
    hub = centre + cs * axis
    widths = [_shortSide(geometries[i], kx, ky) for i in members]
    return Wheel(
        members=tuple(int(i) for i in members),
        hub_deg=(float(hub[0] / kx), float(hub[1] / ky)),
        axis=(float(axis[0]), float(axis[1])),
        hubHeight_m=float(cz),
        radius_m=radius,
        width_m=float(np.median(widths)),
    )


def _shortSide(geom: object, kx: float, ky: float) -> float:
    rect = shapely.get_coordinates(shapely.oriented_envelope(geom)) * (kx, ky)
    sides = np.hypot(*np.diff(rect[:3], axis=0).T)
    return float(sides.min()) if sides.size else 0.0
