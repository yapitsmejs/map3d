"""Lofted structures: stacks of concentric round parts that widen upward (SPEC section 8).

OSM maps funnel-shaped structures, such as the Supertrees at Gardens by the
Bay, as stacked circular building parts: a trunk, then rings that widen with
height. Extruded one by one they render as a stepped pile of discs. Such a
stack is recognised here and drawn as one smooth loft through the parts'
outlines (:func:`map3d.mesh.scenery.buildingParts`).

A stack is recognised when it has at least two parts that are each round
(isoperimetric quotient at least :data:`ROUNDNESS`), share a centre (within
:data:`CENTRE_TOLERANCE_m`, or a fifth of the smallest radius if larger), are
ordered bottom to top with vertical gaps of at most :data:`MAX_GAP_m`, never
narrow upward, and end at least :data:`MIN_FLARE` wider than they start.
Stacks that narrow upward (set-back towers) keep their steps.

Mapping is rarely that tidy, so within a cluster of concentric round parts
the profile is a chain by top height: parts thinner than
:data:`MIN_THICKNESS_m` (planters, kerbs) stay separate buildings, and of
parts sharing a top the widest shapes the profile. The other parts of the
cluster are absorbed into the loft (not drawn on their own).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import shapely

__all__ = [
    "CENTRE_TOLERANCE_m",
    "MAX_GAP_m",
    "MIN_FLARE",
    "MIN_THICKNESS_m",
    "ROUNDNESS",
    "Loft",
    "findLofts",
]

FloatArray = npt.NDArray[np.float64]

ROUNDNESS = 0.85  # 4 pi area / perimeter^2 of a round part (a square is 0.79)
CENTRE_TOLERANCE_m = 2.0
MAX_GAP_m = 6.0
MIN_FLARE = 1.2  # top radius / bottom radius
MIN_THICKNESS_m = 1.0  # thinner round parts (planters) are not part of a loft
_SAME_TOP_m = 0.25
_RADIUS_SLACK = 0.97  # a part may be this much narrower than the one below (mapping noise)
_A_m = 6378137.0


@dataclass(frozen=True)
class Loft:
    """One lofted stack.

    Attributes:
        members: Building indices whose outlines shape the profile, bottom to top.
        centre_deg: ``(lon, lat)`` of the shared centre.
        absorbed: Other parts of the stack, drawn as part of the loft.
    """

    members: tuple[int, ...]
    centre_deg: tuple[float, float]
    absorbed: tuple[int, ...] = ()


def findLofts(
    geometries: npt.NDArray[np.object_],
    bottom_m: FloatArray,
    top_m: FloatArray,
) -> list[Loft]:
    """Find stacks of concentric round parts that widen upward.

    Args:
        geometries: Footprints in lon/lat.
        bottom_m: Underside of each part (orthometric).
        top_m: Top of each part (orthometric).

    Returns:
        The lofts (each building in at most one).
    """
    n = geometries.size
    if n == 0:
        return []
    centroids = shapely.centroid(geometries)
    lon, lat = shapely.get_x(centroids), shapely.get_y(centroids)
    kx = math.radians(1.0) * _A_m * np.cos(np.radians(lat))
    ky = math.radians(1.0) * _A_m
    area = shapely.area(geometries) * kx * ky
    perimeter = _perimeter_m(geometries, kx, ky)
    round_ = (perimeter > 0) & (4.0 * math.pi * area >= ROUNDNESS * perimeter**2)
    radius = np.sqrt(np.maximum(area, 0.0) / math.pi)
    candidates = np.flatnonzero(round_ & np.isfinite(bottom_m) & np.isfinite(top_m))
    if candidates.size < 2:
        return []
    # Cluster round parts by centre (greedy, smallest index first).
    points = shapely.points(lon[candidates], lat[candidates])
    tree = shapely.STRtree(points)
    tolerance_deg = CENTRE_TOLERANCE_m / ky
    assigned = np.zeros(n, dtype=bool)
    lofts: list[Loft] = []
    for k, i in enumerate(candidates):
        if assigned[i]:
            continue
        reach = max(tolerance_deg, 0.2 * radius[i] / ky)
        near = candidates[tree.query(points[k], predicate="dwithin", distance=reach)]
        near = near[~assigned[near]]
        dist = np.hypot((lon[near] - lon[i]) * kx[i], (lat[near] - lat[i]) * ky)
        near = near[
            dist <= np.maximum(CENTRE_TOLERANCE_m, 0.2 * np.minimum(radius[near], radius[i]))
        ]
        near = near[(top_m[near] - bottom_m[near]) >= MIN_THICKNESS_m]
        if near.size < 2:
            continue
        stack = _profileChain(near, top_m, radius)
        if stack.size >= 2 and _isFlaringStack(stack, bottom_m, top_m, radius):
            assigned[near] = True
            lofts.append(
                Loft(
                    tuple(int(s) for s in stack),
                    (float(np.mean(lon[stack])), float(np.mean(lat[stack]))),
                    tuple(int(s) for s in near if s not in set(stack.tolist())),
                )
            )
    return lofts


def _profileChain(
    near: npt.NDArray[np.int64], top: FloatArray, radius: FloatArray
) -> npt.NDArray[np.int64]:
    """Parts ordered by top; of parts sharing a top, the widest."""
    order = near[np.lexsort((-radius[near], top[near]))]
    keep = np.ones(order.size, dtype=bool)
    keep[1:] = np.diff(top[order]) > _SAME_TOP_m
    return order[keep]


def _isFlaringStack(
    stack: npt.NDArray[np.int64], bottom: FloatArray, top: FloatArray, radius: FloatArray
) -> bool:
    gaps = bottom[stack[1:]] - top[stack[:-1]]
    if np.any(gaps > MAX_GAP_m) or np.any(top[stack[1:]] <= top[stack[:-1]]):
        return False
    r = radius[stack]
    if np.any(r[1:] < _RADIUS_SLACK * r[:-1]):
        return False
    return bool(r[-1] >= MIN_FLARE * r[0])


def _perimeter_m(geometries: npt.NDArray[np.object_], kx: FloatArray, ky: float) -> FloatArray:
    rings = shapely.get_exterior_ring(geometries)
    coords, owner = shapely.get_coordinates(rings, return_index=True)
    same = owner[1:] == owner[:-1]
    d = np.diff(coords, axis=0)
    seg = np.hypot(d[:, 0] * kx[owner[1:]], d[:, 1] * ky) * same
    perimeter = np.zeros(geometries.size)
    np.add.at(perimeter, owner[1:], seg)
    return perimeter
