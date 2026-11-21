"""Tests for ``map3d.layers.lofts`` (stacks of concentric round parts that widen upward)."""

from __future__ import annotations

import math

import numpy as np
import shapely

from map3d.layers.lofts import findLofts

LON0, LAT0 = 103.864, 1.282
KX, KY = 111_319.5 * math.cos(math.radians(LAT0)), 110_574.0


def _disc(r_m: float, dx_m: float = 0.0, square: bool = False) -> shapely.Polygon:
    shape = shapely.box(-r_m, -r_m, r_m, r_m) if square else shapely.Point(0, 0).buffer(r_m)
    return shapely.transform(shape, lambda xy: (xy + (dx_m, 0.0)) / (KX, KY) + (LON0, LAT0))


def _find(parts: list[tuple[shapely.Polygon, float, float]]):
    geoms = np.array([p[0] for p in parts], dtype=object)
    return findLofts(geoms, np.array([p[1] for p in parts]), np.array([p[2] for p in parts]))


def testFindLofts_supertreeLikeStackIsOneLoft():
    # Trunk 0-30 m, rings 30-33, 36-41 (a 3 m gap), widening: a Supertree.
    parts = [(_disc(6.5), 36.0, 41.0), (_disc(3.0), 0.0, 30.0), (_disc(5.0, dx_m=0.4), 30.0, 33.0)]
    lofts = _find(parts)
    assert len(lofts) == 1
    assert lofts[0].members == (1, 2, 0)  # bottom to top
    lon, lat = lofts[0].centre_deg
    assert abs(lon - LON0) * KX < 0.5 and abs(lat - LAT0) * KY < 0.5


def testFindLofts_narrowingSquareGappedOrSeparateStacksAreNot():
    setBack = [(_disc(8.0), 0.0, 30.0), (_disc(5.0), 30.0, 60.0)]  # narrows upward
    squares = [(_disc(3.0, square=True), 0.0, 30.0), (_disc(6.0, square=True), 30.0, 33.0)]
    gapped = [(_disc(3.0), 0.0, 20.0), (_disc(6.0), 30.0, 33.0)]
    apart = [(_disc(3.0), 0.0, 30.0), (_disc(6.0, dx_m=10.0), 30.0, 33.0)]
    noFlare = [(_disc(5.0), 0.0, 30.0), (_disc(5.2), 30.0, 33.0)]
    lone = [(_disc(5.0), 0.0, 30.0)]
    for parts in (setBack, squares, gapped, apart, noFlare, lone):
        assert _find(parts) == []
    assert findLofts(np.array([], dtype=object), np.zeros(0), np.zeros(0)) == []


def testFindLofts_planterAndDuplicatePartsDoNotBreakTheStack():
    parts = [
        (_disc(5.4), 0.0, 0.5),  # a planter round the trunk: stays a building of its own
        (_disc(2.8), 0.0, 30.0),  # trunk
        (_disc(4.4), 17.0, 33.0),  # a duplicate outline of the canopy's lower ring
        (_disc(3.6), 30.0, 33.0),  # the ring itself, narrower: absorbed
        (_disc(4.7), 36.0, 41.0),
    ]
    lofts = _find(parts)
    assert len(lofts) == 1
    assert lofts[0].members == (1, 2, 4)
    assert lofts[0].absorbed == (3,)
