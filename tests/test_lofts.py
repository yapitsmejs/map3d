"""Tests for ``map3d.layers.lofts`` (stacks of concentric round parts that widen upward)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import shapely
import shapely.affinity

from map3d.layers.lofts import findLofts
from map3d.layers.wheels import findWheels

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
    gapped = [(_disc(3.0), 0.0, 15.0), (_disc(6.0), 30.0, 33.0)]
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


def testFindLofts_aWideBaseBelowTheTrunkStaysOut():
    parts = [
        (_disc(8.0), 0.0, 13.0),  # a wide planter base
        (_disc(4.0), 0.0, 30.0),  # trunk
        (_disc(6.0), 30.0, 33.0),
        (_disc(8.5), 41.0, 45.0),  # 8 m above the last ring
    ]
    lofts = _find(parts)
    assert len(lofts) == 1
    assert lofts[0].members == (1, 2, 3)
    assert 0 not in lofts[0].absorbed


def _span(s_m: float, width_m: float = 5.0, axis_deg: float = 30.0) -> shapely.Polygon:
    """A small span footprint at ``s_m`` along a line through (LON0, LAT0)."""
    a = math.radians(axis_deg)
    centre = np.array([math.cos(a), math.sin(a)]) * s_m
    box = shapely.box(-4.0, -width_m / 2, 4.0, width_m / 2)
    box = shapely.affinity.rotate(box, axis_deg, origin=(0, 0))
    return shapely.transform(box, lambda xy: (xy + centre) / (KX, KY) + (LON0, LAT0))


def testFindWheels_spansOnACircleAreOneWheel():
    import shapely.affinity  # noqa: F401

    radius, hub = 75.0, 90.0
    s = np.linspace(-70.0, 70.0, 15)
    half = np.sqrt(radius**2 - s**2)
    geoms = np.array([_span(x) for x in s] + [_disc(5.0, dx_m=300.0)], dtype=object)
    bottom = np.concatenate([hub - half, [0.0]])
    top = np.concatenate([hub + half, [30.0]])
    wheels = findWheels(geoms, bottom, top)
    assert len(wheels) == 1
    w = wheels[0]
    assert w.members == tuple(range(15))
    assert w.radius_m == pytest.approx(radius, rel=0.01)
    assert w.hubHeight_m == pytest.approx(hub, abs=0.5)
    assert w.width_m == pytest.approx(5.0, abs=0.2)
    assert (
        abs(w.axis[0] * math.sin(math.radians(30)) - w.axis[1] * math.cos(math.radians(30))) < 0.01
    )
    # A row of equal towers is no wheel; nor are too few spans.
    assert findWheels(geoms[:15], np.zeros(15), np.full(15, 100.0)) == []
    assert findWheels(geoms[:4], bottom[:4], top[:4]) == []
    assert findWheels(np.array([], dtype=object), np.zeros(0), np.zeros(0)) == []
