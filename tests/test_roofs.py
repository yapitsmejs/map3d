"""Tests for ``map3d.layers.roofs`` (roof height fields and their meshes)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import shapely

from map3d.layers.roofs import RoofShape, makeRoof, roofShapeCodes

LON0, LAT0 = 103.86, 1.28
KX, KY = 111_319.5 * math.cos(math.radians(LAT0)), 110_574.0


def _deg(poly_m: shapely.Polygon) -> shapely.Polygon:
    """A polygon given in metres around (LON0, LAT0), in lon/lat."""
    return shapely.transform(poly_m, lambda xy: xy / (KX, KY) + (LON0, LAT0))


def _rect(w: float, h: float, x0: float = 0.0, y0: float = 0.0) -> shapely.Polygon:
    return _deg(shapely.box(x0 - w / 2, y0 - h / 2, x0 + w / 2, y0 + h / 2))


def _local(roof, x_m, y_m):
    """Rise at points given in metres around (LON0, LAT0)."""
    lon = np.asarray(x_m, dtype=float) / KX + LON0
    lat = np.asarray(y_m, dtype=float) / KY + LAT0
    return roof.riseAt(lon, lat)


def testShapeCodes_namesAliasesAndUnknown():
    codes = roofShapeCodes(["dome", "Gambrel", "half_hipped", None, "", "weird", "round"])
    assert list(codes) == [
        RoofShape.DOME,
        RoofShape.GABLED,
        RoofShape.HIPPED,
        RoofShape.FLAT,
        RoofShape.FLAT,
        RoofShape.FLAT,
        RoofShape.ROUND,
    ]


def testMakeRoof_flatOrNoHeightIsNone():
    assert makeRoof(_rect(20, 10), RoofShape.FLAT, 30.0) is None
    assert makeRoof(_rect(20, 10), RoofShape.GABLED, 0.0) is None
    assert makeRoof(shapely.LineString([(0, 0), (1, 1)]), RoofShape.GABLED, 10.0) is None


def testLocalPlane_roundTripsAndIsMetric():
    roof = makeRoof(_rect(20, 10), RoofShape.GABLED, 30.0)
    x, y = roof.toLocal([LON0 + 10 / KX], [LAT0])
    assert x[0] == pytest.approx(10.0, abs=0.01) and y[0] == pytest.approx(0.0, abs=0.01)
    lon, lat = roof.toLonLat(x, y)
    assert lon[0] == pytest.approx(LON0 + 10 / KX, abs=1e-12)
    assert roof.riseAt(lon, lat)[0] == pytest.approx(_local(roof, x, y)[0])


def testGabled_ridgeAlongLongSideWithDefaultPitch():
    roof = makeRoof(_rect(20, 10), RoofShape.GABLED, 30.0)
    assert roof.mode == "axis"
    expected = 5.0 * math.tan(math.radians(30.0))
    assert roof.rise_m == pytest.approx(expected, rel=1e-3)
    rise = _local(roof, [0, 8, 0, 0, 0], [0, 0, 5, -5, 2.5])
    np.testing.assert_allclose(rise, [expected, expected, 0, 0, expected / 2], rtol=1e-3, atol=1e-3)


def testGabled_directionAndOrientationTurnTheRidge():
    across = makeRoof(_rect(20, 10), RoofShape.GABLED, 30.0, orientation="across")
    east = makeRoof(_rect(20, 10), RoofShape.GABLED, 30.0, direction_deg=90.0)
    for roof in (across, east):
        rise = _local(roof, [0, 10, -10, 0], [0, 0, 0, 5])
        assert rise[0] == pytest.approx(roof.rise_m, abs=1e-3)
        assert rise[1] == pytest.approx(0.0, abs=1e-3) and rise[2] == pytest.approx(0.0, abs=1e-3)
        assert rise[3] == pytest.approx(roof.rise_m, abs=1e-3)


def testSkillion_highOppositeTheDirectionItFaces():
    roof = makeRoof(_rect(20, 10), RoofShape.SKILLION, 30.0, roofHeight_m=3.0, direction_deg=180.0)
    north, middle, south = _local(roof, [0, 0, 0], [5, 0, -5])
    assert (north, middle, south) == pytest.approx((3.0, 1.5, 0.0), abs=1e-3)


def testHipped_rectangleHasFourPlanes():
    roof = makeRoof(_rect(20, 10), RoofShape.HIPPED, 30.0, roofHeight_m=4.0)
    assert roof.mode == "axis"
    rise = _local(roof, [0, 5, 10, -10, 0, 7.5], [0, 0, 0, 0, 5, 0])
    np.testing.assert_allclose(rise, [4.0, 4.0, 0.0, 0.0, 0.0, 2.0], atol=1e-3)


def testHipped_squareAndWideBoxes():
    square = makeRoof(_rect(10, 10), RoofShape.HIPPED, 30.0, roofHeight_m=2.0)
    assert _local(square, [0, 2.5], [0, 0]) == pytest.approx([2.0, 1.0], abs=1e-3)
    wide = makeRoof(_rect(20, 10), RoofShape.HIPPED, 30.0, roofHeight_m=4.0, direction_deg=90.0)
    assert _local(wide, [0, 5, 0], [0, 0, 2.5]) == pytest.approx([4.0, 4.0, 2.0], abs=1e-3)
    _assertMeshCoversFootprint(wide)


def testDome_quarterCircleOverACircle():
    circle = _deg(shapely.Point(0, 0).buffer(20.0, quad_segs=32))
    roof = makeRoof(circle, RoofShape.DOME, 35.0)
    assert roof.mode == "radial" and roof.curved
    assert roof.rise_m == pytest.approx(20.0, rel=0.01)  # radius of the footprint's area
    rise = _local(roof, [0, 10, 0, 20, 30], [0, 0, -10, 0, 0])
    expected = roof.rise_m * np.array([1.0, math.sqrt(0.75), math.sqrt(0.75), 0.0, 0.0])
    np.testing.assert_allclose(rise, expected, atol=0.05)


def testDome_clampedToTheAvailableHeight():
    circle = _deg(shapely.Point(0, 0).buffer(40.0))
    roof = makeRoof(circle, RoofShape.DOME, 30.0, roofHeight_m=35.0)
    assert roof.rise_m == pytest.approx(30.0)


def testPyramidalAndOnion_profiles():
    pyramid = makeRoof(_rect(10, 10), RoofShape.PYRAMIDAL, 30.0, roofHeight_m=6.0)
    assert _local(pyramid, [0, 2.5, 0, 5], [0, 0, 2.5, 5]) == pytest.approx(
        [6.0, 3.0, 3.0, 0.0], abs=1e-3
    )
    onion = makeRoof(_rect(10, 10), RoofShape.ONION, 30.0)
    top, side = _local(onion, [0, 4], [0, 0])
    assert top == pytest.approx(onion.rise_m, abs=1e-3) and 0.0 < side < 0.5 * onion.rise_m


def testRound_rectangleIsABarrelAndIrregularIsAShell():
    barrel = makeRoof(_rect(30, 12), RoofShape.ROUND, 30.0)
    assert barrel.mode == "axis"
    assert barrel.rise_m == pytest.approx(6.0, rel=1e-3)
    end, mid = _local(barrel, [15, 15], [0, 6])
    assert end == pytest.approx(6.0, rel=1e-3) and mid == pytest.approx(
        0.0, abs=1e-3
    )  # barrel ends
    leaf = _deg(shapely.Polygon([(-40, 0), (-20, 15), (20, 15), (40, 0), (20, -15), (-20, -15)]))
    shell = makeRoof(leaf, RoofShape.ROUND, 30.0, roofHeight_m=35.0)
    assert shell.mode == "inset" and shell.rise_m == pytest.approx(30.0)
    unmapped = makeRoof(leaf, RoofShape.ROUND, 30.0)
    assert unmapped.rise_m == pytest.approx(15.0)  # default: half the height, walls remain
    centre, edge, outside = _local(shell, [0, 39.9, 60], [0, 0, 0])
    assert centre == pytest.approx(30.0, rel=0.01) and edge < 3.0 and outside == 0.0


def testDome_nonStarShapedFootprintFallsBackToInset():
    u = _deg(
        shapely.Polygon(
            [(-30, -20), (30, -20), (30, 20), (20, 20), (20, -10), (-20, -10), (-20, 20), (-30, 20)]
        )
    )
    roof = makeRoof(u, RoofShape.DOME, 20.0)
    assert roof.mode == "inset"
    _assertMeshCoversFootprint(roof)


@pytest.mark.parametrize(
    "shape",
    [
        RoofShape.SKILLION,
        RoofShape.GABLED,
        RoofShape.HIPPED,
        RoofShape.PYRAMIDAL,
        RoofShape.DOME,
        RoofShape.ONION,
        RoofShape.ROUND,
    ],
)
def testMesh_trianglesTileTheFootprintAndOutlineIsWhole(shape):
    footprint = _deg(shapely.Polygon([(-12, -8), (14, -9), (15, 9), (-11, 10)]))
    roof = makeRoof(footprint, shape, 25.0)
    _assertMeshCoversFootprint(roof)


def testMesh_courtyardHoleIsKept():
    ring = _deg(shapely.box(-20, -20, 20, 20).difference(shapely.box(-5, -5, 5, 5)))
    roof = makeRoof(ring, RoofShape.HIPPED, 20.0)
    _assertMeshCoversFootprint(roof)


def _assertMeshCoversFootprint(roof) -> None:
    tris = roof.triangles()
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    assert 0.5 * np.abs(cross).sum() == pytest.approx(shapely.area(roof.footprint), rel=1e-6)
    start, end = roof.outlineSegments()
    perimeter = shapely.length(shapely.boundary(roof.footprint))
    assert np.linalg.norm(end - start, axis=1).sum() == pytest.approx(perimeter, rel=1e-6)
