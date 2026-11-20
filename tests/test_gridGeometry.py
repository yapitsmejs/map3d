"""Tests for ``map3d.gridGeometry`` (SPEC section 5)."""

from __future__ import annotations

import math

import numpy as np
import pyproj
import pytest

from map3d.gridGeometry import GridGeometry
from map3d.request import SceneRequest

IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _heading(heading_deg: float) -> list[list[float]]:
    h_rad = math.radians(heading_deg)
    return [
        [math.cos(h_rad), -math.sin(h_rad), 0.0],
        [math.sin(h_rad), math.cos(h_rad), 0.0],
        [0.0, 0.0, 1.0],
    ]


def _geom(
    axes: object = IDENTITY, lx: float = 2.0, ly: float = 1.5, s: float = 0.5
) -> GridGeometry:
    req = SceneRequest(
        sceneCenter_deg=(1.2834, 103.8607),
        gridSpacing_m=s,
        gridAxes_ENU=axes,  # type: ignore[arg-type]
        swathHorizontal_m=lx,
        swathVertical_m=ly,
    )
    return GridGeometry.fromRequest(req)


def testGridAxes_areCellCenteredAndSymmetric():
    geom = _geom()
    np.testing.assert_allclose(geom.gridAxisX_m(), [-0.75, -0.25, 0.25, 0.75])
    np.testing.assert_allclose(geom.gridAxisY_m(), [-0.5, 0.0, 0.5])
    assert (geom.nRows, geom.nCols) == (3, 4)
    assert geom.shape == (3, 4)


def testNodesEnu_identityAxesRowZeroAtMinusY():
    geom = _geom()
    e_m, n_m = geom.nodesEnu()
    assert e_m.shape == (3, 4)
    np.testing.assert_allclose(e_m[0], [-0.75, -0.25, 0.25, 0.75])
    np.testing.assert_allclose(n_m[:, 0], [-0.5, 0.0, 0.5])
    assert n_m[0, 0] < n_m[-1, 0]


def testNodesEnu_heading90PointsGridYEast():
    geom = _geom(axes=_heading(90.0))
    e_m, n_m = geom.nodesEnu()
    gx_m, gy_m = np.meshgrid(geom.gridAxisX_m(), geom.gridAxisY_m())
    np.testing.assert_allclose(e_m, gy_m, atol=1e-12)
    np.testing.assert_allclose(n_m, -gx_m, atol=1e-12)


def testNodesEnu_chunkMatchesFullSlice():
    geom = _geom(axes=_heading(33.0), lx=10.0, ly=7.0)
    eFull, nFull = geom.nodesEnu()
    rows, cols = slice(3, 9), slice(5, 17)
    eChunk, nChunk = geom.nodesEnu(rows, cols)
    np.testing.assert_array_equal(eChunk, eFull[rows, cols])
    np.testing.assert_array_equal(nChunk, nFull[rows, cols])


def testAffine_pixelCentersAreNodes():
    geom = _geom(axes=_heading(-47.0), lx=10.0, ly=7.0)
    e_m, n_m = geom.nodesEnu()
    rowIdx, colIdx = np.mgrid[0 : geom.nRows, 0 : geom.nCols]
    eAff, nAff = geom.affine() @ (colIdx + 0.5, rowIdx + 0.5)
    np.testing.assert_allclose(eAff, e_m, atol=1e-9)
    np.testing.assert_allclose(nAff, n_m, atol=1e-9)


def testAffine_originIsMinusXMinusYCorner():
    geom = _geom(axes=_heading(30.0))
    e_m, n_m = geom.affine() @ (0.0, 0.0)
    gx_m, gy_m = -1.0, -0.75
    axes = geom.gridAxes_ENU
    assert e_m == pytest.approx(gx_m * axes[0, 0] + gy_m * axes[1, 0])
    assert n_m == pytest.approx(gx_m * axes[0, 1] + gy_m * axes[1, 1])


def testOrthoCrsPlusAffine_reproducesNodeEnuExactly():
    geom = _geom(axes=_heading(30.0), lx=5000.0, ly=5000.0, s=250.0)
    e_m, n_m = geom.nodesEnu()
    rowIdx, colIdx = np.mgrid[0 : geom.nRows, 0 : geom.nCols]
    x_m, y_m = geom.affine() @ (colIdx + 0.5, rowIdx + 0.5)
    toGeo = pyproj.Transformer.from_crs(geom.orthoCrs(), "EPSG:4326", always_xy=True)
    lon_deg, lat_deg = toGeo.transform(x_m, y_m)
    eBack, nBack, _ = geom.frame.geodeticToEnu(lat_deg, lon_deg, 0.0)
    np.testing.assert_allclose(eBack, e_m, atol=1e-6, rtol=0)
    np.testing.assert_allclose(nBack, n_m, atol=1e-6, rtol=0)


def testIterChunks_tilesGridExactlyOnce():
    geom = _geom(lx=10.0, ly=7.0)  # 14 x 20
    coverage = np.zeros(geom.shape, dtype=int)
    chunks = list(geom.iterChunks(4, 6))
    for rows, cols in chunks:
        coverage[rows, cols] += 1
    assert np.all(coverage == 1)
    assert len(chunks) == math.ceil(14 / 4) * math.ceil(20 / 6)
    assert chunks[0] == (slice(0, 4), slice(0, 6))
    assert chunks[-1] == (slice(12, 14), slice(18, 20))


@pytest.mark.parametrize(("chunkRows", "chunkCols"), [(0, 4), (4, 0), (-1, 2)])
def testIterChunks_rejectsNonPositive(chunkRows: int, chunkCols: int):
    with pytest.raises(ValueError, match="positive"):
        list(_geom().iterChunks(chunkRows, chunkCols))


def testFromRequest_carriesFrameAtCenter():
    geom = _geom()
    assert geom.frame.lat0_deg == 1.2834
    assert geom.frame.lon0_deg == 103.8607
    assert geom.gridSpacing_m == 0.5


def testFootprintBounds_containAllNodeFootprints():
    geom = _geom(axes=_heading(37.0), lx=4000.0, ly=3000.0, s=100.0)
    minLon, minLat, maxLon, maxLat = geom.footprintBounds_deg()
    e_m, n_m = geom.nodesEnu()
    lat_deg, lon_deg, _ = geom.frame.columnPointAtHeight(e_m, n_m, 0.0)
    assert minLon < lon_deg.min() and lon_deg.max() < maxLon
    assert minLat < lat_deg.min() and lat_deg.max() < maxLat


def testFootprintBounds_matchRotatedCornersViaOrtho():
    geom = _geom(axes=_heading(45.0), lx=2000.0, ly=2000.0, s=100.0)
    bounds = geom.footprintBounds_deg()
    # 45 deg rotation: extreme E/N extents are at +-sqrt(2) * 1000 m.
    reach_m = 1000.0 * math.sqrt(2.0)
    toGeo = pyproj.Transformer.from_crs(geom.orthoCrs(), "EPSG:4326", always_xy=True)
    lonW, _ = toGeo.transform(-reach_m, 0.0)
    lonE, _ = toGeo.transform(reach_m, 0.0)
    _, latS = toGeo.transform(0.0, -reach_m)
    _, latN = toGeo.transform(0.0, reach_m)
    assert bounds == pytest.approx((lonW, latS, lonE, latN), abs=1e-9)


def testFootprintBounds_marginExpands():
    geom = _geom(lx=1000.0, ly=1000.0, s=100.0)
    inner = geom.footprintBounds_deg()
    outer = geom.footprintBounds_deg(margin_m=500.0)
    assert outer[0] < inner[0] and outer[1] < inner[1]
    assert outer[2] > inner[2] and outer[3] > inner[3]
    # 500 m of latitude is ~0.0045 deg.
    assert inner[1] - outer[1] == pytest.approx(500.0 / 110_574.0, rel=0.01)


def testFootprintBounds_rejectsNegativeMargin():
    with pytest.raises(ValueError, match="margin"):
        _geom().footprintBounds_deg(margin_m=-1.0)
