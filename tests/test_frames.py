"""Tests for ``map3d.frames`` -- checked against independent pyproj pipelines."""

from __future__ import annotations

import numpy as np
import pyproj
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from map3d import frames

LAT0_deg = 1.2834
LON0_deg = 103.8607
_rng = np.random.default_rng(12345)


def _topocentric(lat0_deg: float, lon0_deg: float) -> pyproj.Transformer:
    return pyproj.Transformer.from_pipeline(
        "+proj=pipeline +step +proj=cart +ellps=WGS84 "
        f"+step +proj=topocentric +ellps=WGS84 +lat_0={lat0_deg} +lon_0={lon0_deg} +h_0=0"
    )


def _randomLocalPoints(n: int = 2000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_deg = LAT0_deg + _rng.uniform(-0.04, 0.04, n)  # ~ +-4.5 km
    lon_deg = LON0_deg + _rng.uniform(-0.04, 0.04, n)
    height_m = _rng.uniform(-100.0, 500.0, n)
    return lat_deg, lon_deg, height_m


def testGeodeticToEcef_matchesPyproj():
    lat_deg, lon_deg, height_m = _randomLocalPoints()
    x_m, y_m, z_m = frames.geodeticToEcef(lat_deg, lon_deg, height_m)
    toEcef = pyproj.Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    xRef, yRef, zRef = toEcef.transform(lon_deg, lat_deg, height_m)
    np.testing.assert_allclose(x_m, xRef, atol=1e-6, rtol=0)
    np.testing.assert_allclose(y_m, yRef, atol=1e-6, rtol=0)
    np.testing.assert_allclose(z_m, zRef, atol=1e-6, rtol=0)


@settings(max_examples=200, deadline=None)
@given(
    lat_deg=st.floats(-89.9, 89.9),
    lon_deg=st.floats(-180.0, 180.0),
    height_m=st.floats(-500.0, 10000.0),
)
def testEcefToGeodetic_roundTrips(lat_deg: float, lon_deg: float, height_m: float):
    x_m, y_m, z_m = frames.geodeticToEcef(lat_deg, lon_deg, height_m)
    latBack_deg, lonBack_deg, heightBack_m = frames.ecefToGeodetic(x_m, y_m, z_m)
    assert latBack_deg == pytest.approx(lat_deg, abs=1e-10)
    lonDiff_deg = (lonBack_deg - lon_deg + 180.0) % 360.0 - 180.0
    assert lonDiff_deg == pytest.approx(0.0, abs=1e-10)
    assert heightBack_m == pytest.approx(height_m, abs=1e-6)


def testEnuRotation_isOrthonormalRightHanded():
    rot = frames.enuRotation(LAT0_deg, LON0_deg)
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-15)
    assert np.linalg.det(rot) == pytest.approx(1.0)


def testEnuRotation_upIsEllipsoidNormal():
    rot = frames.enuRotation(LAT0_deg, LON0_deg)
    lat_rad, lon_rad = np.radians(LAT0_deg), np.radians(LON0_deg)
    expectedUp = [
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad),
    ]
    np.testing.assert_allclose(rot[2], expectedUp, atol=1e-15)


def testLocalFrameGeodeticToEnu_matchesPyprojTopocentric():
    frame = frames.LocalFrame(LAT0_deg, LON0_deg)
    lat_deg, lon_deg, height_m = _randomLocalPoints()
    e_m, n_m, u_m = frame.geodeticToEnu(lat_deg, lon_deg, height_m)
    eRef, nRef, uRef = _topocentric(LAT0_deg, LON0_deg).transform(lon_deg, lat_deg, height_m)
    np.testing.assert_allclose(e_m, eRef, atol=1e-6, rtol=0)
    np.testing.assert_allclose(n_m, nRef, atol=1e-6, rtol=0)
    np.testing.assert_allclose(u_m, uRef, atol=1e-6, rtol=0)


def testLocalFrameEnuToGeodetic_roundTrips():
    frame = frames.LocalFrame(LAT0_deg, LON0_deg)
    lat_deg, lon_deg, height_m = _randomLocalPoints()
    latBack_deg, lonBack_deg, heightBack_m = frame.enuToGeodetic(
        *frame.geodeticToEnu(lat_deg, lon_deg, height_m)
    )
    np.testing.assert_allclose(latBack_deg, lat_deg, atol=1e-10, rtol=0)
    np.testing.assert_allclose(lonBack_deg, lon_deg, atol=1e-10, rtol=0)
    np.testing.assert_allclose(heightBack_m, height_m, atol=1e-6, rtol=0)


def testLocalFrameOrigin_isCenterOnEllipsoid():
    frame = frames.LocalFrame(LAT0_deg, LON0_deg)
    e_m, n_m, u_m = frame.geodeticToEnu(LAT0_deg, LON0_deg, 0.0)
    assert (float(e_m), float(n_m), float(u_m)) == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
    np.testing.assert_allclose(
        frame.origin_ECEF_m, frames.geodeticToEcef(LAT0_deg, LON0_deg, 0.0), atol=1e-9
    )


def testLocalFrameCurvature_groundFallsAway():
    # ~3.5 km east on the ellipsoid sits ~0.96 m below the tangent plane.
    frame = frames.LocalFrame(LAT0_deg, LON0_deg)
    lat_deg, lon_deg, _ = frame.enuToGeodetic(3500.0, 0.0, 0.0)
    _, _, u_m = frame.geodeticToEnu(lat_deg, lon_deg, 0.0)
    assert float(u_m) == pytest.approx(-(3500.0**2) / (2 * 6_378_137.0), rel=0.01)


def testColumnPointAtHeight_hitsRequestedHeightOnColumn():
    frame = frames.LocalFrame(LAT0_deg, LON0_deg)
    n = 2000
    e_m = _rng.uniform(-3600.0, 3600.0, n)
    n_m = _rng.uniform(-3600.0, 3600.0, n)
    heightEll_m = _rng.uniform(-50.0, 600.0, n)
    lat_deg, lon_deg, u_m = frame.columnPointAtHeight(e_m, n_m, heightEll_m)
    eBack, nBack, uBack = frame.geodeticToEnu(lat_deg, lon_deg, heightEll_m)
    np.testing.assert_allclose(eBack, e_m, atol=1e-6, rtol=0)
    np.testing.assert_allclose(nBack, n_m, atol=1e-6, rtol=0)
    np.testing.assert_allclose(uBack, u_m, atol=1e-6, rtol=0)


def testColumnPointAtHeight_footprintMatchesOrthoProjection():
    frame = frames.LocalFrame(LAT0_deg, LON0_deg)
    e_m = np.array([-2500.0, 0.0, 1234.5, 3535.5])
    n_m = np.array([1800.0, 0.0, -987.6, 3535.5])
    lat_deg, lon_deg, _ = frame.columnPointAtHeight(e_m, n_m, np.zeros(4))
    toOrtho = pyproj.Transformer.from_crs("EPSG:4326", frame.orthoCrs(), always_xy=True)
    x_m, y_m = toOrtho.transform(lon_deg, lat_deg)
    np.testing.assert_allclose(x_m, e_m, atol=1e-6, rtol=0)
    np.testing.assert_allclose(y_m, n_m, atol=1e-6, rtol=0)


def testColumnPointAtHeight_broadcastsScalarHeight():
    frame = frames.LocalFrame(LAT0_deg, LON0_deg)
    _, _, u_m = frame.columnPointAtHeight(np.zeros((2, 3)), np.zeros((2, 3)), 10.0)
    assert u_m.shape == (2, 3)
    np.testing.assert_allclose(u_m, 10.0, atol=1e-6)


def testColumnPointAtHeight_raisesWhenNotConverging(monkeypatch):
    frame = frames.LocalFrame(LAT0_deg, LON0_deg)
    monkeypatch.setattr(frames, "_COLUMN_MAX_ITER", 0)
    with pytest.raises(RuntimeError, match="did not converge"):
        frame.columnPointAtHeight(np.array([3000.0]), np.array([3000.0]), np.array([100.0]))


def testOrthoCrs_isEllipsoidalOrthographicAtCenter():
    crs = frames.LocalFrame(LAT0_deg, LON0_deg).orthoCrs()
    op = crs.coordinate_operation
    assert op is not None
    assert "Orthographic" in op.method_name
    params = {p.name: p.value for p in op.params}
    assert params["Latitude of natural origin"] == pytest.approx(LAT0_deg)
    assert params["Longitude of natural origin"] == pytest.approx(LON0_deg)
    assert crs.ellipsoid is not None
    assert crs.ellipsoid.name == "WGS 84"
