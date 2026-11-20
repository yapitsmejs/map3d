"""Coordinate frames: WGS84 geodetic, ECEF, and scene-center ENU (SPEC section 5).

All functions are vectorised over NumPy arrays (scalars are accepted) and use
float64 throughout. Angles are degrees at the API boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pyproj

__all__ = [
    "WGS84_A_m",
    "WGS84_F",
    "LocalFrame",
    "ecefToGeodetic",
    "enuRotation",
    "geodeticToEcef",
]

FloatArray = npt.NDArray[np.float64]

WGS84_A_m = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
_B_m = WGS84_A_m * (1.0 - WGS84_F)
_E2 = WGS84_F * (2.0 - WGS84_F)  # first eccentricity squared
_EP2 = _E2 / (1.0 - _E2)  # second eccentricity squared

_COLUMN_TOL_m = 1e-7
_COLUMN_MAX_ITER = 10


def geodeticToEcef(
    lat_deg: npt.ArrayLike, lon_deg: npt.ArrayLike, height_m: npt.ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Convert WGS84 geodetic coordinates to ECEF.

    Args:
        lat_deg: Geodetic latitude, in degrees.
        lon_deg: Longitude, in degrees.
        height_m: Height above the WGS84 ellipsoid, in meters.

    Returns:
        ``(x_m, y_m, z_m)`` ECEF coordinates, broadcast to a common shape.
    """
    lat_rad = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon_rad = np.radians(np.asarray(lon_deg, dtype=np.float64))
    h_m = np.asarray(height_m, dtype=np.float64)
    sinLat = np.sin(lat_rad)
    cosLat = np.cos(lat_rad)
    primeVertical_m = WGS84_A_m / np.sqrt(1.0 - _E2 * sinLat**2)
    x_m = (primeVertical_m + h_m) * cosLat * np.cos(lon_rad)
    y_m = (primeVertical_m + h_m) * cosLat * np.sin(lon_rad)
    z_m = (primeVertical_m * (1.0 - _E2) + h_m) * sinLat
    return x_m, y_m, z_m


def ecefToGeodetic(
    x_m: npt.ArrayLike, y_m: npt.ArrayLike, z_m: npt.ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Convert ECEF to WGS84 geodetic coordinates (Heikkinen closed form).

    Exact to well below a millimetre for points near the Earth's surface.
    Undefined at the Earth's centre.

    Args:
        x_m: ECEF x, in meters.
        y_m: ECEF y, in meters.
        z_m: ECEF z, in meters.

    Returns:
        ``(lat_deg, lon_deg, height_m)``, height above the WGS84 ellipsoid.
    """
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    z = np.asarray(z_m, dtype=np.float64)
    a, b, e2 = WGS84_A_m, _B_m, _E2
    p = np.hypot(x, y)
    bigF = 54.0 * b**2 * z**2
    bigG = p**2 + (1.0 - e2) * z**2 - e2 * (a**2 - b**2)
    c = e2**2 * bigF * p**2 / bigG**3
    s = np.cbrt(1.0 + c + np.sqrt(c**2 + 2.0 * c))
    k = s + 1.0 + 1.0 / s
    bigP = bigF / (3.0 * k**2 * bigG**2)
    bigQ = np.sqrt(1.0 + 2.0 * e2**2 * bigP)
    r0 = -(bigP * e2 * p) / (1.0 + bigQ) + np.sqrt(
        0.5 * a**2 * (1.0 + 1.0 / bigQ)
        - bigP * (1.0 - e2) * z**2 / (bigQ * (1.0 + bigQ))
        - 0.5 * bigP * p**2
    )
    pMinus = p - e2 * r0
    bigU = np.hypot(pMinus, z)
    bigV = np.sqrt(pMinus**2 + (1.0 - e2) * z**2)
    z0 = b**2 * z / (a * bigV)
    height_m = bigU * (1.0 - b**2 / (a * bigV))
    lat_deg = np.degrees(np.arctan2(z + _EP2 * z0, p))
    lon_deg = np.degrees(np.arctan2(y, x))
    return lat_deg, lon_deg, height_m


def enuRotation(lat0_deg: float, lon0_deg: float) -> FloatArray:
    """Rotation whose rows are the East, North, Up unit vectors in ECEF.

    ``v_ENU = R @ v_ECEF`` and ``v_ECEF = R.T @ v_ENU``.

    Args:
        lat0_deg: Geodetic latitude of the frame origin, in degrees.
        lon0_deg: Longitude of the frame origin, in degrees.

    Returns:
        3x3 orthonormal, right-handed rotation matrix.
    """
    lat_rad = np.radians(lat0_deg)
    lon_rad = np.radians(lon0_deg)
    sinLat, cosLat = np.sin(lat_rad), np.cos(lat_rad)
    sinLon, cosLon = np.sin(lon_rad), np.cos(lon_rad)
    return np.array(
        [
            [-sinLon, cosLon, 0.0],
            [-sinLat * cosLon, -sinLat * sinLon, cosLat],
            [cosLat * cosLon, cosLat * sinLon, sinLat],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class LocalFrame:
    """East-North-Up tangent frame with origin on the ellipsoid at the scene center.

    Attributes:
        lat0_deg: Geodetic latitude of the origin, in degrees.
        lon0_deg: Longitude of the origin, in degrees.
        origin_ECEF_m: ECEF position of the origin (height 0), in meters.
        rotation: Rows are the E, N, U unit vectors in ECEF.
    """

    lat0_deg: float
    lon0_deg: float
    origin_ECEF_m: FloatArray = field(init=False, repr=False, compare=False)
    rotation: FloatArray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        origin = np.array(geodeticToEcef(self.lat0_deg, self.lon0_deg, 0.0), dtype=np.float64)
        rotation = enuRotation(self.lat0_deg, self.lon0_deg)
        origin.setflags(write=False)
        rotation.setflags(write=False)
        object.__setattr__(self, "origin_ECEF_m", origin)
        object.__setattr__(self, "rotation", rotation)

    def ecefToEnu(
        self, x_m: npt.ArrayLike, y_m: npt.ArrayLike, z_m: npt.ArrayLike
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Convert ECEF coordinates to this ENU frame.

        Args:
            x_m: ECEF x, in meters.
            y_m: ECEF y, in meters.
            z_m: ECEF z, in meters.

        Returns:
            ``(e_m, n_m, u_m)``.
        """
        dx = np.asarray(x_m, dtype=np.float64) - self.origin_ECEF_m[0]
        dy = np.asarray(y_m, dtype=np.float64) - self.origin_ECEF_m[1]
        dz = np.asarray(z_m, dtype=np.float64) - self.origin_ECEF_m[2]
        r = self.rotation
        return (
            r[0, 0] * dx + r[0, 1] * dy + r[0, 2] * dz,
            r[1, 0] * dx + r[1, 1] * dy + r[1, 2] * dz,
            r[2, 0] * dx + r[2, 1] * dy + r[2, 2] * dz,
        )

    def enuToEcef(
        self, e_m: npt.ArrayLike, n_m: npt.ArrayLike, u_m: npt.ArrayLike
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Convert ENU coordinates in this frame to ECEF.

        Args:
            e_m: East, in meters.
            n_m: North, in meters.
            u_m: Up, in meters.

        Returns:
            ``(x_m, y_m, z_m)``.
        """
        e = np.asarray(e_m, dtype=np.float64)
        n = np.asarray(n_m, dtype=np.float64)
        u = np.asarray(u_m, dtype=np.float64)
        r = self.rotation
        o = self.origin_ECEF_m
        return (
            o[0] + r[0, 0] * e + r[1, 0] * n + r[2, 0] * u,
            o[1] + r[0, 1] * e + r[1, 1] * n + r[2, 1] * u,
            o[2] + r[0, 2] * e + r[1, 2] * n + r[2, 2] * u,
        )

    def geodeticToEnu(
        self, lat_deg: npt.ArrayLike, lon_deg: npt.ArrayLike, height_m: npt.ArrayLike
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Convert geodetic coordinates (ellipsoidal height) to this ENU frame.

        Args:
            lat_deg: Geodetic latitude, in degrees.
            lon_deg: Longitude, in degrees.
            height_m: Height above the WGS84 ellipsoid, in meters.

        Returns:
            ``(e_m, n_m, u_m)``.
        """
        return self.ecefToEnu(*geodeticToEcef(lat_deg, lon_deg, height_m))

    def enuToGeodetic(
        self, e_m: npt.ArrayLike, n_m: npt.ArrayLike, u_m: npt.ArrayLike
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Convert ENU coordinates in this frame to geodetic coordinates.

        Args:
            e_m: East, in meters.
            n_m: North, in meters.
            u_m: Up, in meters.

        Returns:
            ``(lat_deg, lon_deg, height_m)``, height above the WGS84 ellipsoid.
        """
        return ecefToGeodetic(*self.enuToEcef(e_m, n_m, u_m))

    def columnPointAtHeight(
        self, e_m: npt.ArrayLike, n_m: npt.ArrayLike, heightEll_m: npt.ArrayLike
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Find the point on a node column that has a given ellipsoidal height.

        A node column is the line through ``(e_m, n_m)`` parallel to this
        frame's Up axis (SPEC section 5.3). Away from the origin the local vertical
        tilts, so the geodetic position of a column point depends on its height.

        Args:
            e_m: Column East coordinate, in meters.
            n_m: Column North coordinate, in meters.
            heightEll_m: Target height above the WGS84 ellipsoid, in meters.

        Returns:
            ``(lat_deg, lon_deg, u_m)`` of that point: its geodetic position and
            its Up coordinate in this frame.

        Raises:
            RuntimeError: If the fixed-point iteration fails to converge.
        """
        e, n, h = np.broadcast_arrays(
            np.asarray(e_m, dtype=np.float64),
            np.asarray(n_m, dtype=np.float64),
            np.asarray(heightEll_m, dtype=np.float64),
        )
        # Initial guess: spherical curvature drop below the tangent plane.
        # Each element stops updating once it converges, so results are
        # bitwise independent of which other points share the batch.
        u = np.array(h - (e**2 + n**2) / (2.0 * WGS84_A_m), dtype=np.float64)
        active = np.ones(u.shape, dtype=bool)
        for _ in range(_COLUMN_MAX_ITER):
            _, _, hGot = self.enuToGeodetic(e[active], n[active], u[active])
            delta = h[active] - hGot
            u[active] = u[active] + delta
            stillActive = np.abs(delta) >= _COLUMN_TOL_m
            active[active] = stillActive
            if not active.any():
                lat_deg, lon_deg, _ = self.enuToGeodetic(e, n, u)
                return lat_deg, lon_deg, u
        raise RuntimeError("columnPointAtHeight did not converge")

    def orthoCrs(self) -> pyproj.CRS:
        """Ellipsoidal orthographic CRS centered on this frame's origin.

        Its projected coordinates equal ENU East/North for points on the
        ellipsoid, so a node column's footprint is exact in this CRS.

        Returns:
            The pyproj CRS.
        """
        return pyproj.CRS.from_proj4(
            f"+proj=ortho +lat_0={self.lat0_deg!r} +lon_0={self.lon0_deg!r} "
            "+ellps=WGS84 +units=m +no_defs"
        )
