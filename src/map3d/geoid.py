"""Geoid undulation lookup for MSL heights (SPEC section 4: EGM2008).

``H_msl = h_ellipsoid - N``, where ``N`` is the geoid undulation returned here.
The EGM2008 grid (PROJ-data ``us_nga_egm08_25.tif``) ships inside each offline
data pack; this module only interpolates it.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio

__all__ = ["GeoidGrid"]

FloatArray = npt.NDArray[np.float64]


class GeoidGrid:
    """Regular latitude/longitude grid of geoid undulations, bilinearly interpolated.

    Longitude wrap-around is not handled: queries must fall inside the grid's
    longitude range (packs are clipped to a bounding box anyway).
    """

    def __init__(
        self, lat_deg: npt.ArrayLike, lon_deg: npt.ArrayLike, undulation_m: npt.ArrayLike
    ) -> None:
        """Create a grid from node axes and values.

        Args:
            lat_deg: Node latitudes, strictly ascending or descending, shape ``(nLat,)``.
            lon_deg: Node longitudes, strictly ascending, shape ``(nLon,)``.
            undulation_m: Geoid undulation at each node, shape ``(nLat, nLon)``.

        Raises:
            ValueError: If the axes or values are malformed.
        """
        lat = np.asarray(lat_deg, dtype=np.float64).ravel()
        lon = np.asarray(lon_deg, dtype=np.float64).ravel()
        values = np.asarray(undulation_m, dtype=np.float64)
        if lat.size < 2 or lon.size < 2:
            raise ValueError("geoid grid axes need at least 2 nodes each")
        if lat[0] > lat[-1]:
            lat = lat[::-1]
            values = values[::-1, :]
        if np.any(np.diff(lat) <= 0) or np.any(np.diff(lon) <= 0):
            raise ValueError("geoid grid axes must be strictly monotonic (longitude ascending)")
        if values.shape != (lat.size, lon.size):
            raise ValueError(
                f"undulation shape {values.shape} does not match axes ({lat.size}, {lon.size})"
            )
        self._lat_deg = lat
        self._lon_deg = lon
        self._values_m = np.ascontiguousarray(values)
        self._latIndex = np.arange(lat.size, dtype=np.float64)
        self._lonIndex = np.arange(lon.size, dtype=np.float64)

    @classmethod
    def fromGeoTiff(cls, path: str | PathLike[str]) -> GeoidGrid:
        """Load a north-up geographic GeoTIFF (band 1) whose pixel centers are nodes.

        Args:
            path: Path to the GeoTIFF.

        Returns:
            The geoid grid.

        Raises:
            ValueError: If the raster is not a north-up geographic grid.
        """
        with rasterio.open(Path(path)) as src:
            t = src.transform
            if src.crs is None or not src.crs.is_geographic or t.b != 0 or t.d != 0:
                raise ValueError(f"{path} must be a north-up geographic (lat/lon) grid")
            values = src.read(1).astype(np.float64)
            lon_deg = t.c + (np.arange(src.width) + 0.5) * t.a
            lat_deg = t.f + (np.arange(src.height) + 0.5) * t.e
        return cls(lat_deg, lon_deg, values)

    def undulation_m(self, lat_deg: npt.ArrayLike, lon_deg: npt.ArrayLike) -> FloatArray:
        """Geoid undulation ``N`` at the given positions (bilinear).

        Args:
            lat_deg: Geodetic latitude, in degrees.
            lon_deg: Longitude, in degrees.

        Returns:
            Undulation in meters, broadcast to the inputs' common shape.

        Raises:
            ValueError: If any position lies outside the grid.
        """
        lat, lon = np.broadcast_arrays(
            np.asarray(lat_deg, dtype=np.float64), np.asarray(lon_deg, dtype=np.float64)
        )
        if (
            np.any(lat < self._lat_deg[0])
            or np.any(lat > self._lat_deg[-1])
            or np.any(lon < self._lon_deg[0])
            or np.any(lon > self._lon_deg[-1])
        ):
            raise ValueError("position outside the geoid grid coverage")
        fi = np.interp(lat, self._lat_deg, self._latIndex)
        fj = np.interp(lon, self._lon_deg, self._lonIndex)
        i0 = np.clip(np.floor(fi).astype(np.intp), 0, self._lat_deg.size - 2)
        j0 = np.clip(np.floor(fj).astype(np.intp), 0, self._lon_deg.size - 2)
        ti = fi - i0
        tj = fj - j0
        v = self._values_m
        result: FloatArray = (
            (1 - ti) * (1 - tj) * v[i0, j0]
            + (1 - ti) * tj * v[i0, j0 + 1]
            + ti * (1 - tj) * v[i0 + 1, j0]
            + ti * tj * v[i0 + 1, j0 + 1]
        )
        return result
