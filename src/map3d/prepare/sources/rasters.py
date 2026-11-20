"""Raster source fetchers: FABDEM, Copernicus GLO-30, EGM2008, ESA WorldCover, Meta canopy.

Each fetcher clips the tiles covering a lon/lat box with windowed range reads
and returns pack-ready GeoTIFFs in the source's native CRS. Tiles that don't
exist (open ocean) are skipped.

Access paths, verified 2026-11-20 (see ``docs/research.md``):

- **FABDEM V1-2:** 10x10 degree zips of 1x1 degree GeoTIFFs at the University
  of Bristol data repository. Only the needed members are read, through the
  zip's central directory.
- **GLO-30:** public COGs on AWS (``copernicus-dem-30m``).
- **EGM2008:** the PROJ CDN COG ``us_nga_egm08_25.tif``.
- **ESA WorldCover 2021 v200:** 3x3 degree COGs on AWS.
- **Meta/WRI canopy height:** zoom-9 quadkey GeoTIFFs on AWS (Web Mercator,
  uint8 metres, row strips).
"""

from __future__ import annotations

import math
import zipfile
from pathlib import Path

import rasterio.errors

from map3d.prepare.http import HttpRangeFile, HttpSession
from map3d.prepare.rasterClip import clipRaster

__all__ = [
    "EGM2008_URL",
    "cellCorners",
    "cornerName",
    "fabdemMember",
    "fabdemZipUrl",
    "fetchEgm2008",
    "fetchFabdem",
    "fetchGlo30",
    "fetchMetaCanopy",
    "fetchWorldCover",
    "glo30Url",
    "metaCanopyUrl",
    "quadkey",
    "quadkeyBounds",
    "quadkeysCovering",
    "worldCoverUrl",
]

Bounds = tuple[float, float, float, float]

GLO30_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
FABDEM_BASE = "https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn"
EGM2008_URL = "https://cdn.proj.org/us_nga_egm08_25.tif"
WORLDCOVER_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
META_CANOPY_BASE = (
    "https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/alsgedi_global_v6_float/chm"
)
META_ZOOM = 9


def cellCorners(bounds: Bounds, step: int) -> list[tuple[int, int]]:
    """Lower-left corners ``(lat, lon)`` of the ``step``-degree cells covering a box.

    Args:
        bounds: ``(minLon, minLat, maxLon, maxLat)``.
        step: Cell size in degrees.

    Returns:
        Corners in row-major order (south to north, west to east).
    """
    minLon, minLat, maxLon, maxLat = bounds
    lats = range(math.floor(minLat / step) * step, math.ceil(maxLat / step) * step, step)
    lons = range(math.floor(minLon / step) * step, math.ceil(maxLon / step) * step, step)
    return [(lat, lon) for lat in lats for lon in lons]


def cornerName(lat: int, lon: int) -> str:
    """Tile name like ``N01E103`` / ``S10W071`` for a lower-left corner."""
    return f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}"


def glo30Url(lat: int, lon: int) -> str:
    """URL of the GLO-30 COG whose lower-left corner is ``(lat, lon)``."""
    name = (
        f"Copernicus_DSM_COG_10_{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00_"
        f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00_DEM"
    )
    return f"{GLO30_BASE}/{name}/{name}.tif"


def fabdemZipUrl(lat: int, lon: int) -> str:
    """URL of the 10-degree FABDEM zip holding the 1-degree tile at ``(lat, lon)``."""
    lat10 = math.floor(lat / 10) * 10
    lon10 = math.floor(lon / 10) * 10
    return f"{FABDEM_BASE}/{cornerName(lat10, lon10)}-{cornerName(lat10 + 10, lon10 + 10)}_FABDEM_V1-2.zip"


def fabdemMember(lat: int, lon: int) -> str:
    """Zip member name of the 1-degree FABDEM tile at ``(lat, lon)``."""
    return f"{cornerName(lat, lon)}_FABDEM_V1-2.tif"


def worldCoverUrl(lat: int, lon: int) -> str:
    """URL of the 3-degree WorldCover tile whose lower-left corner is ``(lat, lon)``."""
    return f"{WORLDCOVER_BASE}/ESA_WorldCover_10m_2021_v200_{cornerName(lat, lon)}_Map.tif"


def metaCanopyUrl(key: str) -> str:
    """URL of the Meta/WRI canopy-height tile for a quadkey."""
    return f"{META_CANOPY_BASE}/{key}.tif"


def quadkey(lat_deg: float, lon_deg: float, zoom: int) -> str:
    """Bing/Web-Mercator quadkey of the tile containing a point.

    Args:
        lat_deg: Latitude.
        lon_deg: Longitude.
        zoom: Zoom level.

    Returns:
        Quadkey string of length ``zoom``.
    """
    n = 2**zoom
    x = min(n - 1, int((lon_deg + 180.0) / 360.0 * n))
    lat_rad = math.radians(lat_deg)
    y = min(n - 1, int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n))
    return _key(x, y, zoom)


def quadkeyBounds(key: str) -> Bounds:
    """Lon/lat bounds of a quadkey tile."""
    x, y = _xy(key)
    n = 2 ** len(key)

    def lat(row: float) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * row / n))))

    return (x / n * 360.0 - 180.0, lat(y + 1), (x + 1) / n * 360.0 - 180.0, lat(y))


def quadkeysCovering(bounds: Bounds, zoom: int) -> list[str]:
    """Quadkeys of all zoom-level tiles intersecting a lon/lat box."""
    minLon, minLat, maxLon, maxLat = bounds
    nw, se = quadkey(maxLat, minLon, zoom), quadkey(minLat, maxLon, zoom)
    x0, y0 = _xy(nw)
    x1, y1 = _xy(se)
    keys = []
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            keys.append(_key(x, y, zoom))
    return keys


def fetchGlo30(session: HttpSession, bounds: Bounds, workDir: Path) -> list[Path]:
    """Clip the GLO-30 tiles covering ``bounds``."""
    return _fetchTiles(
        session,
        bounds,
        workDir,
        "copernicusGlo30",
        [(cornerName(a, o), glo30Url(a, o)) for a, o in cellCorners(bounds, 1)],
    )


def fetchWorldCover(session: HttpSession, bounds: Bounds, workDir: Path) -> list[Path]:
    """Clip the ESA WorldCover tiles covering ``bounds`` (nodata 0)."""
    tiles = [(cornerName(a, o), worldCoverUrl(a, o)) for a, o in cellCorners(bounds, 3)]
    return _fetchTiles(session, bounds, workDir, "esaWorldCover", tiles, nodata=0.0)


def fetchMetaCanopy(session: HttpSession, bounds: Bounds, workDir: Path) -> list[Path]:
    """Clip the Meta/WRI canopy-height tiles covering ``bounds``."""
    tiles = [(k, metaCanopyUrl(k)) for k in quadkeysCovering(bounds, META_ZOOM)]
    return _fetchTiles(session, bounds, workDir, "metaCanopy", tiles)


def fetchEgm2008(session: HttpSession, bounds: Bounds, workDir: Path) -> list[Path]:
    """Clip the global EGM2008 grid with a few cells of margin."""
    out = clipRaster(EGM2008_URL, bounds, workDir / "egm2008.tif", session=session, marginPixels=3)
    return [] if out is None else [out]


def fetchFabdem(session: HttpSession, bounds: Bounds, workDir: Path) -> list[Path]:
    """Extract and clip the FABDEM 1-degree tiles covering ``bounds``.

    Only the needed members of each 10-degree zip are transferred.
    """
    out: list[Path] = []
    archives: dict[str, zipfile.ZipFile | None] = {}
    for lat, lon in cellCorners(bounds, 1):
        url = fabdemZipUrl(lat, lon)
        if url not in archives:
            try:
                archives[url] = zipfile.ZipFile(HttpRangeFile(session, url))
            except FileNotFoundError:
                archives[url] = None
        archive = archives[url]
        member = fabdemMember(lat, lon)
        if archive is None or member not in archive.namelist():
            continue
        raw = workDir / "raw" / member
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(archive.read(member))
        clipped = clipRaster(str(raw), bounds, workDir / f"fabdem_{cornerName(lat, lon)}.tif")
        if clipped is not None:
            out.append(clipped)
    return out


def _fetchTiles(
    session: HttpSession,
    bounds: Bounds,
    workDir: Path,
    sourceId: str,
    tiles: list[tuple[str, str]],
    nodata: float | None = None,
) -> list[Path]:
    out: list[Path] = []
    for name, url in tiles:
        try:
            clipped = clipRaster(
                url, bounds, workDir / f"{sourceId}_{name}.tif", session=session, nodata=nodata
            )
        except rasterio.errors.RasterioIOError:
            continue  # tile does not exist (e.g. open ocean)
        if clipped is not None:
            out.append(clipped)
    return out


def _xy(key: str) -> tuple[int, int]:
    x = y = 0
    for ch in key:
        x, y = x * 2 + (int(ch) & 1), y * 2 + (int(ch) >> 1)
    return x, y


def _key(x: int, y: int, zoom: int) -> str:
    return "".join(
        str((1 if x & (1 << (i - 1)) else 0) + (2 if y & (1 << (i - 1)) else 0))
        for i in range(zoom, 0, -1)
    )
