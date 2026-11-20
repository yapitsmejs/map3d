"""Google Open Buildings 2.5D Temporal fetcher (``googleOpenBuildings25d``).

Data lives in a public GCS bucket, so no Earth Engine account is needed:

- ``v1/manifests/{s2token}_EPSG_{utm}_{year}_06_30.json`` are Earth Engine
  ingestion manifests. Each lists tile URIs (relative to ``uriPrefix``) and
  their UTM affine transforms.
- Tiles are 25000x25000 px GeoTIFFs at 0.5 m with internal overviews. Bands
  are ``building_fractional_count``, ``building_height`` and
  ``building_presence``; nodata is -99.

The height band is window-read and decimated 8x to ~4 m, which is its
effective resolution, so a city costs MBs instead of ~6.5 GB per year.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rasterio.warp import transform_bounds

from map3d.prepare.http import HttpSession
from map3d.prepare.rasterClip import clipRaster

__all__ = ["BUCKET_HTTP", "fetchGoogleBuildings", "gsToHttps", "tilesIntersecting", "utmEpsgs"]

Bounds = tuple[float, float, float, float]

BUCKET = "open-buildings-temporal-data"
BUCKET_HTTP = f"https://storage.googleapis.com/{BUCKET}"
_LISTING = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
HEIGHT_BAND = 2
NODATA = -99.0
DECIMATION = 8


def utmEpsgs(bounds: Bounds) -> list[int]:
    """EPSG codes of the WGS84 UTM zones (north/south) a lon/lat box touches."""
    minLon, minLat, maxLon, maxLat = bounds
    zones = range(int((minLon + 180) // 6) + 1, int((maxLon + 180) // 6) + 2)
    hemispheres = ([326] if maxLat >= 0 else []) + ([327] if minLat < 0 else [])
    return sorted(h * 100 + z for h in hemispheres for z in zones)


def gsToHttps(uri: str) -> str:
    """Convert a ``gs://bucket/key`` URI to its public HTTPS URL."""
    return uri.replace("gs://", "https://storage.googleapis.com/", 1)


def tilesIntersecting(manifest: dict[str, Any], bounds: Bounds, epsg: int) -> list[tuple[str, str]]:
    """Tiles of a manifest whose UTM extent intersects a lon/lat box.

    Args:
        manifest: Parsed manifest JSON.
        bounds: ``(minLon, minLat, maxLon, maxLat)``.
        epsg: The manifest's UTM EPSG code.

    Returns:
        ``(https URL, tile id)`` pairs.
    """
    left, bottom, right, top = transform_bounds("EPSG:4326", f"EPSG:{epsg}", *bounds)
    prefix = gsToHttps(manifest["uriPrefix"])
    tiles = []
    for tileset in manifest["tilesets"]:
        for source in tileset["sources"]:
            t, dims = source["affineTransform"], source["dimensions"]
            x0 = t["translateX"]
            x1 = x0 + dims["width"] * t["scaleX"]
            y1 = t["translateY"]
            y0 = y1 + dims["height"] * t["scaleY"]
            if x0 < right and left < x1 and y0 < top and bottom < y1:
                uri = source["uris"][0]
                tiles.append((prefix + uri, Path(uri).stem))
    return tiles


def fetchGoogleBuildings(
    session: HttpSession, bounds: Bounds, workDir: Path, year: int = 2023
) -> list[Path]:
    """Clip the building-height band of every tile covering ``bounds``.

    Args:
        session: HTTP session.
        bounds: ``(minLon, minLat, maxLon, maxLat)``.
        workDir: Working directory (manifests are cached in ``raw/``).
        year: Data year (2016-2023).

    Returns:
        Pack-ready GeoTIFFs (UTM, ~4 m, nodata -99).
    """
    names = _listManifests(session, year)
    out: list[Path] = []
    for epsg in utmEpsgs(bounds):
        for name in (n for n in names if f"_EPSG_{epsg}_" in n):
            manifest = _manifest(session, name, workDir)
            for url, tileId in tilesIntersecting(manifest, bounds, epsg):
                clipped = clipRaster(
                    url,
                    bounds,
                    workDir / f"googleOpenBuildings25d_{tileId}.tif",
                    session=session,
                    band=HEIGHT_BAND,
                    decimation=DECIMATION,
                    marginPixels=DECIMATION,
                    nodata=NODATA,
                )
                if clipped is not None:
                    out.append(clipped)
    return out


def _listManifests(session: HttpSession, year: int) -> list[str]:
    names: list[str] = []
    token: str | None = None
    while True:
        params: dict[str, Any] = {
            "prefix": "v1/manifests/",
            "fields": "items(name),nextPageToken",
            "maxResults": 1000,
        }
        if token:
            params["pageToken"] = token
        page = session.get(_LISTING, params=params).json()
        names += [
            item["name"] for item in page.get("items", []) if f"_{year}_06_30" in item["name"]
        ]
        token = page.get("nextPageToken")
        if not token:
            return names


def _manifest(session: HttpSession, name: str, workDir: Path) -> dict[str, Any]:
    cached = workDir / "raw" / Path(name).name
    session.download(f"{BUCKET_HTTP}/{name}", cached)
    result: dict[str, Any] = json.loads(cached.read_text(encoding="utf-8"))
    return result
