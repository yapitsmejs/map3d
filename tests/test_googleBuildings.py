"""Tests for ``map3d.prepare.sources.googleBuildings`` (Open Buildings 2.5D Temporal)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pytest
import rasterio
from _packSupport import mockSession, writeBands
from _support import rangeServer
from rasterio.transform import from_origin

from map3d.prepare.sources import googleBuildings as gob

SG = (103.80, 1.28, 103.84, 1.32)


def testUtmEpsgs_northSouthAndZoneSpanning():
    assert gob.utmEpsgs(SG) == [32648]
    assert gob.utmEpsgs((107.5, -0.5, 108.5, 0.5)) == [
        32648,
        32649,
        32748,
        32749,
    ]  # zone edge at 108E


def testGsToHttps():
    assert gob.gsToHttps("gs://bucket/a/b.tif") == "https://storage.googleapis.com/bucket/a/b.tif"


def _manifest(prefix: str, tiles: list[tuple[str, float, float]], size: int = 25000) -> dict:
    return {
        "uriPrefix": prefix,
        "tilesets": [
            {
                "sources": [
                    {
                        "uris": [uri],
                        "affineTransform": {
                            "scaleX": 0.5,
                            "translateX": x0,
                            "scaleY": -0.5,
                            "translateY": y1,
                        },
                        "dimensions": {"width": size, "height": size},
                    }
                    for uri, x0, y1 in tiles
                ]
            }
        ],
    }


def testTilesIntersecting_selectsByUtmExtent():
    # SG box is around E 366-371 km, N 141-146 km in UTM 48N.
    manifest = _manifest(
        "gs://open-buildings-temporal-data/v1/geotiffs/31",
        [
            ("aa_2023_06_30/tile_hit.tif", 366000.0, 146000.0),
            ("aa_2023_06_30/tile_far.tif", 100000.0, 146000.0),
        ],
    )
    tiles = gob.tilesIntersecting(manifest, SG, 32648)
    assert tiles == [
        (
            "https://storage.googleapis.com/open-buildings-temporal-data/v1/geotiffs/31aa_2023_06_30/tile_hit.tif",
            "tile_hit",
        )
    ]


def _threeBandTile(path: Path, x0: float, y1: float) -> bytes:
    data = np.stack([np.full((400, 400), v, np.float32) for v in (0.5, 21.0, 0.9)])
    data[1, :8, :8] = -99.0
    return writeBands(path, data, "EPSG:32648", from_origin(x0, y1, 0.5, 0.5), -99.0).read_bytes()


def testFetch_listsManifestsSelectsTilesAndClipsHeightBand(tmp_path: Path):
    from rasterio.warp import transform_bounds

    query = (103.83, 1.295, 103.835, 1.30)
    left, _, _, top = transform_bounds("EPSG:4326", "EPSG:32648", *query)
    x0, y1 = float(int(left) - 20), float(int(top) + 20)  # 200 m tile covering the query box
    prefix = "gs://open-buildings-temporal-data/v1/geotiffs/31"
    manifest = _manifest(prefix, [("aa_2023_06_30/tile_hit.tif", x0, y1)], size=400)
    manifestName = "v1/manifests/31_EPSG_32648_2023_06_30.json"
    tileUrl = gob.gsToHttps(prefix) + "aa_2023_06_30/tile_hit.tif"
    tileBytes = _threeBandTile(tmp_path / "t.tif", x0, y1)
    files = {
        f"{gob.BUCKET_HTTP}/{manifestName}": json.dumps(manifest).encode(),
        tileUrl: tileBytes,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/storage/v1/b/"):
            if request.url.params.get("pageToken") == "p2":
                items = [
                    {"name": manifestName},
                    {"name": "v1/manifests/31_EPSG_32648_2022_06_30.json"},
                ]
                return httpx.Response(200, json={"items": items})
            return httpx.Response(
                200,
                json={
                    "items": [{"name": "v1/manifests/01_EPSG_32723_2023_06_30.json"}],
                    "nextPageToken": "p2",
                },
            )
        return rangeServer(files)(request)

    with mockSession(handler) as session:
        out = gob.fetchGoogleBuildings(session, query, tmp_path / "work", year=2023)
    assert [p.name for p in out] == ["googleOpenBuildings25d_tile_hit.tif"]
    with rasterio.open(out[0]) as src:
        assert src.nodata == -99.0
        assert src.res == pytest.approx((4.0, 4.0))  # 0.5 m decimated 8x
        values = src.read(1)
        assert np.nanmax(values) == 21.0  # band 2 = building_height
    assert (
        tmp_path / "work" / "raw" / "31_EPSG_32648_2023_06_30.json"
    ).is_file()  # manifest cached


def testFetch_noManifestForZoneReturnsEmpty(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"items": []})

    with mockSession(handler) as session:
        assert gob.fetchGoogleBuildings(session, SG, tmp_path / "work") == []
