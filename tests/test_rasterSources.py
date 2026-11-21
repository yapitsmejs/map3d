"""Tests for ``map3d.prepare.sources.rasters`` (tile naming, quadkeys, fetchers)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pytest
import rasterio
from _packSupport import mockSession
from _support import rangeServer

from map3d.prepare.sources import rasters

SG = (103.6, 1.2, 104.1, 1.5)


def testCellCorners_coverBoundsAtStep():
    assert rasters.cellCorners(SG, 1) == [(1, 103), (1, 104)]
    assert rasters.cellCorners((103.5, -0.5, 104.5, 0.5), 1) == [
        (-1, 103),
        (-1, 104),
        (0, 103),
        (0, 104),
    ]
    assert rasters.cellCorners(SG, 3) == [(0, 102)]
    assert rasters.cellCorners(SG, 10) == [(0, 100)]
    assert rasters.cellCorners((103.0, 1.0, 104.0, 2.0), 1) == [(1, 103)]  # max edge exclusive


@pytest.mark.parametrize(
    ("lat", "lon", "name"),
    [(1, 103, "N01E103"), (-1, 103, "S01E103"), (0, -71, "N00W071"), (-10, 90, "S10E090")],
)
def testCornerName(lat: int, lon: int, name: str):
    assert rasters.cornerName(lat, lon) == name


def testUrls_followProviderNaming():
    assert rasters.glo30Url(1, 103).endswith(
        "/Copernicus_DSM_COG_10_N01_00_E103_00_DEM/Copernicus_DSM_COG_10_N01_00_E103_00_DEM.tif"
    )
    assert rasters.glo30Url(-2, -71).endswith("Copernicus_DSM_COG_10_S02_00_W071_00_DEM.tif")
    assert rasters.worldCoverUrl(0, 102).endswith(
        "/v200/2021/map/ESA_WorldCover_10m_2021_v200_N00E102_Map.tif"
    )
    assert rasters.fabdemZipUrl(1, 103).endswith("/N00E100-N10E110_FABDEM_V1-2.zip")
    assert rasters.fabdemZipUrl(-1, 103).endswith("/S10E100-N00E110_FABDEM_V1-2.zip")
    assert rasters.fabdemMember(1, 103) == "N01E103_FABDEM_V1-2.tif"


def testQuadkey_roundTripsAndContainsPoint():
    qk = rasters.quadkey(1.3, 103.8, 9)
    assert len(qk) == 9 and set(qk) <= set("0123")
    minLon, minLat, maxLon, maxLat = rasters.quadkeyBounds(qk)
    assert minLon <= 103.8 < maxLon and minLat <= 1.3 < maxLat
    assert rasters.quadkey(-33.9, 151.2, 3) == "311"


def testQuadkeysCoveringBounds():
    keys = rasters.quadkeysCovering(SG, 9)
    assert len(keys) == len(set(keys)) >= 1
    for key in keys:
        b = rasters.quadkeyBounds(key)
        assert b[0] < SG[2] and SG[0] < b[2] and b[1] < SG[3] and SG[1] < b[3]


def _tif(writeRaster, path: Path, bounds, value: float, crs="EPSG:4326") -> bytes:
    return writeRaster(
        path, bounds, values=np.full((40, 40), value, np.float32), shape=(40, 40), crs=crs
    ).read_bytes()


def testFetchTiles_clipsEachTileAndSkipsMissing(tmp_path: Path, writeRaster):
    files = {
        rasters.glo30Url(1, 103): _tif(
            writeRaster, tmp_path / "a.tif", (103.0, 1.0, 104.0, 2.0), 7.0
        )
    }
    # Tile (1, 104) is absent (ocean) -> skipped.
    with mockSession(rangeServer(files)) as session:
        out = rasters.fetchGlo30(session, SG, tmp_path / "work")
    assert [p.name for p in out] == ["copernicusGlo30_N01E103.tif"]
    with rasterio.open(out[0]) as src:
        assert float(src.read(1).mean()) == 7.0


def testFetchFabdem_extractsOnlyNeededZipMembers(tmp_path: Path, writeRaster):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "N01E103_FABDEM_V1-2.tif",
            _tif(writeRaster, tmp_path / "f.tif", (103.0, 1.0, 104.0, 2.0), 3.0),
        )
        zf.writestr("N05E105_FABDEM_V1-2.tif", b"not needed")
    log: list[httpx.Request] = []
    with mockSession(rangeServer({rasters.fabdemZipUrl(1, 103): buf.getvalue()}, log)) as session:
        out = rasters.fetchFabdem(session, SG, tmp_path / "work")
    assert [p.name for p in out] == ["fabdem_N01E103.tif"]
    with rasterio.open(out[0]) as src:
        assert float(src.read(1).mean()) == 3.0


def testFetchFabdem_missingZipIsSkipped(tmp_path: Path):
    with mockSession(rangeServer({})) as session:
        assert rasters.fetchFabdem(session, SG, tmp_path / "work") == []


def testFetchEgm2008_singleClippedFile(tmp_path: Path, writeRaster):
    files = {
        rasters.EGM2008_URL: _tif(
            writeRaster, tmp_path / "g.tif", (-180.0, -90.0, 180.0, 90.0), 8.5
        )
    }
    with mockSession(rangeServer(files)) as session:
        out = rasters.fetchEgm2008(session, SG, tmp_path / "work")
    assert [p.name for p in out] == ["egm2008.tif"]
    with rasterio.open(out[0]) as src:
        assert src.bounds.left <= SG[0] and src.bounds.right >= SG[2]


def testFetchWorldCoverAndCanopy(tmp_path: Path, writeRaster):
    qk = rasters.quadkeysCovering(SG, 9)
    files = {
        rasters.worldCoverUrl(0, 102): _tif(
            writeRaster, tmp_path / "w.tif", (102.0, 0.0, 105.0, 3.0), 30.0
        )
    }
    for key in qk:
        b = rasters.quadkeyBounds(key)
        files[rasters.metaCanopyUrl(key)] = _tif(writeRaster, tmp_path / f"{key}.tif", b, 5.0)
    with mockSession(rangeServer(files)) as session:
        wc = rasters.fetchWorldCover(session, SG, tmp_path / "work")
        canopy = rasters.fetchMetaCanopy(session, SG, tmp_path / "work")
    assert [p.name for p in wc] == ["esaWorldCover_N00E102.tif"]
    assert sorted(p.name for p in canopy) == sorted(f"metaCanopy_{k}.tif" for k in qk)
