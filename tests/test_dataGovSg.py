"""Tests for ``map3d.prepare.sources.dataGovSg`` (adapters + download client)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import shapely
from _packSupport import mockSession

from map3d.pack.geoparquet import readGeoParquet
from map3d.prepare.sources import dataGovSg as dgs

BOUNDS = (103.6, 1.2, 104.1, 1.5)


def _geojson(path: Path, features: list[tuple[object, dict]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "geometry": shapely.geometry.mapping(g), "properties": p}
                    for g, p in features
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _box(i: int) -> shapely.Polygon:
    return shapely.box(103.8 + i * 0.001, 1.3, 103.8005 + i * 0.001, 1.3005)


def testJoinStreetCodes_blockSetsWithPrefixTiebreak():
    codeBlocks = {
        "YIA06H": {"101", "102", "103"},
        "TAS40G": {"101", "102"},  # same blocks exist on two streets -> prefix decides
        "ZZZ01A": {"999"},  # unknown block -> unresolved
    }
    nameBlocks = {
        "YISHUN AVE 6": {"101", "102", "103", "104"},
        "TAMPINES ST 33": {"101", "102"},
        "TOA PAYOH LOR 1": {"101", "102"},
    }
    assert dgs.joinStreetCodes(codeBlocks, nameBlocks) == {
        "YIA06H": "YISHUN AVE 6",
        "TAS40G": "TAMPINES ST 33",
    }


def testJoinStreetCodes_unresolvableTieIsDropped():
    codeBlocks = {"QQQ01A": {"1", "2"}}
    nameBlocks = {"ALPHA RD": {"1", "2"}, "BETA RD": {"1", "2"}}
    assert dgs.joinStreetCodes(codeBlocks, nameBlocks) == {}


def testAdaptHdb_joinsStoreysAndClips(tmp_path: Path):
    footprints = _geojson(
        tmp_path / "hdb.geojson",
        [
            (_box(0), {"BLK_NO": "101", "ST_COD": "YIA06H", "POSTAL_COD": "760101"}),
            (_box(1), {"BLK_NO": "102 ", "ST_COD": "YIA06H", "POSTAL_COD": "760102"}),
            (_box(2), {"BLK_NO": "7", "ST_COD": "NOP01X", "POSTAL_COD": "000007"}),
            (
                shapely.box(110.0, 1.0, 110.1, 1.1),
                {"BLK_NO": "103", "ST_COD": "YIA06H", "POSTAL_COD": "x"},
            ),
        ],
    )
    csvPath = tmp_path / "info.csv"
    csvPath.write_text(
        "blk_no,street,max_floor_lvl,year_completed\n101,YISHUN AVE 6,12,1990\n102,YISHUN AVE 6,9,1991\n103,YISHUN AVE 6,4,1992\n",
        encoding="utf-8",
    )
    out = dgs.adaptHdbBuildings(footprints, csvPath, BOUNDS, tmp_path / "out.parquet")
    table, geoms = readGeoParquet(out)
    assert table.column("sourceFeatureId").to_pylist() == ["760101", "760102", "000007"]
    assert table.column("numFloors").to_pylist() == [12, 9, None]
    assert table.column("height_m").to_pylist() == [None, None, None]


def testAdaptUraBuildings(tmp_path: Path):
    src = _geojson(tmp_path / "ura.geojson", [(_box(0), {"OBJECTID": "32", "BLDG_TYPE": "HDB"})])
    table, _ = readGeoParquet(dgs.adaptUraBuildings(src, BOUNDS, tmp_path / "o.parquet"))
    assert table.to_pylist() == [{"sourceFeatureId": "32", "height_m": None, "numFloors": None}]


def testAdaptMasterPlanRail_keepsAbovegroundAsBridges(tmp_path: Path):
    line = shapely.LineString([(103.8, 1.3), (103.81, 1.31)])
    src = _geojson(
        tmp_path / "rail.geojson",
        [
            (line, {"OBJECTID": "1", "GRND_LEVEL": "ABOVEGROUND", "RAIL_TYPE": "MRT"}),
            (line, {"OBJECTID": "2", "GRND_LEVEL": "UNDERGROUND", "RAIL_TYPE": "MRT"}),
        ],
    )
    table, _ = readGeoParquet(dgs.adaptMasterPlanRail(src, BOUNDS, tmp_path / "o.parquet"))
    assert table.to_pylist() == [
        {"sourceFeatureId": "1", "kind": "rail", "isBridge": True, "level": None, "width_m": None}
    ]


def testAdaptLtaStructures_excludesTunnelsAndUnderpasses(tmp_path: Path):
    src = _geojson(
        tmp_path / "lta.geojson",
        [
            (_box(0), {"OBJECTID_1": "1", "TYP_CD": "Flyover", "STRUCTURE_NAME": "A FLYOVER"}),
            (_box(1), {"OBJECTID_1": "2", "TYP_CD": "Vehicular Bridge", "STRUCTURE_NAME": "B"}),
            (_box(2), {"OBJECTID_1": "3", "TYP_CD": "Vehicular Tunnel", "STRUCTURE_NAME": "C"}),
            (_box(3), {"OBJECTID_1": "4", "TYP_CD": "Vehicular Underpass", "STRUCTURE_NAME": "D"}),
        ],
    )
    table, _ = readGeoParquet(dgs.adaptLtaStructures(src, BOUNDS, tmp_path / "o.parquet"))
    assert table.column("structureType").to_pylist() == ["Flyover", "Vehicular Bridge"]


def testDownloadDataset_pollsAndCaches(tmp_path: Path):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("/initiate-download"):
            return httpx.Response(201, json={"code": 0, "data": {"message": "initiated"}})
        if request.url.path.endswith("/poll-download"):
            if calls.count(request.url.path) == 1:
                return httpx.Response(201, json={"code": 0, "data": {"status": "PENDING"}})
            return httpx.Response(
                201, json={"code": 0, "data": {"url": "https://blobs.example/file.csv"}}
            )
        return httpx.Response(200, content=b"blk_no,street\n")

    with mockSession(handler) as session:
        path = dgs.downloadDataset(session, "d_abc", tmp_path / "info.csv", isCsv=True)
        assert path.read_bytes() == b"blk_no,street\n"
        dgs.downloadDataset(session, "d_abc", tmp_path / "info.csv", isCsv=True)
    assert sum(p.endswith("initiate-download") for p in calls) == 1  # cached second time


def testDownloadDataset_givesUpWhenNeverReady(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(201, json={"code": 0, "data": {"status": "PENDING"}})

    with mockSession(handler) as session:
        with pytest.raises(RuntimeError, match="not ready"):
            dgs.downloadDataset(session, "d_abc", tmp_path / "x.geojson", isCsv=False, maxPolls=3)


def testSessionThrottle_forDataGovSg():
    assert dgs.MIN_INTERVAL_s["api-open.data.gov.sg"] >= 5.0
