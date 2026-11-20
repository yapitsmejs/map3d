"""Tests for ``map3d.prepare.sources.overture`` (local Parquet with Overture's schema)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest
import shapely

from map3d.pack.geoparquet import readGeoParquet
from map3d.prepare.sources import overture

BOUNDS = (103.80, 1.28, 103.84, 1.32)


def _bbox(geom) -> dict[str, float]:
    x0, y0, x1, y1 = shapely.bounds(geom)
    return {"xmin": x0, "xmax": x1, "ymin": y0, "ymax": y1}


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        row["bbox"] = _bbox(row["geometry"])
        row["geometry"] = shapely.to_wkb(row["geometry"])
    pq.write_table(pa.Table.from_pylist(rows), path)


@pytest.fixture
def release(tmp_path: Path) -> str:
    base = tmp_path / "bucket" / "release" / "2026-11-18.0"
    (tmp_path / "bucket" / "release" / "2026-10-21.0").mkdir(parents=True)
    _write(
        base / "theme=buildings" / "type=building" / "part-0.parquet",
        [
            {
                "id": "b-in",
                "height": 42.0,
                "num_floors": 12,
                "is_underground": False,
                "geometry": shapely.box(103.81, 1.29, 103.811, 1.291),
            },
            {
                "id": "b-noattrs",
                "height": None,
                "num_floors": None,
                "is_underground": None,
                "geometry": shapely.box(103.82, 1.30, 103.821, 1.301),
            },
            {
                "id": "b-under",
                "height": 5.0,
                "num_floors": 1,
                "is_underground": True,
                "geometry": shapely.box(103.815, 1.295, 103.816, 1.296),
            },
            {
                "id": "b-out",
                "height": 10.0,
                "num_floors": 3,
                "is_underground": False,
                "geometry": shapely.box(104.5, 1.5, 104.51, 1.51),
            },
        ],
    )
    line = shapely.LineString([(103.81, 1.30), (103.83, 1.30)])
    flags = lambda values, between=None: [{"values": values, "between": between}]  # noqa: E731
    _write(
        base / "theme=transportation" / "type=segment" / "part-0.parquet",
        [
            {
                "id": "s-bridge",
                "subtype": "road",
                "road_flags": flags(["is_bridge"]),
                "rail_flags": None,
                "level_rules": [{"value": 1, "between": None}],
                "width_rules": [{"value": 12.0, "between": None}],
                "geometry": line,
            },
            {
                "id": "s-partial",
                "subtype": "road",
                "road_flags": flags(["is_bridge"], [0.25, 0.75]),
                "rail_flags": None,
                "level_rules": [{"value": 1, "between": [0.25, 0.75]}],
                "width_rules": None,
                "geometry": line,
            },
            {
                "id": "s-tunnel",
                "subtype": "road",
                "road_flags": flags(["is_tunnel"]),
                "rail_flags": None,
                "level_rules": [{"value": -1, "between": None}],
                "width_rules": None,
                "geometry": line,
            },
            {
                "id": "s-rail",
                "subtype": "rail",
                "road_flags": None,
                "rail_flags": flags(["is_bridge"]),
                "level_rules": None,
                "width_rules": None,
                "geometry": line,
            },
            {
                "id": "s-water",
                "subtype": "water",
                "road_flags": None,
                "rail_flags": None,
                "level_rules": None,
                "width_rules": None,
                "geometry": line,
            },
        ],
    )
    return str(base.as_posix())


def testLatestRelease_picksNewest(tmp_path: Path, release: str):
    fs = pafs.LocalFileSystem()
    assert overture.latestRelease(fs, str((tmp_path / "bucket").as_posix())) == "2026-11-18.0"


def testBuildings_canonicalAndFiltered(tmp_path: Path, release: str):
    out = overture.fetchOvertureBuildings(
        pafs.LocalFileSystem(), release, BOUNDS, tmp_path / "b.parquet"
    )
    table, geoms = readGeoParquet(out)
    assert table.column("sourceFeatureId").to_pylist() == ["b-in", "b-noattrs"]
    assert table.column("height_m").to_pylist() == [42.0, None]
    assert table.column("numFloors").to_pylist() == [12, None]
    assert geoms.shape == (2,)


def testTransport_splitsFlagsDropsTunnelsAndWater(tmp_path: Path, release: str):
    out = overture.fetchOvertureTransport(
        pafs.LocalFileSystem(), release, BOUNDS, tmp_path / "t.parquet"
    )
    table, geoms = readGeoParquet(out)
    rows = table.to_pylist()
    ids = [r["sourceFeatureId"] for r in rows]
    assert ids == ["s-bridge", "s-partial#0", "s-partial#1", "s-partial#2", "s-rail"]
    byId = {r["sourceFeatureId"]: r for r in rows}
    assert byId["s-bridge"] == {
        "sourceFeatureId": "s-bridge",
        "kind": "road",
        "isBridge": True,
        "level": 1,
        "width_m": 12.0,
    }
    assert [byId[f"s-partial#{k}"]["isBridge"] for k in range(3)] == [False, True, False]
    assert [byId[f"s-partial#{k}"]["level"] for k in range(3)] == [0, 1, 0]
    assert byId["s-rail"]["kind"] == "rail" and byId["s-rail"]["isBridge"] is True
    middle = geoms[ids.index("s-partial#1")]
    np.testing.assert_allclose(shapely.bounds(middle), [103.815, 1.30, 103.825, 1.30], atol=1e-9)


def testS3FileSystem_passesCaBundle(tmp_path: Path):
    bundle = tmp_path / "ca.pem"
    bundle.write_text("x", encoding="ascii")
    fs = overture.s3FileSystem(bundle)
    assert isinstance(fs, pafs.S3FileSystem)
    assert isinstance(overture.s3FileSystem(None), pafs.S3FileSystem)
