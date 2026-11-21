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
    _write(
        base / "theme=buildings" / "type=building_part" / "part-0.parquet",
        [
            {
                "id": f"p-{k}",
                "building_id": "b-in",
                "height": 42.0 if k else 60.0,
                "min_height": 30.0 if k else None,
                "num_floors": None,
                "min_floor": None,
                "roof_shape": "dome" if k else None,
                "roof_height": 8.0 if k else None,
                "roof_direction": None,
                "roof_orientation": None,
                "facade_color": "#5D7877" if k else None,
                "roof_color": None,
                "is_underground": False,
                "geometry": shapely.box(103.81 + 5e-4 * k, 1.29, 103.8105 + 5e-4 * k, 1.291),
            }
            for k in range(2)
        ]
        + [
            {
                "id": "p-under",
                "building_id": "b-noattrs",
                "height": 3.0,
                "min_height": None,
                "num_floors": None,
                "min_floor": None,
                "roof_shape": None,
                "roof_height": None,
                "roof_direction": None,
                "roof_orientation": None,
                "is_underground": True,
                "geometry": shapely.box(103.82, 1.30, 103.821, 1.301),
            }
        ],
    )
    _write(
        base / "theme=base" / "type=water" / "part-0.parquet",
        [
            {
                "id": "w-bay",
                "subtype": "ocean",
                "class": "ocean",
                "geometry": shapely.box(103.83, 1.27, 103.85, 1.29),  # crosses the box edge
            },
            {
                "id": "w-pool",
                "subtype": "human_made",
                "class": "swimming_pool",
                "geometry": shapely.box(103.81, 1.30, 103.8101, 1.3001),
            },
            {
                "id": "w-stream",
                "subtype": "stream",
                "class": "stream",
                "geometry": shapely.LineString([(103.81, 1.31), (103.82, 1.31)]),
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
                "class": "primary",
                "subtype": "road",
                "road_flags": flags(["is_bridge"]),
                "rail_flags": None,
                "level_rules": [{"value": 1, "between": None}],
                "width_rules": [{"value": 12.0, "between": None}],
                "geometry": line,
            },
            {
                "id": "s-partial",
                "class": "secondary",
                "subtype": "road",
                "road_flags": flags(["is_bridge"], [0.25, 0.75]),
                "rail_flags": None,
                "level_rules": [{"value": 1, "between": [0.25, 0.75]}],
                "width_rules": None,
                "geometry": line,
            },
            {
                "id": "s-tunnel",
                "class": "motorway",
                "subtype": "road",
                "road_flags": flags(["is_tunnel"]),
                "rail_flags": None,
                "level_rules": [{"value": -1, "between": None}],
                "width_rules": None,
                "geometry": line,
            },
            {
                "id": "s-rail",
                "class": "standard_gauge",
                "subtype": "rail",
                "road_flags": None,
                "rail_flags": flags(["is_bridge"]),
                "level_rules": None,
                "width_rules": None,
                "geometry": line,
            },
            {
                "id": "s-foot",
                "subtype": "road",
                "class": "footway",
                "road_flags": flags(["is_bridge"]),
                "rail_flags": None,
                "level_rules": None,
                "width_rules": None,
                "geometry": line,
            },
            {
                "id": "s-water",
                "class": None,
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
    # b-in is mapped as parts, which replace it; the underground part is dropped.
    assert table.column("sourceFeatureId").to_pylist() == ["b-noattrs", "p-0", "p-1"]
    assert table.column("height_m").to_pylist() == [None, 60.0, 42.0]
    assert table.column("numFloors").to_pylist() == [None, None, None]
    assert table.column("minHeight_m").to_pylist() == [None, None, 30.0]
    assert table.column("roofShape").to_pylist() == [None, None, "dome"]
    assert table.column("roofHeight_m").to_pylist() == [None, None, 8.0]
    assert set(table.column_names) >= {"minFloor", "roofDirection_deg", "roofOrientation"}
    assert table.column("facadeColor").to_pylist() == [None, None, "#5D7877"]
    assert table.column("roofColor").to_pylist() == [None, None, None]
    assert geoms.shape == (3,)


def testBuildings_olderReleaseWithoutPartsOr3dColumns(tmp_path: Path):
    base = tmp_path / "bucket" / "release" / "2024-01-01.0"
    _write(
        base / "theme=buildings" / "type=building" / "part-0.parquet",
        [
            {
                "id": "b-old",
                "height": 9.0,
                "num_floors": 3,
                "is_underground": False,
                "geometry": shapely.box(103.81, 1.29, 103.811, 1.291),
            }
        ],
    )
    out = overture.fetchOvertureBuildings(
        pafs.LocalFileSystem(), str(base.as_posix()), BOUNDS, tmp_path / "b.parquet"
    )
    table, _ = readGeoParquet(out)
    assert table.column("sourceFeatureId").to_pylist() == ["b-old"]
    assert table.column("roofShape").to_pylist() == [None]


def testTransport_splitsFlagsDropsTunnelsAndWater(tmp_path: Path, release: str):
    out = overture.fetchOvertureTransport(
        pafs.LocalFileSystem(), release, BOUNDS, tmp_path / "t.parquet"
    )
    table, geoms = readGeoParquet(out)
    rows = table.to_pylist()
    ids = [r["sourceFeatureId"] for r in rows]
    assert ids == ["s-bridge", "s-partial#0", "s-partial#1", "s-partial#2", "s-rail", "s-foot"]
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
    assert byId["s-foot"]["kind"] == "footway" and byId["s-foot"]["isBridge"] is True
    middle = geoms[ids.index("s-partial#1")]
    np.testing.assert_allclose(shapely.bounds(middle), [103.815, 1.30, 103.825, 1.30], atol=1e-9)


def testS3FileSystem_passesCaBundle(tmp_path: Path):
    bundle = tmp_path / "ca.pem"
    bundle.write_text("x", encoding="ascii")
    fs = overture.s3FileSystem(bundle)
    assert isinstance(fs, pafs.S3FileSystem)
    assert isinstance(overture.s3FileSystem(None), pafs.S3FileSystem)


def testWater_polygonsClippedToTheBoxLinesDropped(tmp_path: Path, release: str):
    out = overture.fetchOvertureWater(
        pafs.LocalFileSystem(), release, BOUNDS, tmp_path / "w.parquet"
    )
    table, geoms = readGeoParquet(out)
    assert table.column("sourceFeatureId").to_pylist() == ["w-bay", "w-pool"]
    assert table.column("landClass").to_pylist() == [80, 80]
    x0, y0, x1, y1 = shapely.total_bounds(geoms)
    assert (x0, y0) == pytest.approx((103.81, 1.28)) and (x1, y1) == pytest.approx((103.84, 1.3001))
