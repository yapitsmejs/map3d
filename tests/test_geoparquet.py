"""Tests for ``map3d.pack.geoparquet``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyproj
import pytest
import shapely

from map3d.pack import geoparquet


def _sample() -> tuple[np.ndarray, dict[str, list[object]]]:
    geoms = np.array(
        [shapely.box(103.80, 1.30, 103.81, 1.31), shapely.box(103.82, 1.28, 103.83, 1.29)]
    )
    return geoms, {"sourceFeatureId": ["a", "b"], "height_m": [12.5, None]}


def testWriteRead_roundTrips(tmp_path: Path):
    geoms, cols = _sample()
    path = tmp_path / "b.parquet"
    geoparquet.writeGeoParquet(path, geoms, cols)
    table, geomsBack = geoparquet.readGeoParquet(path)
    assert shapely.equals(geomsBack, geoms).all()
    assert table.column("sourceFeatureId").to_pylist() == ["a", "b"]
    assert table.column("height_m").to_pylist() == [12.5, None]
    assert "geometry" not in table.column_names


def testWrite_recordsGeoMetadata(tmp_path: Path):
    geoms, cols = _sample()
    path = tmp_path / "b.parquet"
    geoparquet.writeGeoParquet(path, geoms, cols)
    geo = json.loads(pq.read_schema(path).metadata[b"geo"])
    assert geo["primary_column"] == "geometry"
    column = geo["columns"]["geometry"]
    assert column["encoding"] == "WKB"
    assert column["geometry_types"] == ["Polygon"]
    assert column["bbox"] == pytest.approx([103.80, 1.28, 103.83, 1.31])


def testRead_selectsColumns(tmp_path: Path):
    geoms, cols = _sample()
    path = tmp_path / "b.parquet"
    geoparquet.writeGeoParquet(path, geoms, cols)
    table, _ = geoparquet.readGeoParquet(path, columns=["height_m"])
    assert table.column_names == ["height_m"]


def testInspect_reportsBoundsColumnsRows(tmp_path: Path):
    geoms, cols = _sample()
    path = tmp_path / "b.parquet"
    geoparquet.writeGeoParquet(path, geoms, cols)
    info = geoparquet.inspectGeoParquet(path)
    assert info.numRows == 2
    assert info.columns == ("sourceFeatureId", "height_m")
    assert info.bounds_deg == pytest.approx((103.80, 1.28, 103.83, 1.31))


def testInspect_computesBoundsWhenMetadataLacksBbox(tmp_path: Path):
    geoms, cols = _sample()
    path = tmp_path / "b.parquet"
    _writeRaw(
        path,
        geoms,
        {
            "version": "1.0.0",
            "primary_column": "geometry",
            "columns": {"geometry": {"encoding": "WKB"}},
        },
    )
    assert geoparquet.inspectGeoParquet(path).bounds_deg == pytest.approx(
        (103.80, 1.28, 103.83, 1.31)
    )


def testInspect_emptyFileHasNoBounds(tmp_path: Path):
    path = tmp_path / "e.parquet"
    geoparquet.writeGeoParquet(path, np.array([], dtype=object), {"sourceFeatureId": []})
    info = geoparquet.inspectGeoParquet(path)
    assert info.numRows == 0
    assert info.bounds_deg is None


def testInspect_acceptsEpsg4326Crs(tmp_path: Path):
    geoms, _ = _sample()
    path = tmp_path / "c.parquet"
    crs = pyproj.CRS("EPSG:4326").to_json_dict()
    _writeRaw(
        path,
        geoms,
        {
            "version": "1.1.0",
            "primary_column": "geometry",
            "columns": {"geometry": {"encoding": "WKB", "crs": crs}},
        },
    )
    assert geoparquet.inspectGeoParquet(path).numRows == 2


@pytest.mark.parametrize(
    ("geo", "message"),
    [
        (None, "not a GeoParquet"),
        (
            {
                "version": "1.0.0",
                "primary_column": "geometry",
                "columns": {"geometry": {"encoding": "point"}},
            },
            "WKB",
        ),
        (
            {
                "version": "1.0.0",
                "primary_column": "geometry",
                "columns": {
                    "geometry": {"encoding": "WKB", "crs": pyproj.CRS("EPSG:3857").to_json_dict()}
                },
            },
            "CRS84",
        ),
    ],
)
def testInspect_rejectsUnsupported(tmp_path: Path, geo: dict[str, object] | None, message: str):
    geoms, _ = _sample()
    path = tmp_path / "bad.parquet"
    _writeRaw(path, geoms, geo)
    with pytest.raises(ValueError, match=message):
        geoparquet.inspectGeoParquet(path)


def _writeRaw(path: Path, geoms: np.ndarray, geo: dict[str, object] | None) -> None:
    table = pa.table({"sourceFeatureId": ["a", "b"], "geometry": shapely.to_wkb(geoms)})
    if geo is not None:
        table = table.replace_schema_metadata({b"geo": json.dumps(geo).encode()})
    pq.write_table(table, path)
