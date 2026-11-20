"""Minimal GeoParquet (v1.1, WKB, lon/lat) read/write helpers for pack vector data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import pyproj
import shapely

__all__ = ["GeoParquetInfo", "inspectGeoParquet", "readGeoParquet", "writeGeoParquet"]

GEOMETRY_COLUMN = "geometry"
_CRS84 = pyproj.CRS("OGC:CRS84")

Bounds = tuple[float, float, float, float]


@dataclass(frozen=True)
class GeoParquetInfo:
    """Summary of a GeoParquet file.

    Attributes:
        numRows: Number of features.
        columns: Attribute column names (excluding the geometry column).
        bounds_deg: ``(minLon, minLat, maxLon, maxLat)``, or ``None`` if empty.
        geometryColumn: Name of the primary geometry column.
    """

    numRows: int
    columns: tuple[str, ...]
    bounds_deg: Bounds | None
    geometryColumn: str


def writeGeoParquet(
    path: str | PathLike[str],
    geometries: npt.NDArray[np.object_],
    columns: dict[str, Any],
) -> None:
    """Write features to a GeoParquet file in lon/lat (OGC:CRS84).

    Args:
        path: Output path.
        geometries: Shapely geometries, one per feature.
        columns: Attribute columns (name -> sequence), same length as ``geometries``.
    """
    geoms = np.asarray(geometries, dtype=object)
    data: dict[str, Any] = dict(columns)
    data[GEOMETRY_COLUMN] = pa.array(shapely.to_wkb(geoms).tolist(), type=pa.binary())
    table = pa.table(data)
    columnMeta: dict[str, Any] = {
        "encoding": "WKB",
        "geometry_types": sorted({str(g.geom_type) for g in geoms}),
    }
    if geoms.size:
        columnMeta["bbox"] = [float(v) for v in shapely.total_bounds(geoms)]
    geo = {
        "version": "1.1.0",
        "primary_column": GEOMETRY_COLUMN,
        "columns": {GEOMETRY_COLUMN: columnMeta},
    }
    table = table.replace_schema_metadata({b"geo": json.dumps(geo).encode()})
    pq.write_table(table, Path(path))


def inspectGeoParquet(path: str | PathLike[str]) -> GeoParquetInfo:
    """Validate a GeoParquet file and summarise it.

    Args:
        path: Input path.

    Returns:
        File summary.

    Raises:
        ValueError: If the file is not WKB-encoded GeoParquet in lon/lat.
    """
    pqFile = pq.ParquetFile(Path(path))
    geometryColumn, columnMeta = _geoColumn(pqFile.schema_arrow.metadata, path)
    columns = tuple(n for n in pqFile.schema_arrow.names if n != geometryColumn)
    numRows = pqFile.metadata.num_rows
    bounds: Bounds | None = None
    if numRows:
        if "bbox" in columnMeta:
            bounds = _asBounds(columnMeta["bbox"])
        else:
            wkb = pqFile.read(columns=[geometryColumn]).column(geometryColumn).to_numpy(False)
            bounds = _asBounds(shapely.total_bounds(shapely.from_wkb(wkb)))
    return GeoParquetInfo(numRows, columns, bounds, geometryColumn)


def readGeoParquet(
    path: str | PathLike[str], columns: list[str] | None = None
) -> tuple[pa.Table, npt.NDArray[np.object_]]:
    """Read a GeoParquet file.

    Args:
        path: Input path.
        columns: Attribute columns to read (default: all).

    Returns:
        ``(attributes, geometries)``: an Arrow table without the geometry
        column, and a NumPy array of shapely geometries.
    """
    info = inspectGeoParquet(path)
    wanted = list(info.columns) if columns is None else list(columns)
    table = pq.read_table(Path(path), columns=[*wanted, info.geometryColumn])
    wkb = table.column(info.geometryColumn).to_numpy(zero_copy_only=False)
    geoms: npt.NDArray[np.object_] = shapely.from_wkb(wkb)
    return table.drop_columns([info.geometryColumn]), geoms


def _geoColumn(metadata: dict[bytes, bytes] | None, path: object) -> tuple[str, dict[str, Any]]:
    if not metadata or b"geo" not in metadata:
        raise ValueError(f"{path} is not a GeoParquet file (missing 'geo' metadata)")
    geo = json.loads(metadata[b"geo"])
    name = geo["primary_column"]
    columnMeta: dict[str, Any] = geo["columns"][name]
    if columnMeta.get("encoding") != "WKB":
        raise ValueError(
            f"{path}: geometry encoding must be WKB, got {columnMeta.get('encoding')!r}"
        )
    if "crs" in columnMeta:
        crsJson = columnMeta["crs"]
        crs = None if crsJson is None else pyproj.CRS.from_json_dict(crsJson)
        if crs is None or not crs.equals(_CRS84, ignore_axis_order=True):
            raise ValueError(f"{path}: geometry CRS must be lon/lat WGS84 (OGC:CRS84 / EPSG:4326)")
    return name, columnMeta


def _asBounds(values: Any) -> Bounds:
    minX, minY, maxX, maxY = (float(v) for v in values)
    return (minX, minY, maxX, maxY)
