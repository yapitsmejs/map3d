"""Shared pytest fixtures: tiny raster / vector file writers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
import shapely
from rasterio.transform import from_bounds

from map3d.pack.geoparquet import writeGeoParquet

RasterWriter = Callable[..., Path]
VectorWriter = Callable[..., Path]


@pytest.fixture
def writeRaster() -> RasterWriter:
    """Return ``f(path, bounds, values=None, crs="EPSG:4326", shape=(8, 8), nodata=None)``."""

    def _write(
        path: Path,
        bounds: tuple[float, float, float, float],
        values: np.ndarray | None = None,
        crs: str | None = "EPSG:4326",
        shape: tuple[int, int] = (8, 8),
        nodata: float | None = None,
    ) -> Path:
        data = np.zeros(shape, dtype=np.float32) if values is None else values
        height, width = data.shape
        path.parent.mkdir(parents=True, exist_ok=True)
        profile: dict[str, Any] = dict(
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=str(data.dtype),
            transform=from_bounds(*bounds, width, height),
            nodata=nodata,
        )
        if crs is not None:
            profile["crs"] = crs
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)
        return path

    return _write


@pytest.fixture
def writeBuildings() -> VectorWriter:
    """Return ``f(path, boxes, heights=None)`` writing canonical buildings GeoParquet."""

    def _write(
        path: Path,
        boxes: list[tuple[float, float, float, float]],
        heights: list[float | None] | None = None,
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        n = len(boxes)
        writeGeoParquet(
            path,
            np.array([shapely.box(*b) for b in boxes], dtype=object),
            {
                "sourceFeatureId": [f"b{i}" for i in range(n)],
                "height_m": heights if heights is not None else [10.0] * n,
                "numFloors": [None] * n,
            },
        )
        return path

    return _write
