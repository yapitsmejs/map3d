"""Tests for ``map3d.geoid.GeoidGrid`` (bilinear geoid undulation lookup)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from map3d.geoid import GeoidGrid


def _linearField(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    return 2.0 * lat_deg - 0.5 * lon_deg + 7.0


def _grid(descendingLat: bool = False) -> GeoidGrid:
    lat_deg = np.linspace(0.0, 3.0, 13)
    lon_deg = np.linspace(102.0, 106.0, 17)
    if descendingLat:
        lat_deg = lat_deg[::-1]
    latMesh, lonMesh = np.meshgrid(lat_deg, lon_deg, indexing="ij")
    return GeoidGrid(lat_deg, lon_deg, _linearField(latMesh, lonMesh))


@pytest.mark.parametrize("descendingLat", [False, True])
def testUndulation_bilinearExactOnLinearField(descendingLat: bool):
    grid = _grid(descendingLat)
    rng = np.random.default_rng(7)
    lat_deg = rng.uniform(0.0, 3.0, 500)
    lon_deg = rng.uniform(102.0, 106.0, 500)
    np.testing.assert_allclose(
        grid.undulation_m(lat_deg, lon_deg), _linearField(lat_deg, lon_deg), atol=1e-9
    )


def testUndulation_exactAtGridNodesAndEdges():
    grid = _grid()
    assert float(grid.undulation_m(0.0, 102.0)) == pytest.approx(_linearField(0.0, 102.0))
    assert float(grid.undulation_m(3.0, 106.0)) == pytest.approx(_linearField(3.0, 106.0))


def testUndulation_preservesShape():
    out = _grid().undulation_m(np.full((2, 3), 1.0), np.full((2, 3), 104.0))
    assert out.shape == (2, 3)


def testUndulation_rejectsOutsideCoverage():
    with pytest.raises(ValueError, match="outside"):
        _grid().undulation_m(np.array([1.0, 3.5]), np.array([104.0, 104.0]))


@pytest.mark.parametrize(
    ("lat", "lon", "values", "message"),
    [
        ([0.0, 1.0], [0.0, 1.0, 2.0], np.zeros((3, 3)), "shape"),
        ([0.0], [0.0, 1.0], np.zeros((1, 2)), "at least 2"),
        ([0.0, 1.0, 0.5], [0.0, 1.0], np.zeros((3, 2)), "monotonic"),
        ([0.0, 1.0], [1.0, 0.0], np.zeros((2, 2)), "monotonic"),
    ],
)
def testGeoidGrid_rejectsBadAxes(
    lat: list[float], lon: list[float], values: np.ndarray, message: str
):
    with pytest.raises(ValueError, match=message):
        GeoidGrid(np.array(lat), np.array(lon), values)


def testFromGeoTiff_readsNorthUpGrid(tmp_path: Path):
    res_deg = 0.25
    nRows, nCols = 12, 16
    west_deg, north_deg = 102.0, 3.0
    latCenters = north_deg - (np.arange(nRows) + 0.5) * res_deg
    lonCenters = west_deg + (np.arange(nCols) + 0.5) * res_deg
    latMesh, lonMesh = np.meshgrid(latCenters, lonCenters, indexing="ij")
    values = _linearField(latMesh, lonMesh).astype(np.float32)
    path = tmp_path / "geoid.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=nRows,
        width=nCols,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(west_deg, north_deg, res_deg, res_deg),
    ) as dst:
        dst.write(values, 1)

    grid = GeoidGrid.fromGeoTiff(path)
    np.testing.assert_allclose(
        grid.undulation_m(np.array([1.0, 2.2]), np.array([103.0, 104.9])),
        _linearField(np.array([1.0, 2.2]), np.array([103.0, 104.9])),
        atol=1e-4,
    )


def testFromGeoTiff_rejectsRotatedOrNonGeographic(tmp_path: Path):
    path = tmp_path / "bad.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_origin(0.0, 10.0, 1.0, 1.0),
    ) as dst:
        dst.write(np.zeros((2, 2), dtype=np.float32), 1)
    with pytest.raises(ValueError, match="geographic"):
        GeoidGrid.fromGeoTiff(path)
