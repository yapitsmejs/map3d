"""Synthetic data packs with analytic ground truth, for tests and demos.

Everything is defined in the scene-center ENU frame (meters), then written to a
real pack through :func:`map3d.pack.ingest.ingestFiles`, so the full read path
is exercised. The returned :class:`SyntheticScene` exposes the exact ground
truth (terrain function, feature boxes, ...) to compare scene outputs against.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import rasterio
import shapely
from rasterio.transform import from_bounds

from map3d.frames import LocalFrame
from map3d.pack.geoparquet import writeGeoParquet
from map3d.pack.ingest import ingestFiles

__all__ = [
    "SyntheticBuilding",
    "SyntheticRoad",
    "SyntheticScene",
    "SyntheticTree",
    "makeSyntheticPack",
]

FloatArray = npt.NDArray[np.float64]
TerrainKind = Literal["plane", "hill"]

_METERS_PER_DEG_LAT = 110_574.0
_WATER_NORTH_OF_m = 350.0
_CANOPY_BOX_m = (-400.0, -400.0, -300.0, -300.0)
_CANOPY_HEIGHT_m = 15.0


@dataclass(frozen=True)
class SyntheticBuilding:
    """ENU-aligned box building: footprint ``[e0, e1] x [n0, n1]``."""

    e0_m: float
    n0_m: float
    e1_m: float
    n1_m: float
    height_m: float | None
    numFloors: int | None


@dataclass(frozen=True)
class SyntheticRoad:
    """Straight road/rail centerline from ``(e0, n0)`` to ``(e1, n1)``."""

    e0_m: float
    n0_m: float
    e1_m: float
    n1_m: float
    isBridge: bool
    level: int
    width_m: float
    kind: str = "road"


@dataclass(frozen=True)
class SyntheticTree:
    """Single tree point."""

    e_m: float
    n_m: float
    height_m: float


@dataclass(frozen=True)
class SyntheticScene:
    """Ground truth for a synthetic pack.

    Attributes:
        packDir: Pack root directory.
        frame: Scene-center ENU frame used to define everything.
        terrain: Terrain kind (``"plane"`` or ``"hill"``).
        halfSize_m: Half-size of the square pack area, in meters.
        geoidUndulation_m: Constant geoid undulation everywhere.
        buildings: Building definitions, in file order.
        roads: Road/bridge definitions, in file order.
        trees: Tree definitions.
    """

    packDir: Path
    frame: LocalFrame
    terrain: TerrainKind
    halfSize_m: float
    geoidUndulation_m: float = 8.0
    buildings: tuple[SyntheticBuilding, ...] = field(default_factory=tuple)
    roads: tuple[SyntheticRoad, ...] = field(default_factory=tuple)
    trees: tuple[SyntheticTree, ...] = field(default_factory=tuple)

    def terrainHeight_m(self, e_m: npt.ArrayLike, n_m: npt.ArrayLike) -> FloatArray:
        """Ground height above MSL (EGM2008) at ENU positions.

        Args:
            e_m: East, in meters.
            n_m: North, in meters.

        Returns:
            Terrain height, in meters.
        """
        e = np.asarray(e_m, dtype=np.float64)
        n = np.asarray(n_m, dtype=np.float64)
        if self.terrain == "plane":
            return 10.0 + 0.02 * e - 0.01 * n
        return 10.0 + 30.0 * np.exp(-(e**2 + n**2) / (2.0 * 150.0**2))

    def canopyHeight_m(self, e_m: npt.ArrayLike, n_m: npt.ArrayLike) -> FloatArray:
        """Canopy height above ground: a 15 m patch, zero elsewhere."""
        inPatch = _inBox(
            np.asarray(e_m, dtype=np.float64), np.asarray(n_m, dtype=np.float64), _CANOPY_BOX_m
        )
        return np.where(inPatch, _CANOPY_HEIGHT_m, 0.0)

    def landClass(self, e_m: npt.ArrayLike, n_m: npt.ArrayLike) -> npt.NDArray[np.uint8]:
        """WorldCover class: tree (10) in the canopy patch, water (80) in the north, else grass (30)."""
        e = np.asarray(e_m, dtype=np.float64)
        n = np.asarray(n_m, dtype=np.float64)
        classes = np.where(n > _WATER_NORTH_OF_m, 80, 30)
        return np.where(_inBox(e, n, _CANOPY_BOX_m), 10, classes).astype(np.uint8)

    def sceneBounds_deg(self, halfSize_m: float) -> tuple[float, float, float, float]:
        """Lon/lat bounds of the ENU square ``[-halfSize, halfSize]^2``.

        Args:
            halfSize_m: Half-size of the square, in meters.

        Returns:
            ``(minLon, minLat, maxLon, maxLat)``.
        """
        t = np.linspace(-halfSize_m, halfSize_m, 33)
        ones = np.full_like(t, halfSize_m)
        e_m = np.concatenate([t, ones, t, -ones])
        n_m = np.concatenate([-ones, t, ones, t])
        lat, lon, _ = self.frame.columnPointAtHeight(e_m, n_m, 0.0)
        return (float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))


_BUILDINGS = (
    SyntheticBuilding(-200.0, -150.0, -120.0, -50.0, 30.0, None),
    SyntheticBuilding(50.0, 50.0, 150.0, 90.0, 12.0, None),
    SyntheticBuilding(300.0, -300.0, 320.0, -280.0, None, 4),
)
_ROADS = (
    SyntheticRoad(-400.0, 200.0, 400.0, 200.0, isBridge=True, level=1, width_m=12.0),
    SyntheticRoad(-400.0, -450.0, 400.0, -450.0, isBridge=False, level=0, width_m=8.0),
)
_TREES = (SyntheticTree(0.0, -250.0, 8.0),)


def makeSyntheticPack(
    packDir: str | os.PathLike[str],
    center_deg: tuple[float, float] = (1.3, 103.8),
    terrain: TerrainKind = "plane",
    halfSize_m: float = 1000.0,
    resolution_m: float = 5.0,
) -> SyntheticScene:
    """Create a synthetic pack covering an ENU square around ``center_deg``.

    Args:
        packDir: Pack root to create.
        center_deg: ``(lat, lon)`` of the ENU origin.
        terrain: ``"plane"`` (linear, so bilinear sampling is exact) or ``"hill"``
            (Gaussian, 30 m high, sigma 150 m).
        halfSize_m: Half-size of the square pack area, in meters.
        resolution_m: Approximate raster pixel size, in meters.

    Returns:
        The ground truth for the pack.

    Raises:
        ValueError: For an unknown terrain kind.
    """
    if terrain not in ("plane", "hill"):
        raise ValueError(f"unknown terrain {terrain!r}; use 'plane' or 'hill'")
    frame = LocalFrame(*center_deg)
    scene = SyntheticScene(
        Path(packDir), frame, terrain, halfSize_m, buildings=_BUILDINGS, roads=_ROADS, trees=_TREES
    )
    packBounds = scene.sceneBounds_deg(halfSize_m)
    width = math.ceil((packBounds[2] - packBounds[0]) * _METERS_PER_DEG_LAT / resolution_m)
    height = math.ceil((packBounds[3] - packBounds[1]) * _METERS_PER_DEG_LAT / resolution_m)
    e_m, n_m = _pixelCentersEnu(frame, packBounds, (height, width))

    with tempfile.TemporaryDirectory() as staging:
        tmp = Path(staging)
        rasters: dict[str, tuple[npt.NDArray[np.generic], str]] = {
            "syntheticGround": (scene.terrainHeight_m(e_m, n_m).astype(np.float32), "ground.tif"),
            "syntheticCanopy": (scene.canopyHeight_m(e_m, n_m).astype(np.float32), "canopy.tif"),
            "syntheticLandCover": (scene.landClass(e_m, n_m), "landcover.tif"),
        }
        first = True
        for sourceId, (values, name) in rasters.items():
            _writeGeoTiff(tmp / name, values, packBounds)
            ingestFiles(packDir, sourceId, [tmp / name], bounds_deg=packBounds if first else None)
            first = False

        pad = 0.01
        geoidBounds = (
            packBounds[0] - pad,
            packBounds[1] - pad,
            packBounds[2] + pad,
            packBounds[3] + pad,
        )
        _writeGeoTiff(
            tmp / "geoid.tif", np.full((4, 4), scene.geoidUndulation_m, np.float32), geoidBounds
        )
        ingestFiles(packDir, "syntheticGeoid", [tmp / "geoid.tif"])

        writeGeoParquet(
            tmp / "buildings.parquet",
            np.array([_enuPolygon(frame, b) for b in _BUILDINGS], dtype=object),
            {
                "sourceFeatureId": [f"synB{i}" for i in range(len(_BUILDINGS))],
                "height_m": [b.height_m for b in _BUILDINGS],
                "numFloors": [b.numFloors for b in _BUILDINGS],
            },
        )
        ingestFiles(packDir, "syntheticBuildings", [tmp / "buildings.parquet"])

        writeGeoParquet(
            tmp / "transport.parquet",
            np.array([_enuLine(frame, r) for r in _ROADS], dtype=object),
            {
                "sourceFeatureId": [f"synR{i}" for i in range(len(_ROADS))],
                "kind": [r.kind for r in _ROADS],
                "isBridge": [r.isBridge for r in _ROADS],
                "level": [r.level for r in _ROADS],
                "width_m": [r.width_m for r in _ROADS],
            },
        )
        ingestFiles(packDir, "syntheticTransport", [tmp / "transport.parquet"])

        bridges = [r for r in _ROADS if r.isBridge]
        writeGeoParquet(
            tmp / "elevated.parquet",
            np.array([_enuDeckFootprint(frame, r) for r in bridges], dtype=object),
            {
                "sourceFeatureId": [f"synE{i}" for i in range(len(bridges))],
                "structureType": ["Flyover"] * len(bridges),
            },
        )
        ingestFiles(packDir, "syntheticElevatedStructures", [tmp / "elevated.parquet"])

        treeLat, treeLon, _ = frame.columnPointAtHeight(
            np.array([t.e_m for t in _TREES]), np.array([t.n_m for t in _TREES]), 0.0
        )
        writeGeoParquet(
            tmp / "trees.parquet",
            shapely.points(treeLon, treeLat),
            {
                "sourceFeatureId": [f"synT{i}" for i in range(len(_TREES))],
                "height_m": [t.height_m for t in _TREES],
            },
        )
        ingestFiles(packDir, "syntheticTrees", [tmp / "trees.parquet"])
    return scene


def _inBox(
    e: FloatArray, n: FloatArray, box: tuple[float, float, float, float]
) -> npt.NDArray[np.bool_]:
    return (e >= box[0]) & (e <= box[2]) & (n >= box[1]) & (n <= box[3])


def _pixelCentersEnu(
    frame: LocalFrame, bounds: tuple[float, float, float, float], shape: tuple[int, int]
) -> tuple[FloatArray, FloatArray]:
    minLon, minLat, maxLon, maxLat = bounds
    rows, cols = shape
    lon = minLon + (np.arange(cols) + 0.5) * (maxLon - minLon) / cols
    lat = maxLat - (np.arange(rows) + 0.5) * (maxLat - minLat) / rows
    lonMesh, latMesh = np.meshgrid(lon, lat)
    e_m, n_m, _ = frame.geodeticToEnu(latMesh, lonMesh, 0.0)
    return e_m, n_m


def _writeGeoTiff(
    path: Path, values: npt.NDArray[np.generic], bounds: tuple[float, float, float, float]
) -> None:
    rows, cols = values.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype=str(values.dtype),
        crs="EPSG:4326",
        transform=from_bounds(*bounds, cols, rows),
    ) as dst:
        dst.write(values, 1)


def _enuPolygon(frame: LocalFrame, b: SyntheticBuilding) -> shapely.Polygon:
    e_m = np.array([b.e0_m, b.e1_m, b.e1_m, b.e0_m])
    n_m = np.array([b.n0_m, b.n0_m, b.n1_m, b.n1_m])
    lat, lon, _ = frame.columnPointAtHeight(e_m, n_m, 0.0)
    return shapely.Polygon(np.column_stack([lon, lat]))


def _enuDeckFootprint(frame: LocalFrame, r: SyntheticRoad) -> shapely.Polygon:
    """Rectangle of the road's width around an axis-aligned (E-W) centerline."""
    half_m = 0.5 * r.width_m
    box = SyntheticBuilding(
        min(r.e0_m, r.e1_m), r.n0_m - half_m, max(r.e0_m, r.e1_m), r.n0_m + half_m, None, None
    )
    return _enuPolygon(frame, box)


def _enuLine(frame: LocalFrame, r: SyntheticRoad) -> shapely.LineString:
    t = np.linspace(0.0, 1.0, 9)
    lat, lon, _ = frame.columnPointAtHeight(
        r.e0_m + t * (r.e1_m - r.e0_m), r.n0_m + t * (r.n1_m - r.n0_m), 0.0
    )
    return shapely.LineString(np.column_stack([lon, lat]))
