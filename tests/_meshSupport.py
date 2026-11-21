"""Shared helpers for the mesh tests (importable from tests: ``from _meshSupport import ...``)."""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

from map3d.pack.synthetic import SyntheticScene
from map3d.request import SceneRequest
from map3d.scene import Scene, buildScene


def sceneOver(synthetic: SyntheticScene, angle_deg: float = 0.0, height_m: float = 1000.0) -> Scene:
    """A 1 km wide, 5 m scene centred on a synthetic pack, its grid rotated ``angle_deg``."""
    h = math.radians(angle_deg)
    axes = [[math.cos(h), -math.sin(h), 0.0], [math.sin(h), math.cos(h), 0.0], [0.0, 0.0, 1.0]]
    centre = (synthetic.frame.lat0_deg, synthetic.frame.lon0_deg)
    return buildScene(SceneRequest(centre, 5.0, axes, 1000.0, height_m), synthetic.packDir)


def writeOrtho(
    scene: Scene, path: Path, colour: tuple[int, int, int], size: int = 100, bands: int = 3
) -> Path:
    """A uniform-colour orthophoto GeoTIFF (EPSG:4326) covering the scene with 50 m margin."""
    bounds = scene.geometry.footprintBounds_deg(margin_m=50.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=bands,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_bounds(*bounds, size, size),
    ) as dst:
        data = np.zeros((bands, size, size), dtype=np.uint8)
        data[:3] = np.array(colour, dtype=np.uint8)[:bands, None, None]
        dst.write(data)
    return path


def decodePng(data: bytes) -> np.ndarray:
    """PNG bytes -> ``(height, width, 3)`` pixels."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    with warnings.catch_warnings(), MemoryFile(data) as mem:
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with mem.open() as src:
            return np.moveaxis(src.read(), 0, -1)
