"""Imagery-derived building palettes and explicitly procedural facade materials.

Top-down imagery can inform roof colours, but it contains no reliable facade or
window observations. This module therefore combines conservative roof sampling
with coordinated wall colours and a repeatable, illustrative window/floor tile.
Low-resolution, dark, vegetated, or otherwise unreliable samples fall back to the
neutral conceptual palette.
"""

from __future__ import annotations

import colorsys
import functools
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import rasterio
import shapely

from map3d.layers.buildings import BuildingSet
from map3d.mesh.gltf import encodePng
from map3d.pack.reader import RasterWindow, readRasterWindow

if TYPE_CHECKING:
    from map3d.scene import Scene

__all__ = [
    "FACADE_BAY_WIDTH_m",
    "FACADE_FLOOR_HEIGHT_m",
    "NEUTRAL_ROOF",
    "NEUTRAL_WALL",
    "BuildingAppearance",
    "facadeTexture",
    "facadeUv",
]

FloatArray = npt.NDArray[np.float64]
ColorArray = npt.NDArray[np.uint8]

NEUTRAL_ROOF = (214, 214, 212)  # also the conceptual texture's roof colour
NEUTRAL_WALL = (240, 240, 244)  # walls a touch brighter than roofs, as in map renders
FACADE_BAY_WIDTH_m = 3.0
FACADE_FLOOR_HEIGHT_m = 3.0
_MAX_SAMPLES_PER_BUILDING = 256
_MIN_FOOTPRINT_PIXELS = 1.0
_MIN_SAMPLES = 2
_FACADE_TEXTURE_SIZE = 96


@dataclass(frozen=True, eq=False)
class BuildingAppearance:
    """Per-building conceptual colours and their imagery provenance.

    Attributes:
        roofColors: sRGB roof colours, shape ``(buildingCount, 3)``.
        wallColors: Coordinated sRGB facade tints, same shape.
        imageryDerived: Which buildings had reliable imagery samples.
        sourceId: Pack imagery source id, ``byoBuildingImagery``, or ``None``.
        attribution: Imagery attribution to add to the tileset, if any.
    """

    roofColors: ColorArray
    wallColors: ColorArray
    imageryDerived: npt.NDArray[np.bool_]
    sourceId: str | None = None
    attribution: str | None = None

    @property
    def coloredCount(self) -> int:
        """Number of buildings whose colours came from reliable imagery samples."""
        return int(self.imageryDerived.sum())

    @classmethod
    def neutral(cls, buildings: BuildingSet) -> BuildingAppearance:
        """Return the original conceptual colours for every building.

        Args:
            buildings: Placed building set.

        Returns:
            Neutral appearance with no imagery provenance.
        """
        count = len(buildings)
        return cls(
            np.tile(np.array(NEUTRAL_ROOF, dtype=np.uint8), (count, 1)),
            np.tile(np.array(NEUTRAL_WALL, dtype=np.uint8), (count, 1)),
            np.zeros(count, dtype=bool),
        )

    @classmethod
    def fromScene(
        cls, scene: Scene, buildingImageryPath: str | os.PathLike[str] | None = None
    ) -> BuildingAppearance:
        """Derive building colours from an orthophoto or the pack's imagery.

        The optional RGB GeoTIFF takes precedence over pack imagery. Sampling is
        capped per building and uses only pixel centers inside the footprint.
        Small, dark, vegetated, and undersampled footprints retain neutral colours.

        Args:
            scene: Built scene.
            buildingImageryPath: Optional higher-resolution RGB orthophoto.

        Returns:
            Imagery-derived appearance, with neutral fallback per building.

        Raises:
            ValueError: If a supplied file is absent or has fewer than three bands.
        """
        buildings = scene.models.buildings
        bounds = scene.geometry.footprintBounds_deg(margin_m=scene.geometry.gridSpacing_m)
        if buildingImageryPath is not None:
            path = Path(buildingImageryPath)
            if not path.is_file():
                raise ValueError(f"building imagery not found: {path}")
            with rasterio.open(path) as src:
                if src.count < 3:
                    raise ValueError("building imagery must be an RGB GeoTIFF (at least 3 bands)")
            sourceId = "byoBuildingImagery"
            bands = (
                readRasterWindow([path], bounds, sourceId, band=1),
                readRasterWindow([path], bounds, sourceId, band=2),
                readRasterWindow([path], bounds, sourceId, band=3),
            )
            attribution = f"Building imagery: {path.name} (user supplied)"
        else:
            pack = scene.pack
            entry = pack.primarySource("imagery") if pack is not None else None
            if pack is None or entry is None:
                return cls.neutral(buildings)
            sourceId = entry.sourceId
            try:
                bands = (
                    pack.readRaster(sourceId, bounds, band=1),
                    pack.readRaster(sourceId, bounds, band=2),
                    pack.readRaster(sourceId, bounds, band=3),
                )
            except ValueError:
                # Optional pack imagery may be declared but not cover this scene.
                return cls.neutral(buildings)
            attribution = entry.attribution.replace(
                "{accessed}", entry.accessedDate or "(unknown date)"
            )
        return _fromImagery(buildings, bands, sourceId, attribution)


def _fromImagery(
    buildings: BuildingSet,
    bands: tuple[RasterWindow, RasterWindow, RasterWindow],
    sourceId: str,
    attribution: str,
) -> BuildingAppearance:
    appearance = BuildingAppearance.neutral(buildings)
    if len(buildings) == 0:
        return BuildingAppearance(
            appearance.roofColors,
            appearance.wallColors,
            appearance.imageryDerived,
            sourceId,
            attribution,
        )
    window = bands[0]

    def toPixels(coordinates: FloatArray) -> FloatArray:
        col, row = window.pixelCoords(coordinates[:, 1], coordinates[:, 0])
        return np.column_stack([col, row])

    pixelGeometries = shapely.transform(buildings.geometries, toPixels)
    for i, geometry in enumerate(pixelGeometries):
        rgb = _footprintSamples(geometry, bands)
        areaPixels = float(shapely.area(geometry))
        if areaPixels < _MIN_FOOTPRINT_PIXELS or rgb.shape[0] < _MIN_SAMPLES:
            continue
        luminance = rgb @ np.array([0.2126, 0.7152, 0.0722])
        vegetation = (rgb[:, 1] > rgb[:, 0] * 1.15) & (rgb[:, 1] > rgb[:, 2] * 1.10)
        valid = (luminance >= 30.0) & ~vegetation
        if valid.sum() < _MIN_SAMPLES:
            continue
        roof = np.median(rgb[valid], axis=0)
        appearance.roofColors[i], appearance.wallColors[i] = _coordinatedColors(
            roof, str(buildings.sourceFeatureIds[i])
        )
        appearance.imageryDerived[i] = True
    return BuildingAppearance(
        appearance.roofColors,
        appearance.wallColors,
        appearance.imageryDerived,
        sourceId,
        attribution,
    )


def _footprintSamples(
    geometry: object, bands: tuple[RasterWindow, RasterWindow, RasterWindow]
) -> FloatArray:
    window = bands[0]
    minX, minY, maxX, maxY = shapely.bounds(geometry)
    height, width = window.values.shape
    c0, c1 = max(0, int(np.floor(minX))), min(width, int(np.ceil(maxX)))
    r0, r1 = max(0, int(np.floor(minY))), min(height, int(np.ceil(maxY)))
    if c0 >= c1 or r0 >= r1:
        return np.zeros((0, 3))
    stride = max(1, int(np.ceil(np.sqrt((c1 - c0) * (r1 - r0) / _MAX_SAMPLES_PER_BUILDING))))
    columns, rows = np.meshgrid(np.arange(c0, c1, stride), np.arange(r0, r1, stride))
    inside = shapely.contains_xy(geometry, columns + 0.5, rows + 0.5)
    rows, columns = rows[inside], columns[inside]
    if rows.size == 0:
        return np.zeros((0, 3))
    rgb = np.column_stack([band.values[rows, columns] for band in bands])
    finite = np.isfinite(rgb).all(axis=1)
    return rgb[finite]


def _coordinatedColors(rgb: FloatArray, featureId: str) -> tuple[ColorArray, ColorArray]:
    hue, lightness, saturation = colorsys.rgb_to_hls(*(np.asarray(rgb) / 255.0))
    roof = colorsys.hls_to_rgb(hue, float(np.clip(lightness, 0.25, 0.78)), min(saturation, 0.45))
    variation = hashlib.blake2b(featureId.encode("utf-8"), digest_size=1).digest()[0] / 255.0
    wallHue = (hue + (variation - 0.5) * 0.025) % 1.0
    wallLightness = float(
        np.clip(max(0.58, lightness + 0.27) + (variation - 0.5) * 0.04, 0.55, 0.85)
    )
    wall = colorsys.hls_to_rgb(wallHue, wallLightness, min(saturation * 0.55, 0.32))
    return (
        np.clip(np.round(np.array(roof) * 255.0), 0, 255).astype(np.uint8),
        np.clip(np.round(np.array(wall) * 255.0), 0, 255).astype(np.uint8),
    )


def facadeUv(length_m: FloatArray, height_m: FloatArray) -> npt.NDArray[np.float32]:
    """UVs for wall quads, repeating one window bay per approximately 3 metres.

    Args:
        length_m: Horizontal facade-segment lengths, shape ``(n,)``.
        height_m: Building heights, shape ``(n,)``.

    Returns:
        ``(n, 4, 2)`` UVs ordered bottom-start, bottom-end, top-end, top-start.
    """
    bays = np.maximum(1.0, np.rint(np.asarray(length_m) / FACADE_BAY_WIDTH_m))
    floors = np.maximum(1.0, np.rint(np.asarray(height_m) / FACADE_FLOOR_HEIGHT_m))
    uv = np.zeros((bays.size, 4, 2), dtype=np.float32)
    uv[:, 1, 0] = bays
    uv[:, 2, 0] = bays
    uv[:, 0, 1] = floors
    uv[:, 1, 1] = floors
    return uv


@functools.lru_cache(maxsize=1)
def facadeTexture() -> bytes:
    """Return a transparent-free PNG containing one illustrative facade bay.

    The near-white texture is multiplied by each building's wall colour in glTF.
    A framed, divided glass pane, sill and floor band add scale without claiming
    to reproduce observed windows.
    """
    image = np.full((_FACADE_TEXTURE_SIZE, _FACADE_TEXTURE_SIZE, 3), 255, dtype=np.uint8)
    image[18:70, 24:72] = 150  # frame
    image[22:66, 28:68] = 72  # glass
    image[22:66, 30:33] = 105  # restrained glass highlight
    image[42:46, 28:68] = 170  # horizontal mullion
    image[22:66, 46:50] = 170  # vertical mullion
    image[68:73, 22:74] = 190  # sill
    image[-4:, :] = 220  # floor band
    return encodePng(image)
