"""Building palettes (mapped, imagery-derived or neutral) and procedural facade materials.

Colour precedence per building:

1. **Mapped colours** (OSM ``building:colour`` / ``roof:colour`` via Overture's
   ``facade_color`` / ``roof_color``): a mapped facade colour sets the walls; a
   mapped roof colour sets the roof (and, without a mapped facade colour, the
   coordinated wall tint). Mapped colours keep their hue but are harmonised
   with the map-like style (saturation at most :data:`MAPPED_MAX_SATURATION`):
   OSM's ``roof:colour=red`` arrives as ``#FF3333`` and is meant as roof tiles,
   not signal red.
   A mapped **roof material** (tiles, grass, metal, glass, ...) without a
   mapped roof colour gives a typical colour for it.
2. **Imagery:** top-down imagery can inform roof colours, but it contains no
   reliable facade or window observations, so it is combined with coordinated
   wall colours. Low-resolution, dark, vegetated, or otherwise unreliable
   samples fall back to:
3. the **neutral** conceptual palette.

Walls use one of three repeatable, illustrative facade tiles
(:class:`FacadeStyle`): punched windows and floor bands (default), a glass
curtain wall (mapped ``facade_material=glass``), or plain panels (mapped
``metal``, and raised parts under :data:`PLAIN_RAISED_HEIGHT_m` tall such as
canopies and link bridges, which have no storeys to show).
"""

from __future__ import annotations

import colorsys
import functools
import hashlib
import os
from dataclasses import dataclass
from enum import IntEnum
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
    "MAPPED_MAX_SATURATION",
    "NEUTRAL_ROOF",
    "NEUTRAL_WALL",
    "BuildingAppearance",
    "FacadeStyle",
    "PLAIN_RAISED_HEIGHT_m",
    "facadeTexture",
    "facadeTextureFor",
    "facadeUv",
    "parseColor",
]

FloatArray = npt.NDArray[np.float64]
ColorArray = npt.NDArray[np.uint8]

NEUTRAL_ROOF = (214, 214, 212)  # also the conceptual texture's roof colour
NEUTRAL_WALL = (240, 240, 244)  # walls a touch brighter than roofs, as in map renders
FACADE_BAY_WIDTH_m = 3.0
MAPPED_MAX_SATURATION = 0.5  # HLS saturation cap for mapped colours
PLAIN_RAISED_HEIGHT_m = 6.0  # raised parts lower than this get plain facades
_MAPPED_LIGHTNESS = (0.15, 0.95)
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
        mapped: Which buildings have a mapped facade or roof colour (or roof material).
        facadeStyles: :class:`FacadeStyle` code per building (all windows if ``None``).
    """

    roofColors: ColorArray
    wallColors: ColorArray
    imageryDerived: npt.NDArray[np.bool_]
    sourceId: str | None = None
    attribution: str | None = None
    mapped: npt.NDArray[np.bool_] | None = None
    facadeStyles: npt.NDArray[np.uint8] | None = None

    @property
    def coloredCount(self) -> int:
        """Number of buildings whose colours came from reliable imagery samples."""
        return int(self.imageryDerived.sum())

    @property
    def mappedCount(self) -> int:
        """Number of buildings with a mapped facade or roof colour."""
        return 0 if self.mapped is None else int(self.mapped.sum())

    def styleCounts(self) -> dict[str, int]:
        """Number of buildings per facade style (``windows``, ``glass``, ``plain``)."""
        styles = (
            np.zeros(self.roofColors.shape[0], dtype=np.uint8)
            if self.facadeStyles is None
            else self.facadeStyles
        )
        return {s.name.lower(): int((styles == s).sum()) for s in FacadeStyle}

    def withMappedColors(self, buildings: BuildingSet) -> BuildingAppearance:
        """Apply the buildings' mapped facade and roof colours over these colours.

        Args:
            buildings: The buildings these colours belong to.

        Returns:
            A new appearance; unmapped buildings keep their colours.
        """
        roofs, walls = self.roofColors.copy(), self.wallColors.copy()
        mapped = np.zeros(len(buildings), dtype=bool)
        for i in range(len(buildings)):
            roof = _harmonized(parseColor(buildings.roofColor[i]))
            if roof is None:
                roof = _harmonized(_ROOF_MATERIAL_COLORS.get(str(buildings.roofMaterial[i])))
            facade = _harmonized(parseColor(buildings.facadeColor[i]))
            if roof is not None:
                roofs[i] = roof
                wallTint = _coordinatedColors(
                    roof.astype(np.float64), str(buildings.sourceFeatureIds[i])
                )
                walls[i] = wallTint[1]
            if facade is not None:
                walls[i] = facade
            mapped[i] = roof is not None or facade is not None
        return BuildingAppearance(
            roofs,
            walls,
            self.imageryDerived,
            self.sourceId,
            self.attribution,
            mapped,
            facadeStyles(buildings),
        )

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
            Mapped colours where present, else imagery-derived, else neutral.

        Raises:
            ValueError: If a supplied file is absent or has fewer than three bands.
        """
        return cls._fromImagerySources(scene, buildingImageryPath).withMappedColors(
            scene.models.buildings
        )

    @classmethod
    def _fromImagerySources(
        cls, scene: Scene, buildingImageryPath: str | os.PathLike[str] | None
    ) -> BuildingAppearance:
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


class FacadeStyle(IntEnum):
    """Which illustrative facade tile a building's walls use."""

    WINDOWS = 0
    GLASS = 1
    PLAIN = 2


def facadeStyles(buildings: BuildingSet) -> npt.NDArray[np.uint8]:
    """Facade style per building from its mapped material and shape.

    Args:
        buildings: Placed buildings.

    Returns:
        :class:`FacadeStyle` codes.
    """
    material = np.array([str(m).strip().lower() for m in buildings.facadeMaterial])
    partHeight = buildings.height_m - buildings.minHeight_m
    styles = np.full(len(buildings), FacadeStyle.WINDOWS, dtype=np.uint8)
    styles[material == "metal"] = FacadeStyle.PLAIN
    styles[(buildings.minHeight_m > 0.0) & (partHeight < PLAIN_RAISED_HEIGHT_m)] = FacadeStyle.PLAIN
    styles[material == "glass"] = FacadeStyle.GLASS
    return styles


# Typical colours of mapped roof materials (OSM roof:material), harmonised on use.
_ROOF_MATERIAL_COLORS = {
    "roof_tiles": np.array([178, 92, 66], dtype=np.uint8),
    "grass": np.array([138, 180, 102], dtype=np.uint8),
    "metal": np.array([170, 178, 186], dtype=np.uint8),
    "glass": np.array([176, 200, 216], dtype=np.uint8),
    "concrete": np.array([198, 196, 190], dtype=np.uint8),
    "wood": np.array([150, 116, 84], dtype=np.uint8),
    "slate": np.array([96, 102, 110], dtype=np.uint8),
    "thatch": np.array([186, 160, 110], dtype=np.uint8),
}

# CSS-style colour names seen in OSM building:colour / roof:colour.
_NAMED_COLORS = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "grey": (128, 128, 128),
    "gray": (128, 128, 128),
    "lightgrey": (211, 211, 211),
    "lightgray": (211, 211, 211),
    "darkgrey": (169, 169, 169),
    "darkgray": (169, 169, 169),
    "silver": (192, 192, 192),
    "red": (255, 0, 0),
    "darkred": (139, 0, 0),
    "maroon": (128, 0, 0),
    "brown": (165, 42, 42),
    "orange": (255, 165, 0),
    "yellow": (255, 255, 0),
    "beige": (245, 245, 220),
    "tan": (210, 180, 140),
    "cream": (255, 253, 208),
    "green": (0, 128, 0),
    "darkgreen": (0, 100, 0),
    "olive": (128, 128, 0),
    "teal": (0, 128, 128),
    "blue": (0, 0, 255),
    "lightblue": (173, 216, 230),
    "navy": (0, 0, 128),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
}


def parseColor(value: object) -> ColorArray | None:
    """An sRGB colour from ``#RRGGBB``, ``#RGB`` or a common colour name.

    Args:
        value: Mapped colour value (``None`` or unparseable values give ``None``).

    Returns:
        ``uint8`` RGB, or ``None``.
    """
    if not isinstance(value, str):
        return None
    text = value.strip().lower().replace(" ", "").replace("_", "")
    hexPart = text[1:] if text.startswith("#") else text
    if len(hexPart) in (3, 6) and all(c in "0123456789abcdef" for c in hexPart):
        if len(hexPart) == 3:
            hexPart = "".join(c * 2 for c in hexPart)
        return np.array([int(hexPart[k : k + 2], 16) for k in (0, 2, 4)], dtype=np.uint8)
    named = _NAMED_COLORS.get(text)
    return None if named is None else np.array(named, dtype=np.uint8)


def _harmonized(rgb: ColorArray | None) -> ColorArray | None:
    """A mapped colour with its saturation and lightness held to the map-like style."""
    if rgb is None:
        return None
    hue, lightness, saturation = colorsys.rgb_to_hls(*(rgb.astype(np.float64) / 255.0))
    lightness = float(np.clip(lightness, *_MAPPED_LIGHTNESS))
    out = colorsys.hls_to_rgb(hue, lightness, min(saturation, MAPPED_MAX_SATURATION))
    return np.clip(np.round(np.array(out) * 255.0), 0, 255).astype(np.uint8)


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


def facadeUv(
    length_m: FloatArray, height_m: FloatArray, heightEnd_m: FloatArray | None = None
) -> npt.NDArray[np.float32]:
    """UVs for wall quads, repeating one window bay per approximately 3 metres.

    Floors are counted up from the bottom edge, so where a wall's top slopes
    (a gable end) the window rows stay level and the top cuts through them.

    Args:
        length_m: Horizontal facade-segment lengths, shape ``(n,)``.
        height_m: Wall heights (at the segment start), shape ``(n,)``.
        heightEnd_m: Wall heights at the segment end, if the top slopes.

    Returns:
        ``(n, 4, 2)`` UVs ordered bottom-start, bottom-end, top-end, top-start.
    """
    h0 = np.asarray(height_m, dtype=np.float64)
    h1 = h0 if heightEnd_m is None else np.asarray(heightEnd_m, dtype=np.float64)
    tallest = np.maximum(np.maximum(h0, h1), 1e-9)
    bays = np.maximum(1.0, np.rint(np.asarray(length_m) / FACADE_BAY_WIDTH_m))
    floors = np.maximum(1.0, np.rint(tallest / FACADE_FLOOR_HEIGHT_m))
    uv = np.zeros((bays.size, 4, 2), dtype=np.float32)
    uv[:, 1, 0] = bays
    uv[:, 2, 0] = bays
    uv[:, 0, 1] = floors
    uv[:, 1, 1] = floors
    uv[:, 2, 1] = floors * (1.0 - h1 / tallest)
    uv[:, 3, 1] = floors * (1.0 - h0 / tallest)
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


def facadeTextureFor(style: int) -> bytes:
    """The repeating facade PNG of a :class:`FacadeStyle`."""
    return {
        FacadeStyle.WINDOWS: facadeTexture,
        FacadeStyle.GLASS: _glassTexture,
        FacadeStyle.PLAIN: _plainTexture,
    }[FacadeStyle(style)]()


@functools.lru_cache(maxsize=1)
def _glassTexture() -> bytes:
    """One illustrative curtain-wall bay: tall glass panes, thin mullions, a spandrel."""
    size = _FACADE_TEXTURE_SIZE
    shade = np.linspace(232, 196, size)[:, None]  # sky reflection: lighter at the top
    image = np.empty((size, size, 3), dtype=np.uint8)
    image[...] = np.stack([shade * 0.86, shade * 0.93, shade], axis=-1).astype(np.uint8)
    image[:, :3] = image[:, -3:] = 150  # mullions at the bay edges
    image[:, size // 2 - 1 : size // 2 + 1] = 172  # a light centre mullion
    image[-12:] = (156, 164, 172)  # spandrel at the floor slab
    image[-14:-12] = 140
    return encodePng(image)


@functools.lru_cache(maxsize=1)
def _plainTexture() -> bytes:
    """One plain cladding panel with faint joints."""
    size = _FACADE_TEXTURE_SIZE
    image = np.full((size, size, 3), 246, dtype=np.uint8)
    image[:, :1] = image[:1] = 228  # panel joints
    return encodePng(image)
