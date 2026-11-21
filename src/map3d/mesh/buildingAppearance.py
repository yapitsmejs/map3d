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
canopies and link bridges, which have no storeys to show). Lofted
structures (:mod:`map3d.layers.lofts`) get a steel lattice.
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
        facadeVariants: Texture variant per building (all 0 if ``None``).
    """

    roofColors: ColorArray
    wallColors: ColorArray
    imageryDerived: npt.NDArray[np.bool_]
    sourceId: str | None = None
    attribution: str | None = None
    mapped: npt.NDArray[np.bool_] | None = None
    facadeStyles: npt.NDArray[np.uint8] | None = None
    facadeVariants: npt.NDArray[np.uint8] | None = None

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
        plain = ~self.imageryDerived
        for j in np.flatnonzero(plain):
            walls[j] = _NEUTRAL_WALLS[
                _hash(str(buildings.sourceFeatureIds[j])) % len(_NEUTRAL_WALLS)
            ]
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
            facadeVariants(buildings),
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

    WINDOWS = 0  # punched windows: the default
    GLASS = 1  # glass curtain wall
    PLAIN = 2  # blank cladding: canopies, sheds, structures without storeys
    LATTICE = 3  # steel lattice: lofted structures (the Supertrees)
    RIBBON = 4  # continuous window bands: offices
    RESIDENTIAL = 5  # windows with ledges or balconies: flats, HDB blocks
    RETAIL = 6  # large panes or blank mall walls: shops, malls
    INDUSTRIAL = 7  # profiled cladding, high strip windows: warehouses, factories
    PARKING = 8  # open decks between parapets: car parks
    HERITAGE = 9  # tall shuttered windows: shophouses, civic and religious buildings


FACADE_VARIANTS = 2  # texture variants per style (styles with one variant repeat it)
TALL_OFFICE_m = 80.0  # offices this tall are glazed
TALL_UNCLASSIFIED_m = 120.0  # unclassified towers this tall are glazed
SMALL_ROUND_m2 = 300.0  # round parts smaller than this without storeys are structures

_CLASS_STYLES = {
    **dict.fromkeys(
        (
            "apartments",
            "residential",
            "house",
            "detached",
            "semidetached_house",
            "terrace",
            "dormitory",
            "bungalow",
            "hut",
            "cabin",
            "static_caravan",
        ),
        FacadeStyle.RESIDENTIAL,
    ),
    **dict.fromkeys(("office", "commercial", "hotel", "bank", "service"), FacadeStyle.RIBBON),
    **dict.fromkeys(
        (
            "retail",
            "supermarket",
            "kiosk",
            "marketplace",
            "mall",
            "shop",
            "train_station",
            "transportation",
            "entertainment",
        ),
        FacadeStyle.RETAIL,
    ),
    **dict.fromkeys(
        (
            "industrial",
            "warehouse",
            "factory",
            "manufacture",
            "hangar",
            "storage_tank",
            "agricultural",
            "farm_auxiliary",
            "greenhouse",
            "sports_hall",
            "sports_centre",
            "stadium",
        ),
        FacadeStyle.INDUSTRIAL,
    ),
    **dict.fromkeys(("parking", "garage", "garages", "carport"), FacadeStyle.PARKING),
    **dict.fromkeys(
        (
            "religious",
            "church",
            "cathedral",
            "chapel",
            "mosque",
            "temple",
            "shrine",
            "synagogue",
            "civic",
            "government",
            "public",
            "museum",
            "school",
            "kindergarten",
            "college",
            "townhall",
            "courthouse",
            "fire_station",
        ),
        FacadeStyle.HERITAGE,
    ),
    **dict.fromkeys(
        ("roof", "shed", "outbuilding", "toilets", "transformer_tower", "bunker", "ruins"),
        FacadeStyle.PLAIN,
    ),
}
_HERITAGE_MAX_m = 30.0  # civic and religious buildings taller than this read as offices


def facadeStyles(buildings: BuildingSet) -> npt.NDArray[np.uint8]:
    """Facade style per building from its class, source, height, materials and shape.

    In order (later rules win): the class (offices ribbon windows, flats
    balconies, shops large panes, ...; HDB blocks are flats; tall offices
    glass, as are unclassified towers of :data:`TALL_UNCLASSIFIED_m`); low
    tiled-roof buildings shuttered windows (shophouses); mapped
    facade materials (glass, metal); structures without storeys (small round
    parts, thin raised parts) plain; lofts lattice; wheels plain.

    Args:
        buildings: Placed buildings.

    Returns:
        :class:`FacadeStyle` codes.
    """
    n = len(buildings)
    styles = np.full(n, FacadeStyle.WINDOWS, dtype=np.uint8)
    classes = (
        np.full(n, None, dtype=object)
        if buildings.buildingClass is None
        else buildings.buildingClass
    )
    for i, value in enumerate(classes):
        style = _CLASS_STYLES.get(str(value).strip().lower()) if value else None
        if style is not None:
            styles[i] = style
    styles[buildings.sourceIds == "sgHdbBuildings"] = FacadeStyle.RESIDENTIAL
    height = buildings.height_m
    office = styles == FacadeStyle.RIBBON
    styles[office & (height >= TALL_OFFICE_m)] = FacadeStyle.GLASS
    styles[(styles == FacadeStyle.HERITAGE) & (height > _HERITAGE_MAX_m)] = FacadeStyle.WINDOWS
    unclassified = np.array([not value for value in classes], dtype=bool)
    tower = unclassified & (styles == FacadeStyle.WINDOWS) & (height >= TALL_UNCLASSIFIED_m)
    styles[tower] = FacadeStyle.GLASS
    roofMaterial = np.array([str(m).strip().lower() for m in buildings.roofMaterial])
    tiled = (roofMaterial == "roof_tiles") & (height <= 20.0)
    styles[tiled & np.isin(styles, (FacadeStyle.WINDOWS, FacadeStyle.RETAIL))] = (
        FacadeStyle.HERITAGE
    )
    material = np.array([str(m).strip().lower() for m in buildings.facadeMaterial])
    styles[material == "metal"] = FacadeStyle.PLAIN
    styles[material == "glass"] = FacadeStyle.GLASS
    partHeight = height - buildings.minHeight_m
    styles[(buildings.minHeight_m > 0.0) & (partHeight < PLAIN_RAISED_HEIGHT_m)] = FacadeStyle.PLAIN
    styles[_smallRoundStructures(buildings)] = FacadeStyle.PLAIN
    for loft in buildings.lofts:
        styles[[*loft.members, *loft.absorbed]] = FacadeStyle.LATTICE
    for wheel in buildings.wheels:
        styles[list(wheel.members)] = FacadeStyle.PLAIN
    return styles


def facadeVariants(buildings: BuildingSet) -> npt.NDArray[np.uint8]:
    """A stable texture variant per building (from its feature id), for variety."""
    return np.array(
        [_hash(str(f)) % FACADE_VARIANTS for f in buildings.sourceFeatureIds], dtype=np.uint8
    )


def _smallRoundStructures(buildings: BuildingSet) -> npt.NDArray[np.bool_]:
    """Small round parts with no mapped storeys: columns, trunks, tanks, not rooms."""
    geoms = buildings.geometries
    if geoms.size == 0:
        return np.zeros(0, dtype=bool)
    lat = shapely.get_y(shapely.centroid(geoms))
    kx = 111_320.0 * np.cos(np.radians(lat))
    area = shapely.area(geoms) * kx * 110_574.0
    perimeter = shapely.length(shapely.get_exterior_ring(geoms)) * np.sqrt(kx * 110_574.0)
    roundness = np.where(perimeter > 0, 4.0 * np.pi * area / np.maximum(perimeter, 1e-9) ** 2, 0.0)
    floors = np.full(len(buildings), np.nan) if buildings.numFloors is None else buildings.numFloors
    result: npt.NDArray[np.bool_] = (
        (roundness >= 0.85) & (area < SMALL_ROUND_m2) & ~(np.isfinite(floors) & (floors > 0))
    )
    return result


def _hash(text: str) -> int:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=2).digest()[0]


# Plausible tints for walls with neither a mapped nor an imagery colour: cool
# and warm whites, creams, light greys.
_NEUTRAL_WALLS = np.array(
    [
        (240, 240, 244),
        (238, 233, 223),
        (229, 231, 233),
        (240, 231, 216),
        (226, 230, 238),
        (245, 242, 234),
        (232, 226, 218),
        (236, 238, 232),
    ],
    dtype=np.uint8,
)


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


def facadeTextureFor(style: int, variant: int = 0) -> bytes:
    """The repeating facade PNG of a :class:`FacadeStyle` and variant (one bay x one storey)."""
    return _styleTexture(int(style), int(variant) % FACADE_VARIANTS)


@functools.lru_cache(maxsize=32)
def _styleTexture(style: int, variant: int) -> bytes:
    if style == FacadeStyle.WINDOWS and variant == 0:
        return facadeTexture()
    if style == FacadeStyle.GLASS and variant == 0:
        return _glassTexture()
    if style == FacadeStyle.PLAIN:
        return _plainTexture()
    if style == FacadeStyle.LATTICE:
        return _latticeTexture()
    return encodePng(_DRAW[FacadeStyle(style)](variant))


def _tile(value: int = 240) -> npt.NDArray[np.uint8]:
    return np.full((_FACADE_TEXTURE_SIZE, _FACADE_TEXTURE_SIZE, 3), value, dtype=np.uint8)


def _drawWindows(variant: int) -> npt.NDArray[np.uint8]:
    """Variant 1: a pair of narrow windows per bay."""
    image = _tile(255)
    for c0, c1 in ((14, 42), (54, 82)):
        image[18:70, c0:c1] = 150
        image[22:66, c0 + 3 : c1 - 3] = 80
    image[70:74, 10:86] = 205  # sill
    image[84:96] = 188  # floor band
    return image


def _drawGlass(variant: int) -> npt.NDArray[np.uint8]:
    """Variant 1: darker, more reflective glazing without a centre mullion."""
    size = _FACADE_TEXTURE_SIZE
    shade = np.linspace(205, 150, size)[:, None]
    image = np.stack([shade * 0.82, shade * 0.9, shade], axis=-1).astype(np.uint8)
    image = np.broadcast_to(image, (size, size, 3)).copy()
    image[:, :2] = image[:, -2:] = 120
    image[-10:] = (120, 128, 138)
    return image


def _drawRibbon(variant: int) -> npt.NDArray[np.uint8]:
    """A continuous window band with mullions, over a solid spandrel."""
    image = _tile(238)
    top, bottom = (22, 66) if variant == 0 else (14, 72)
    image[top:bottom] = (92, 104, 118)
    step = 24 if variant == 0 else 32
    image[top:bottom, ::step] = 175
    image[top:bottom, 1::step] = 175
    image[bottom : bottom + 3] = 210
    return image


def _drawResidential(variant: int) -> npt.NDArray[np.uint8]:
    """Variant 0: a window with a ledge and an air-con shelf; 1: a balcony door and railing."""
    image = _tile(246)
    if variant == 0:
        image[20:60, 18:62] = 160
        image[23:57, 21:59] = 92
        image[23:57, 39:41] = 160  # casement divider
        image[60:66, 12:68] = 214  # ledge
        image[46:66, 70:88] = 222  # air-con shelf
        image[46:48, 70:88] = 196
    else:
        image[12:78, 14:82] = 150
        image[15:78, 17:79] = 96
        image[56:78, 10:86] = 236  # balcony railing
        image[56:78, 14:82:6] = 190
        image[56:58, 10:86] = 200
    image[86:96] = 220  # slab edge
    return image


def _drawRetail(variant: int) -> npt.NDArray[np.uint8]:
    """Variant 0: large shop panes; 1: a blank mall wall with a signage band."""
    image = _tile(236)
    if variant == 0:
        image[8:84, 6:90] = 140
        image[10:82, 8:88] = (98, 112, 122)
        image[10:82, 47:49] = 140
    else:
        image[...] = 242
        image[60:72] = 180  # signage band
        image[:, ::48] = 226  # panel joints
    return image


def _drawIndustrial(variant: int) -> npt.NDArray[np.uint8]:
    """Profiled cladding; variant 0 has a high strip window."""
    image = _tile(238)
    image[:, ::8] = 220
    image[:, 1::8] = 226
    if variant == 0:
        image[8:22] = (120, 132, 142)
        image[8:22, ::24] = 190
    image[92:96] = 212
    return image


def _drawParking(variant: int) -> npt.NDArray[np.uint8]:
    """An open deck between parapets; variant 1 screened with louvres."""
    image = _tile(234)
    image[34:84] = 72  # open deck
    if variant == 1:
        image[34:84:6] = 196  # louvres
        image[35:84:6] = 186
    image[34:84, :5] = image[34:84, -5:] = 214  # columns at the bay edges
    return image


def _drawHeritage(variant: int) -> npt.NDArray[np.uint8]:
    """A tall window with louvred shutters (variant 1: a pair of narrow ones), and a cornice."""
    image = _tile(250)
    windows = ((30, 66),) if variant == 0 else ((14, 42), (54, 82))
    for c0, c1 in windows:
        width = c1 - c0
        image[14:80, c0:c1] = 150
        image[17:80, c0 + 3 : c1 - 3] = 168  # shutters
        image[17:80:4, c0 + 3 : c1 - 3] = 150  # louvres
        image[17:80, c0 + width // 2 - 1 : c0 + width // 2 + 1] = 140
    image[80:84] = 214  # sill
    image[88:96] = 230  # cornice band
    image[88:90] = 206
    return image


_DRAW = {
    FacadeStyle.WINDOWS: _drawWindows,
    FacadeStyle.GLASS: _drawGlass,
    FacadeStyle.RIBBON: _drawRibbon,
    FacadeStyle.RESIDENTIAL: _drawResidential,
    FacadeStyle.RETAIL: _drawRetail,
    FacadeStyle.INDUSTRIAL: _drawIndustrial,
    FacadeStyle.PARKING: _drawParking,
    FacadeStyle.HERITAGE: _drawHeritage,
}


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


@functools.lru_cache(maxsize=1)
def _latticeTexture() -> bytes:
    """One diamond of a steel lattice (two crossing diagonals and a ring band)."""
    size = _FACADE_TEXTURE_SIZE
    image = np.full((size, size, 3), 232, dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    onDiagonal = (np.abs(xx - yy) < 3) | (np.abs(xx + yy - (size - 1)) < 3)
    image[onDiagonal] = 120
    image[:3] = image[-3:] = 150  # horizontal rings at the tile edges
    return encodePng(image)
