"""Scene geometry for 3D Tiles: terrain, buildings, decks, textures (SPEC section 7).

Everything is built from the scene's source models (not the grid), in
scene-centre ENU metres:

- **Terrain:** an RTIN mesh of the ground model over a tile, sampled on a
  ``(2^k+1)^2`` lattice in grid coordinates. Each vertex sits on its node
  column at the ground's ellipsoidal height.
- **Buildings:** LOD1 extrusions. Roofs are constrained-Delaunay triangulated
  at roof height; walls run from base to roof along the local vertical.
- **Decks:** slabs whose top follows the deck profile (ground interpolated
  between the bridge ends + level height), with an underside and side walls.
- **Textures:** one per tile, projected top-down. Imagery (pack source or BYO
  file) when available, otherwise a land-class palette rendered from the grid.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pyproj
import shapely
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile

from map3d.gridGeometry import GridGeometry
from map3d.layers.decks import DECK_THICKNESS_m
from map3d.mesh.gltf import MeshPart
from map3d.mesh.martini import checkGridSize, rtinErrors, rtinMesh
from map3d.pack.reader import Pack, RasterWindow, readRasterWindow

if TYPE_CHECKING:
    from map3d.scene import Scene

__all__ = [
    "DECK_COLOR",
    "MIN_MESH_CANOPY_m",
    "WALL_COLOR",
    "ImagerySource",
    "TileExtent",
    "buildingParts",
    "deckParts",
    "enuToGrid",
    "gridToEnu",
    "renderTexture",
    "terrainPart",
]

FloatArray = npt.NDArray[np.float64]

WALL_COLOR = (214, 210, 200)
MIN_MESH_CANOPY_m = 2.0  # lower vegetation (grass, shrubs) stays flat in the mesh
DECK_COLOR = (165, 160, 155)
_CLASS_PALETTE = {
    0: (90, 90, 90),
    10: (46, 110, 46),
    20: (140, 140, 70),
    30: (140, 180, 100),
    40: (215, 200, 110),
    50: (170, 120, 105),
    60: (180, 170, 150),
    70: (240, 240, 240),
    80: (50, 95, 170),
    90: (70, 140, 140),
    95: (30, 100, 80),
    100: (160, 170, 140),
    200: (200, 195, 185),
    210: (95, 95, 95),
    220: (130, 125, 120),
    230: (120, 100, 80),
}


@dataclass(frozen=True)
class TileExtent:
    """Axis-aligned rectangle in grid-frame metres."""

    x0: float
    x1: float
    y0: float
    y1: float

    def uv(self, gx: FloatArray, gy: FloatArray) -> npt.NDArray[np.float32]:
        """Top-down texture coordinates (v = 0 at the far +y edge)."""
        u = (gx - self.x0) / (self.x1 - self.x0)
        v = (self.y1 - gy) / (self.y1 - self.y0)
        return np.column_stack([u, v]).astype(np.float32)


def gridToEnu(
    geometry: GridGeometry, gx: FloatArray, gy: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Grid-frame horizontal coordinates -> ENU east/north."""
    a = geometry.gridAxes_ENU
    return gx * a[0, 0] + gy * a[1, 0], gx * a[0, 1] + gy * a[1, 1]


def enuToGrid(
    geometry: GridGeometry, e: FloatArray, n: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """ENU east/north -> grid-frame horizontal coordinates."""
    a = geometry.gridAxes_ENU
    return e * a[0, 0] + n * a[0, 1], e * a[1, 0] + n * a[1, 1]


def _pixelToGrid(
    geometry: GridGeometry, col: FloatArray, row: FloatArray
) -> tuple[FloatArray, FloatArray]:
    s = geometry.gridSpacing_m
    return col * s - 0.5 * geometry.nCols * s, row * s - 0.5 * geometry.nRows * s


def _upward(
    positions: FloatArray, tris: npt.NDArray[np.uint32], up: bool = True
) -> npt.NDArray[np.uint32]:
    """Flip triangles so their normals point up (or down)."""
    if tris.shape[0] == 0:
        return tris
    p = positions[tris]
    nz = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])[:, 2]
    flip = nz < 0 if up else nz > 0
    out = tris.copy()
    out[flip] = out[flip][:, ::-1]
    return out


def _columnUp(scene: Scene, e: FloatArray, n: FloatArray, orthometric: FloatArray) -> FloatArray:
    """ENU Up of column points at the given orthometric heights."""
    frame = scene.geometry.frame
    lat, lon, _ = frame.columnPointAtHeight(e, n, 0.0)
    hEll = orthometric + scene.models.ground.undulation_m(lat, lon)
    _, _, u = frame.columnPointAtHeight(e, n, hEll)
    return u


def terrainPart(
    scene: Scene,
    extent: TileExtent,
    gridSize: int = 129,
    maxError_m: float = 0.25,
    withCanopy: bool = True,
) -> MeshPart:
    """RTIN mesh of the ground (optionally plus vegetation) over a tile (textured).

    With ``withCanopy``, the surface is ground + canopy wherever the canopy is at
    least :data:`MIN_MESH_CANOPY_m` tall, so tree crowns become part of the draped
    terrain. The grid layers keep bare ground and canopy separate regardless.

    Args:
        scene: Built scene (uses its ground and canopy models).
        extent: Tile extent in grid metres.
        gridSize: RTIN lattice side (``2^k + 1``).
        maxError_m: RTIN tolerance.
        withCanopy: Raise the surface by the canopy (trees) where present.

    Returns:
        Upward-facing textured mesh part.
    """
    size = checkGridSize(gridSize)
    gx = np.linspace(extent.x0, extent.x1, size)
    gy = np.linspace(extent.y0, extent.y1, size)
    gxm, gym = np.meshgrid(gx, gy)
    e, n = gridToEnu(scene.geometry, gxm, gym)
    lat, lon, _ = scene.geometry.frame.columnPointAtHeight(e, n, 0.0)
    ground = scene.models.ground
    height = np.nan_to_num(ground.orthometricAt(lat, lon), nan=0.0) + ground.undulation_m(lat, lon)
    if withCanopy:
        s = scene.geometry.gridSpacing_m
        col = (gxm + 0.5 * scene.geometry.nCols * s) / s
        row = (gym + 0.5 * scene.geometry.nRows * s) / s
        canopy = scene.models.canopy.heightAt(lat, lon, col, row)
        height = height + np.where(canopy >= MIN_MESH_CANOPY_m, canopy, 0.0)
    verts, tris = rtinMesh(rtinErrors(np.ascontiguousarray(height)), maxError_m)
    vx, vy = verts[:, 0], verts[:, 1]
    ve, vn = e[vy, vx], n[vy, vx]
    _, _, vu = scene.geometry.frame.columnPointAtHeight(ve, vn, height[vy, vx])
    positions = np.column_stack([ve, vn, vu])
    return MeshPart(
        positions, _upward(positions, tris.astype(np.uint32)), extent.uv(gxm[vy, vx], gym[vy, vx])
    )


def buildingParts(
    scene: Scene, indices: npt.NDArray[np.int64], extent: TileExtent
) -> tuple[MeshPart, MeshPart]:
    """LOD1 extrusions of the selected buildings.

    Args:
        scene: Built scene.
        indices: Building indices (into ``scene.models.buildings``).
        extent: Tile extent (for roof UVs).

    Returns:
        ``(roofs, walls)``: textured roofs and vertex-coloured walls.
    """
    if indices.size == 0:
        return MeshPart.empty(withUv=True), MeshPart.empty(withColors=True)
    b = scene.models.buildings
    geoms = b.geometries[indices]
    centroids = shapely.centroid(geoms)
    undulation = scene.models.ground.undulation_m(
        shapely.get_y(centroids), shapely.get_x(centroids)
    )
    roofEll = b.roofOrthometric_m[indices] + undulation
    baseEll = b.baseOrthometric_m[indices] + undulation
    frame = scene.geometry.frame

    tris, owner = shapely.get_parts(
        shapely.constrained_delaunay_triangles(geoms), return_index=True
    )
    corners = shapely.get_coordinates(tris).reshape(-1, 4, 2)[:, :3].reshape(-1, 2)
    cornerOwner = np.repeat(owner, 3)
    e, n, u = frame.geodeticToEnu(corners[:, 1], corners[:, 0], roofEll[cornerOwner])
    roofPos = np.column_stack([e, n, u])
    roofTris = np.arange(roofPos.shape[0], dtype=np.uint32).reshape(-1, 3)
    gx, gy = enuToGrid(scene.geometry, e, n)
    roofs = MeshPart(roofPos, _upward(roofPos, roofTris), extent.uv(gx, gy))

    walls = _walls(frame, geoms, baseEll, roofEll, WALL_COLOR)
    return roofs, walls


def deckParts(
    scene: Scene, indices: npt.NDArray[np.int64], extent: TileExtent
) -> tuple[MeshPart, MeshPart]:
    """Slabs for the selected decks.

    Args:
        scene: Built scene.
        indices: Deck indices (into ``scene.models.decks``).
        extent: Tile extent (for deck-top UVs).

    Returns:
        ``(tops, sides)``: textured deck tops; coloured undersides and edges.
    """
    if indices.size == 0:
        return MeshPart.empty(withUv=True), MeshPart.empty(withColors=True)
    d = scene.models.decks
    shapes = d.pixelShapes[indices]
    tris, owner = shapely.get_parts(
        shapely.constrained_delaunay_triangles(shapes), return_index=True
    )
    px = shapely.get_coordinates(tris).reshape(-1, 4, 2)[:, :3].reshape(-1, 2)
    pxOwner = np.repeat(owner, 3)
    topH = _deckTop(d, indices, pxOwner, px)
    gx, gy = _pixelToGrid(scene.geometry, px[:, 0], px[:, 1])
    e, n = gridToEnu(scene.geometry, gx, gy)
    topPos = np.column_stack([e, n, _columnUp(scene, e, n, topH)])
    bottomPos = np.column_stack([e, n, _columnUp(scene, e, n, topH - DECK_THICKNESS_m)])
    triIdx = np.arange(topPos.shape[0], dtype=np.uint32).reshape(-1, 3)
    tops = MeshPart(topPos, _upward(topPos, triIdx), extent.uv(gx, gy))
    underside = MeshPart(
        bottomPos,
        _upward(bottomPos, triIdx, up=False),
        colors=_colors(bottomPos.shape[0], DECK_COLOR),
    )

    rings, ringOwner = shapely.get_rings(shapes, return_index=True)
    coords, coordRing = shapely.get_coordinates(rings, return_index=True)
    a, b = _ringSegments(coordRing)
    segOwner = ringOwner[coordRing[a]]
    ends = np.concatenate([coords[a], coords[b]])
    endOwner = np.concatenate([segOwner, segOwner])
    endTop = _deckTop(d, indices, endOwner, ends)
    sgx, sgy = _pixelToGrid(scene.geometry, ends[:, 0], ends[:, 1])
    se, sn = gridToEnu(scene.geometry, sgx, sgy)
    upTop = _columnUp(scene, se, sn, endTop)
    upBottom = _columnUp(scene, se, sn, endTop - DECK_THICKNESS_m)
    m = a.size
    sides = _quads(
        np.column_stack([se[:m], sn[:m], upBottom[:m]]),
        np.column_stack([se[m:], sn[m:], upBottom[m:]]),
        np.column_stack([se[m:], sn[m:], upTop[m:]]),
        np.column_stack([se[:m], sn[:m], upTop[:m]]),
        DECK_COLOR,
    )
    return tops, MeshPart.concatenate([underside, sides])


@dataclass(frozen=True, eq=False)
class ImagerySource:
    """RGB imagery as three raster windows."""

    bands: tuple[RasterWindow, RasterWindow, RasterWindow]

    @classmethod
    def fromPack(
        cls, pack: Pack | None, bounds_deg: tuple[float, float, float, float]
    ) -> ImagerySource | None:
        """Imagery from the pack's ``imagery`` role, or ``None`` if absent."""
        entry = pack.primarySource("imagery") if pack is not None else None
        if pack is None or entry is None:
            return None
        r, g, b = (pack.readRaster(entry.sourceId, bounds_deg, band=k) for k in (1, 2, 3))
        return cls((r, g, b))

    @classmethod
    def fromFile(
        cls, path: str | os.PathLike[str], bounds_deg: tuple[float, float, float, float]
    ) -> ImagerySource:
        """Imagery from a bring-your-own RGB GeoTIFF (any CRS)."""
        r, g, b = (
            readRasterWindow([Path(path)], bounds_deg, "byoImagery", band=k) for k in (1, 2, 3)
        )
        return cls((r, g, b))

    def sampleRgb(self, lat_deg: FloatArray, lon_deg: FloatArray) -> npt.NDArray[np.uint8]:
        """Bilinear RGB at positions; black where outside."""
        channels = [np.nan_to_num(w.sample(lat_deg, lon_deg), nan=0.0) for w in self.bands]
        return np.clip(np.round(np.stack(channels, axis=-1)), 0, 255).astype(np.uint8)


def renderTexture(
    scene: Scene, imagery: ImagerySource | None, extent: TileExtent, size: int = 256
) -> bytes:
    """Render a tile's top-down texture as JPEG bytes.

    Args:
        scene: Built scene.
        imagery: Imagery to drape, or ``None`` for the land-class palette.
        extent: Tile extent.
        size: Texture side in pixels.

    Returns:
        JPEG bytes.
    """
    t = (np.arange(size) + 0.5) / size
    gx = extent.x0 + t * (extent.x1 - extent.x0)
    gy = extent.y1 - t * (extent.y1 - extent.y0)
    gxm, gym = np.meshgrid(gx, gy)
    if imagery is not None:
        e, n = gridToEnu(scene.geometry, gxm, gym)
        toGeo = _orthoToGeographic(scene.geometry.orthoCrs().to_wkt())
        lon, lat = toGeo.transform(e, n)
        rgb = imagery.sampleRgb(np.asarray(lat), np.asarray(lon))
    else:
        rgb = _classTexture(scene, gxm, gym)
    return _jpeg(rgb)


def _classTexture(scene: Scene, gx: FloatArray, gy: FloatArray) -> npt.NDArray[np.uint8]:
    g = scene.geometry
    s = g.gridSpacing_m
    col = np.clip(((gx + 0.5 * g.nCols * s) / s).astype(np.int64), 0, g.nCols - 1)
    row = np.clip(((gy + 0.5 * g.nRows * s) / s).astype(np.int64), 0, g.nRows - 1)
    classes = np.asarray(scene.layers["landClass"])[row, col]
    lut = np.zeros((256, 3), dtype=np.uint8)
    for code, colour in _CLASS_PALETTE.items():
        lut[code] = colour
    rgb: npt.NDArray[np.uint8] = lut[classes]
    return rgb


def _jpeg(rgb: npt.NDArray[np.uint8]) -> bytes:
    height, width, _ = rgb.shape
    with warnings.catch_warnings(), MemoryFile() as mem:
        warnings.simplefilter("ignore", NotGeoreferencedWarning)  # textures are not georeferenced
        with mem.open(
            driver="JPEG", width=width, height=height, count=3, dtype="uint8", quality=85
        ) as dst:
            dst.write(np.moveaxis(rgb, -1, 0))
        return bytes(mem.read())


_TRANSFORMERS: dict[str, pyproj.Transformer] = {}


def _orthoToGeographic(wkt: str) -> pyproj.Transformer:
    if wkt not in _TRANSFORMERS:
        _TRANSFORMERS[wkt] = pyproj.Transformer.from_crs(
            pyproj.CRS.from_wkt(wkt), "EPSG:4326", always_xy=True
        )
    return _TRANSFORMERS[wkt]


def _deckTop(
    decks: Any, indices: npt.NDArray[np.int64], owner: npt.NDArray[np.int64], px: FloatArray
) -> FloatArray:
    deck = indices[owner]
    top = np.empty(owner.size)
    for k in np.unique(deck):
        mine = deck == k
        line = decks.centerlines[k]
        if line is None:
            ground = np.full(int(mine.sum()), decks.groundStart_m[k])
        else:
            frac = shapely.line_locate_point(line, shapely.points(px[mine]), normalized=True)
            ground = decks.groundStart_m[k] + frac * (decks.groundEnd_m[k] - decks.groundStart_m[k])
        top[mine] = ground + decks.levelHeight_m[k]
    return top


def _ringSegments(
    coordRing: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Indices ``(a, b)`` of consecutive coordinates within each (closed) ring."""
    a = np.flatnonzero(coordRing[:-1] == coordRing[1:])
    return a, a + 1


def _walls(
    frame: Any,
    geoms: npt.NDArray[np.object_],
    baseEll: FloatArray,
    roofEll: FloatArray,
    colour: tuple[int, int, int],
) -> MeshPart:
    rings, ringOwner = shapely.get_rings(geoms, return_index=True)
    coords, coordRing = shapely.get_coordinates(rings, return_index=True)
    a, b = _ringSegments(coordRing)
    owner = ringOwner[coordRing[a]]

    def enu(idx: npt.NDArray[np.int64], h: FloatArray) -> FloatArray:
        e, n, u = frame.geodeticToEnu(coords[idx, 1], coords[idx, 0], h)
        return np.column_stack([e, n, u])

    return _quads(
        enu(a, baseEll[owner]),
        enu(b, baseEll[owner]),
        enu(b, roofEll[owner]),
        enu(a, roofEll[owner]),
        colour,
    )


def _quads(
    p0: FloatArray, p1: FloatArray, p2: FloatArray, p3: FloatArray, colour: tuple[int, int, int]
) -> MeshPart:
    m = p0.shape[0]
    if m == 0:
        return MeshPart.empty(withColors=True)
    positions = np.stack([p0, p1, p2, p3], axis=1).reshape(-1, 3)
    base = (np.arange(m, dtype=np.uint32) * 4)[:, None]
    tris = np.concatenate(
        [base + np.array([0, 1, 2], dtype=np.uint32), base + np.array([0, 2, 3], dtype=np.uint32)]
    )
    return MeshPart(positions, tris.astype(np.uint32), colors=_colors(positions.shape[0], colour))


def _colors(count: int, colour: tuple[int, int, int]) -> npt.NDArray[np.uint8]:
    return np.tile(np.array(colour, dtype=np.uint8), (count, 1))
