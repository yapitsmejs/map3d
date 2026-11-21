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
- **Textures:** one conceptual land-class palette per tile, projected top-down.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pyproj
import shapely

from map3d.layers.decks import DECK_THICKNESS_m, deckTopAboveGround_m
from map3d.layers.landCover import (
    CLASS_BUILDING,
    CLASS_DECK,
    CLASS_RAIL,
    CLASS_ROAD,
    remapClasses,
)
from map3d.mesh.buildingAppearance import NEUTRAL_ROOF, BuildingAppearance, facadeUv
from map3d.mesh.gltf import MeshPart, encodePng
from map3d.mesh.martini import checkGridSize, rtinErrors, rtinMesh
from map3d.mesh.trees import MIN_TREE_HEIGHT_m
from map3d.pack.reader import RasterWindow

if TYPE_CHECKING:
    from map3d.scene import Scene

__all__ = [
    "CONCEPT_COLORS",
    "PIER_COLOR",
    "PIER_SPACING_m",
    "TileExtent",
    "buildingParts",
    "deckParts",
    "renderTexture",
    "terrainPart",
]

FloatArray = npt.NDArray[np.float64]

DECK_COLOR = (172, 174, 180)
PIER_COLOR = (150, 148, 145)
PIER_SPACING_m = 35.0  # pier spacing along a deck centerline
PIER_END_MARGIN_m = 15.0  # no piers within the tapered ends
PIER_CLEARANCE_m = 3.0  # only where the slab clears the ground by this
PIER_SIZE_m = 1.2  # square pier footprint side
PIER_EMBED_m = 0.15  # pier ends embed slightly into slab and ground

# Conceptual (map-like) style: flat pastel colours, shaded by the renderer.
CONCEPT_COLORS: dict[str, tuple[int, int, int]] = {
    "grass": (205, 229, 172),
    "treeCover": (186, 219, 154),  # WorldCover trees where the canopy is under 2 m
    "underTrees": (160, 202, 128),  # ground under canopy >= 2 m (trees stand on it)
    "builtUp": (233, 230, 223),
    "bare": (232, 223, 198),
    "snow": (246, 246, 246),
    "water": (146, 200, 235),
    "wetland": (178, 214, 188),
    "road": (178, 180, 186),
    "rail": (162, 154, 148),
    "deck": (190, 192, 198),
    "roof": NEUTRAL_ROOF,
    "unknown": (228, 228, 222),
}
_LAND_COVER_STYLE = {
    10: "treeCover",
    20: "grass",
    30: "grass",
    40: "grass",
    50: "builtUp",
    60: "bare",
    70: "snow",
    80: "water",
    90: "wetland",
    95: "treeCover",
    100: "grass",
}
_OVERLAY_STYLE = {
    CLASS_BUILDING: "roof",
    CLASS_ROAD: "road",
    CLASS_DECK: "deck",
    CLASS_RAIL: "rail",
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
) -> MeshPart:
    """RTIN mesh of the ground over a tile (textured).

    The surface is the true ground; vegetation is drawn as individual trees
    (:mod:`map3d.mesh.trees`).

    Args:
        scene: Built scene (uses its ground model).
        extent: Tile extent in grid metres.
        gridSize: RTIN lattice side (``2^k + 1``).
        maxError_m: RTIN tolerance.

    Returns:
        Upward-facing textured mesh part.
    """
    size = checkGridSize(gridSize)
    gx = np.linspace(extent.x0, extent.x1, size)
    gy = np.linspace(extent.y0, extent.y1, size)
    gxm, gym = np.meshgrid(gx, gy)
    e, n = scene.geometry.gridToEnu(gxm, gym)
    lat, lon, _ = scene.geometry.frame.columnPointAtHeight(e, n, 0.0)
    ground = scene.models.ground
    height = np.nan_to_num(ground.orthometricAt(lat, lon), nan=0.0) + ground.undulation_m(lat, lon)
    verts, tris = rtinMesh(rtinErrors(np.ascontiguousarray(height)), maxError_m)
    vx, vy = verts[:, 0], verts[:, 1]
    ve, vn = e[vy, vx], n[vy, vx]
    _, _, vu = scene.geometry.frame.columnPointAtHeight(ve, vn, height[vy, vx])
    positions = np.column_stack([ve, vn, vu])
    return MeshPart(
        positions, _upward(positions, tris.astype(np.uint32)), extent.uv(gxm[vy, vx], gym[vy, vx])
    )


def buildingParts(
    scene: Scene,
    indices: npt.NDArray[np.int64],
    extent: TileExtent,
    appearance: BuildingAppearance,
) -> tuple[MeshPart, MeshPart]:
    """LOD1 extrusions of the selected buildings.

    Args:
        scene: Built scene.
        indices: Building indices (into ``scene.models.buildings``).
        extent: Tile extent (for roof UVs).
        appearance: Per-building facade tints (imagery-derived or neutral).

    Returns:
        ``(roofs, walls)``: textured roofs and textured, vertex-tinted walls.
        Wall UVs repeat procedural window bays and floor bands at physical scale.
    """
    if indices.size == 0:
        return MeshPart.empty(withUv=True), MeshPart.empty(withUv=True, withColors=True)
    b = scene.models.buildings
    geoms = b.geometries[indices]
    centroids = shapely.centroid(geoms)
    undulation = scene.models.ground.undulation_m(
        shapely.get_y(centroids), shapely.get_x(centroids)
    )
    roofEll = b.roofOrthometric_m[indices] + undulation
    baseEll = b.baseOrthometric_m[indices] + undulation
    frame = scene.geometry.frame

    corners, cornerOwner = _triCorners(geoms)
    e, n, u = frame.geodeticToEnu(corners[:, 1], corners[:, 0], roofEll[cornerOwner])
    roofPos = np.column_stack([e, n, u])
    roofTris = np.arange(roofPos.shape[0], dtype=np.uint32).reshape(-1, 3)
    gx, gy = scene.geometry.enuToGrid(e, n)
    roofs = MeshPart(roofPos, _upward(roofPos, roofTris), extent.uv(gx, gy))

    walls = _walls(frame, geoms, baseEll, roofEll, appearance.wallColors[indices])
    return roofs, walls


def deckParts(
    scene: Scene, indices: npt.NDArray[np.int64], extent: TileExtent
) -> tuple[MeshPart, MeshPart]:
    """Slabs for the selected decks, clipped to the tile.

    Each deck slab is clipped to ``extent`` so its UVs stay inside the
    tile's texture: an unclipped slab longer than the tile would sample the
    texture outside ``[0, 1]``, and with CLAMP_TO_EDGE wrapping its
    out-of-tile portion would smear one edge pixel across the bridge.

    Args:
        scene: Built scene.
        indices: Deck indices (into ``scene.models.decks``).
        extent: Tile extent (for deck-top UVs and the clip).

    Returns:
        ``(tops, sides)``: textured deck tops; coloured undersides and edges.
    """
    if indices.size == 0:
        return MeshPart.empty(withUv=True), MeshPart.empty(withColors=True)
    d = scene.models.decks
    box = extentBoxPx(scene, extent)
    clipped = []
    kept = []
    for k in indices:
        piece = shapely.intersection(d.pixelShapes[k], box)
        if not piece.is_empty and shapely.area(piece) > 0.0:
            clipped.append(piece)
            kept.append(k)
    if not clipped:
        return MeshPart.empty(withUv=True), MeshPart.empty(withColors=True)
    indices = np.asarray(kept, dtype=np.int64)
    shapes = np.array(clipped, dtype=object)
    px, pxOwner = _triCorners(shapes)
    topH = _deckTop(scene, d, indices, pxOwner, px)
    gx, gy = scene.geometry.pixelToGrid(px[:, 0], px[:, 1])
    e, n = scene.geometry.gridToEnu(gx, gy)
    topPos = np.column_stack([e, n, _columnUp(scene, e, n, topH)])
    bottomPos = np.column_stack([e, n, _columnUp(scene, e, n, topH - DECK_THICKNESS_m)])
    triIdx = np.arange(topPos.shape[0], dtype=np.uint32).reshape(-1, 3)
    tops = MeshPart(topPos, _upward(topPos, triIdx), extent.uv(gx, gy))
    underside = MeshPart(
        bottomPos,
        _upward(bottomPos, triIdx, up=False),
        colors=_colors(bottomPos.shape[0], DECK_COLOR),
    )

    start, end, segOwner = _ringEdges(shapes)
    ends = np.concatenate([start, end])
    endOwner = np.concatenate([segOwner, segOwner])
    endTop = _deckTop(scene, d, indices, endOwner, ends)
    sgx, sgy = scene.geometry.pixelToGrid(ends[:, 0], ends[:, 1])
    se, sn = scene.geometry.gridToEnu(sgx, sgy)
    upTop = _columnUp(scene, se, sn, endTop)
    upBottom = _columnUp(scene, se, sn, endTop - DECK_THICKNESS_m)
    m = segOwner.size
    # Segments lying on the clip box are tile cuts, not real slab edges:
    # drawing their sides would raise a wall where the deck continues into
    # the neighbouring tile.
    cut = _onClipEdge(start, end, box)
    keep = ~np.concatenate([cut, cut])
    sides = _quads(
        np.column_stack([se[:m], sn[:m], upBottom[:m]])[keep[:m]],
        np.column_stack([se[m:], sn[m:], upBottom[m:]])[keep[m:]],
        np.column_stack([se[m:], sn[m:], upTop[m:]])[keep[m:]],
        np.column_stack([se[:m], sn[:m], upTop[:m]])[keep[:m]],
        DECK_COLOR,
    )
    return tops, MeshPart.concatenate([underside, sides, _piers(scene, d, indices, extent)])


def extentBoxPx(scene: Scene, extent: TileExtent) -> Any:
    """The tile extent as a box in grid pixel coordinates."""
    col, row = scene.geometry.gridToPixel(
        np.array([extent.x0, extent.x1]), np.array([extent.y0, extent.y1])
    )
    return shapely.box(col[0], row[0], col[1], row[1])


def _onClipEdge(a: FloatArray, b: FloatArray, box: Any) -> npt.NDArray[np.bool_]:
    """Which ring segments lie along the clip box edge (tile cuts)."""
    x0, y0, x1, y1 = shapely.bounds(box)
    tol = 1e-6
    onEdge = (
        (np.isclose(a[:, 0], x0, tol) & np.isclose(b[:, 0], x0, tol))
        | (np.isclose(a[:, 0], x1, tol) & np.isclose(b[:, 0], x1, tol))
        | (np.isclose(a[:, 1], y0, tol) & np.isclose(b[:, 1], y0, tol))
        | (np.isclose(a[:, 1], y1, tol) & np.isclose(b[:, 1], y1, tol))
    )
    return np.asarray(onEdge, dtype=bool)


def _piers(scene: Scene, d: Any, indices: npt.NDArray[np.int64], extent: TileExtent) -> MeshPart:
    """Box piers grounding centreline decks over clear ground.

    Piers stand every :data:`PIER_SPACING_m` along the centerline (never
    within the end tapers), only where the slab's underside clears the ground
    by more than :data:`PIER_CLEARANCE_m`.

    Piers belong to the tile their position falls in, so a deck spanning
    several tiles does not duplicate them.
    """
    parts: list[MeshPart] = []
    geometry = scene.geometry
    for k in indices:
        line = d.centerlines[k]
        if line is None:
            continue
        length_m = shapely.length(line) * geometry.gridSpacing_m
        positions_m = np.arange(PIER_END_MARGIN_m, length_m - PIER_END_MARGIN_m, PIER_SPACING_m)
        if positions_m.size == 0:
            continue
        points = shapely.line_interpolate_point(line, positions_m / length_m, normalized=True)
        px = np.column_stack([shapely.get_x(points), shapely.get_y(points)])
        topH = _deckTop(scene, d, np.array([k]), np.zeros(px.shape[0], dtype=np.int64), px)
        gx, gy = geometry.pixelToGrid(px[:, 0], px[:, 1])
        e, n = geometry.gridToEnu(gx, gy)
        lat, lon, _ = geometry.frame.columnPointAtHeight(e, n, 0.0)
        groundH = np.asarray(scene.models.ground.orthometricAt(lat, lon))
        keep = (topH - DECK_THICKNESS_m - groundH) > PIER_CLEARANCE_m
        if not keep.any():
            continue
        e, n, topH, groundH = e[keep], n[keep], topH[keep], groundH[keep]
        inTile = (e >= extent.x0) & (e <= extent.x1) & (n >= extent.y0) & (n <= extent.y1)
        e, n, topH, groundH = e[inTile], n[inTile], topH[inTile], groundH[inTile]
        if e.size == 0:
            continue
        upPierTop = _columnUp(scene, e, n, topH - DECK_THICKNESS_m) + PIER_EMBED_m
        upGround = _columnUp(scene, e, n, groundH) - PIER_EMBED_m
        half = 0.5 * PIER_SIZE_m
        corners = np.array([[half, half], [-half, half], [-half, -half], [half, -half]])
        for i in range(4):
            j = (i + 1) % 4
            p0 = np.column_stack([e + corners[i, 0], n + corners[i, 1], upGround])
            p1 = np.column_stack([e + corners[j, 0], n + corners[j, 1], upGround])
            p2 = np.column_stack([e + corners[j, 0], n + corners[j, 1], upPierTop])
            p3 = np.column_stack([e + corners[i, 0], n + corners[i, 1], upPierTop])
            parts.append(_quads(p0, p1, p2, p3, PIER_COLOR))
    return MeshPart.concatenate(parts)


def renderTexture(
    scene: Scene,
    extent: TileExtent,
    appearance: BuildingAppearance,
    size: int = 512,
) -> bytes:
    """Render a tile's conceptual top-down texture as lossless PNG.

    - **Land cover:** the source raster's classes coloured and sampled
      bilinearly, so its coarse pixels blend softly (grid classes if absent).
    - **Crisp overlays** from the grid's ``landClass`` layer: roads, rail, decks
      and building roofs.
    - **Under trees:** a soft green wherever the canopy is at least
      :data:`~map3d.mesh.trees.MIN_TREE_HEIGHT_m` (except roads, rail, decks and roofs), so gaps
      between crowns and distant views (no tree instances) still read as wooded.

    Args:
        scene: Built scene.
        extent: Tile extent.
        appearance: Per-building roof colours (imagery-derived or neutral).
        size: Texture side in pixels.

    Returns:
        PNG bytes.
    """
    t = (np.arange(size) + 0.5) / size
    gx = extent.x0 + t * (extent.x1 - extent.x0)
    gy = extent.y1 - t * (extent.y1 - extent.y0)
    gxm, gym = np.meshgrid(gx, gy)
    e, n = scene.geometry.gridToEnu(gxm, gym)
    toGeo = _orthoToGeographic(scene.geometry.orthoCrs().to_wkt())
    lon, lat = (np.asarray(v) for v in toGeo.transform(e, n))
    return encodePng(_conceptTexture(scene, extent, size, lat, lon, appearance))


_SUPERSAMPLE = 2  # crisp overlays are sampled 2x per axis and averaged (anti-aliased edges)


def _styleLut(table: dict[int, str]) -> npt.NDArray[np.float64]:
    lut = np.tile(np.array(CONCEPT_COLORS["unknown"], dtype=np.float64), (256, 1))
    for code, name in table.items():
        lut[code] = CONCEPT_COLORS[name]
    return lut


def _softLandCover(scene: Scene, lat: FloatArray, lon: FloatArray) -> FloatArray:
    """Land-cover colours sampled bilinearly from the source raster (NaN outside / absent)."""
    rgb = np.full((*lat.shape, 3), np.nan)
    window = scene.models.landCover.window if scene.models is not None else None
    if window is None:
        return rgb
    landLut = _styleLut(_LAND_COVER_STYLE)
    valid = np.isfinite(window.values)
    codes = remapClasses(np.where(valid, window.values, 0).astype(np.int64), window.sourceId)
    for k in range(3):
        band = RasterWindow(
            np.where(valid, landLut[codes, k], np.nan),
            window.transform,
            window.crs,
            window.sourceId,
        )
        rgb[..., k] = band.sample(lat, lon)
    return rgb


def _conceptTexture(
    scene: Scene,
    extent: TileExtent,
    size: int,
    lat: FloatArray,
    lon: FloatArray,
    appearance: BuildingAppearance,
) -> npt.NDArray[np.uint8]:
    k = _SUPERSAMPLE
    t = (np.arange(size * k) + 0.5) / (size * k)
    gx, gy = np.meshgrid(
        extent.x0 + t * (extent.x1 - extent.x0), extent.y1 - t * (extent.y1 - extent.y0)
    )
    col, row = scene.geometry.nodeIndex(gx, gy)
    classes = np.asarray(scene.layers["landClass"])[row, col]
    canopy = np.nan_to_num(np.asarray(scene.layers["canopyHeight_m"])[row, col], nan=0.0)

    # Land cover is soft anyway: sample it at the texture resolution, then repeat.
    soft = np.repeat(np.repeat(_softLandCover(scene, lat, lon), k, axis=0), k, axis=1)
    rgb = np.where(np.isfinite(soft), soft, _styleLut(_LAND_COVER_STYLE)[classes])
    overlay = np.isin(classes, list(_OVERLAY_STYLE))
    rgb[overlay] = _styleLut(_OVERLAY_STYLE)[classes[overlay]]
    featureIds = np.asarray(scene.layers["featureId"])[row, col]
    buildingRoof = (
        (classes == CLASS_BUILDING)
        & (featureIds > 0)
        & (featureIds <= appearance.roofColors.shape[0])
    )
    rgb[buildingRoof] = appearance.roofColors[featureIds[buildingRoof] - 1]
    underTrees = (canopy >= MIN_TREE_HEIGHT_m) & ~overlay  # roads under trees stay roads
    rgb[underTrees] = CONCEPT_COLORS["underTrees"]
    averaged = rgb.reshape(size, k, size, k, 3).mean(axis=(1, 3))
    out: npt.NDArray[np.uint8] = np.clip(np.round(averaged), 0, 255).astype(np.uint8)
    return out


@functools.cache
def _orthoToGeographic(wkt: str) -> pyproj.Transformer:
    return pyproj.Transformer.from_crs(pyproj.CRS.from_wkt(wkt), "EPSG:4326", always_xy=True)


def _deckTop(
    scene: Scene,
    d: Any,
    indices: npt.NDArray[np.int64],
    owner: npt.NDArray[np.int64],
    px: FloatArray,
) -> FloatArray:
    """Orthometric deck-top height at pixel positions (mirrors the grid layers)."""
    deck = indices[owner]
    top = np.empty(owner.size)
    gx, gy = scene.geometry.pixelToGrid(px[:, 0], px[:, 1])
    e, n = scene.geometry.gridToEnu(gx, gy)
    lat, lon, _ = scene.geometry.frame.columnPointAtHeight(e, n, 0.0)
    groundAt = scene.models.ground.orthometricAt(lat, lon)
    for k in np.unique(deck):
        mine = deck == k
        line = d.centerlines[k]
        if line is None:
            top[mine] = d.groundStart_m[k] + d.levelHeight_m[k]
        else:
            frac = shapely.line_locate_point(line, shapely.points(px[mine]), normalized=True)
            length_m = shapely.length(line) * scene.geometry.gridSpacing_m
            top[mine] = groundAt[mine] + deckTopAboveGround_m(
                frac, length_m, d.levelHeight_m[k], d.taperStart_m[k], d.taperEnd_m[k]
            )
    return top


def _triCorners(polygons: npt.NDArray[np.object_]) -> tuple[FloatArray, npt.NDArray[np.int64]]:
    """Constrained-Delaunay triangle corners ``(3m, 2)`` and the polygon of each corner."""
    tris, owner = shapely.get_parts(
        shapely.constrained_delaunay_triangles(polygons), return_index=True
    )
    corners = shapely.get_coordinates(tris).reshape(-1, 4, 2)[:, :3].reshape(-1, 2)
    return corners, np.repeat(owner, 3)


def _ringEdges(
    polygons: npt.NDArray[np.object_],
) -> tuple[FloatArray, FloatArray, npt.NDArray[np.int64]]:
    """Every ring segment as ``(start, end, polygon index)``."""
    rings, ringOwner = shapely.get_rings(polygons, return_index=True)
    coords, coordRing = shapely.get_coordinates(rings, return_index=True)
    a = np.flatnonzero(coordRing[:-1] == coordRing[1:])  # consecutive points of one ring
    return coords[a], coords[a + 1], ringOwner[coordRing[a]]


def _walls(
    frame: Any,
    geoms: npt.NDArray[np.object_],
    baseEll: FloatArray,
    roofEll: FloatArray,
    colors: npt.NDArray[np.uint8],
) -> MeshPart:
    start, end, owner = _ringEdges(geoms)

    def enu(lonLat: FloatArray, h: FloatArray) -> FloatArray:
        e, n, u = frame.geodeticToEnu(lonLat[:, 1], lonLat[:, 0], h)
        return np.column_stack([e, n, u])

    p0 = enu(start, baseEll[owner])
    p1 = enu(end, baseEll[owner])
    p2 = enu(end, roofEll[owner])
    p3 = enu(start, roofEll[owner])
    length_m = np.linalg.norm(p1 - p0, axis=1)
    height_m = np.maximum(roofEll - baseEll, 0.0)[owner]
    return _quads(p0, p1, p2, p3, colors[owner], facadeUv(length_m, height_m))


def _quads(
    p0: FloatArray,
    p1: FloatArray,
    p2: FloatArray,
    p3: FloatArray,
    colour: tuple[int, int, int] | npt.NDArray[np.uint8],
    uv: npt.NDArray[np.float32] | None = None,
) -> MeshPart:
    m = p0.shape[0]
    if m == 0:
        return MeshPart.empty(withColors=True, withUv=uv is not None)
    positions = np.stack([p0, p1, p2, p3], axis=1).reshape(-1, 3)
    base = (np.arange(m, dtype=np.uint32) * 4)[:, None]
    tris = np.concatenate(
        [base + np.array([0, 1, 2], dtype=np.uint32), base + np.array([0, 2, 3], dtype=np.uint32)]
    )
    quadColors = _colors(m, colour) if isinstance(colour, tuple) else np.asarray(colour)
    return MeshPart(
        positions,
        tris.astype(np.uint32),
        uv=None if uv is None else uv.reshape(-1, 2),
        colors=np.repeat(quadColors, 4, axis=0).astype(np.uint8),
    )


def _colors(count: int, colour: tuple[int, int, int]) -> npt.NDArray[np.uint8]:
    return np.tile(np.array(colour, dtype=np.uint8), (count, 1))
