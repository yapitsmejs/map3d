"""3D Tiles 1.1 tileset writer (SPEC section 7).

The swath is split into a quadtree of tiles in grid-frame coordinates; leaves
are at most ``leafSize_m``. Every tile carries a GLB with:

- RTIN terrain, with a tolerance that doubles per level up;
- the buildings whose centroid falls in it (parent tiles keep only buildings
  at least :func:`minBuildingHeight_m` tall, so distant views keep the
  skyline);
- decks (leaves and their parents);
- a top-down texture in the ``"conceptual"`` style (flat map colours from
  land cover, roads, roofs and trees; lit, so the renderer shades the shapes).

Refinement is ``REPLACE``. A non-leaf tile's ``geometricError`` is the larger
of its terrain tolerance and the building height it omits. The root
``transform`` maps scene ENU to ECEF, and glTF content is y-up, per the 3D
Tiles convention.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import shapely

from map3d import __version__
from map3d.mesh.buildingAppearance import BuildingAppearance, facadeTexture
from map3d.mesh.gltf import InstancedPart, MeshPart, writeGlb
from map3d.mesh.scenery import (
    TileExtent,
    buildingParts,
    deckParts,
    extentBoxPx,
    renderTexture,
    terrainPart,
)
from map3d.mesh.trees import TreeSet

if TYPE_CHECKING:
    from map3d.scene import Scene

__all__ = [
    "PARENT_TREE_MIN_m",
    "STYLES",
    "Style",
    "minBuildingHeight_m",
    "quadtreeDepth",
    "write3dTiles",
]

FloatArray = npt.NDArray[np.float64]
Style = Literal["conceptual"]
STYLES: tuple[str, ...] = ("conceptual",)

_BASE_SKYLINE_m = 10.0
PARENT_TREE_MIN_m = 8.0  # trees kept one level above the leaves; none higher up
_BOX_PAD_m = 0.5


def quadtreeDepth(width_m: float, height_m: float, leafSize_m: float) -> int:
    """Levels below the root needed for leaves of at most ``leafSize_m``."""
    return max(0, math.ceil(math.log2(max(width_m, height_m) / leafSize_m)))


def minBuildingHeight_m(level: int, depth: int) -> float:
    """Shortest building kept in a tile at ``level`` (0 at the leaves)."""
    return 0.0 if level >= depth else _BASE_SKYLINE_m * 2.0 ** (depth - 1 - level)


@dataclass(frozen=True)
class _Params:
    depth: int
    rootExtent: TileExtent
    textureSize: int
    terrainGridSize: int
    leafTerrainError_m: float
    trees: TreeSet | None
    buildingAppearance: BuildingAppearance
    facadePng: bytes


def write3dTiles(
    scene: Scene,
    outDir: str | os.PathLike[str],
    leafSize_m: float = 512.0,
    textureSize: int = 512,
    terrainGridSize: int = 129,
    leafTerrainError_m: float = 0.25,
    style: Style = "conceptual",
    withTrees: bool = True,
    buildingImageryPath: str | os.PathLike[str] | None = None,
) -> Path:
    """Write the scene as a 3D Tiles 1.1 tileset.

    Args:
        scene: A scene from :func:`map3d.scene.buildScene` (needs its models).
        outDir: Output directory (``tileset.json`` + ``tiles/``).
        leafSize_m: Maximum leaf tile size.
        textureSize: Texture side in pixels per tile.
        terrainGridSize: RTIN lattice side per tile (``2^k + 1``).
        leafTerrainError_m: RTIN tolerance at the leaves.
        style: ``"conceptual"`` (flat map colours, lit), the default.
        withTrees: Draw individual trees (crowns on trunks, instanced) from
            the canopy sources: all in the leaves, those at least
            :data:`PARENT_TREE_MIN_m` tall one level up. See :mod:`map3d.mesh.trees`.
        buildingImageryPath: Optional high-resolution RGB orthophoto used to
            derive roof and facade palettes. Default: the pack's imagery, with
            neutral fallback where sampling is unreliable. Facade windows and
            floor bands are procedural illustrations, not measured features.

    Returns:
        Path of ``tileset.json``.

    Raises:
        ValueError: If the scene was not built with ``buildScene`` or the style
            is unknown.
    """
    if scene.models is None or scene.pack is None:
        raise ValueError("write3dTiles needs a scene returned by buildScene (with its models)")
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; choose from {', '.join(STYLES)}")
    root = Path(outDir)
    g = scene.geometry
    appearance = BuildingAppearance.fromScene(scene, buildingImageryPath=buildingImageryPath)
    width_m, height_m = g.nCols * g.gridSpacing_m, g.nRows * g.gridSpacing_m
    params = _Params(
        depth=quadtreeDepth(width_m, height_m, leafSize_m),
        rootExtent=TileExtent(-width_m / 2, width_m / 2, -height_m / 2, height_m / 2),
        textureSize=textureSize,
        terrainGridSize=terrainGridSize,
        leafTerrainError_m=leafTerrainError_m,
        trees=TreeSet.fromScene(scene) if withTrees else None,
        buildingAppearance=appearance,
        facadePng=facadeTexture(),
    )
    buildingXY = _centroidsGrid(scene, scene.models.buildings.pixelGeometries)
    tileJson = _writeTile(scene, root, params, buildingXY, 0, 0, 0)
    tileJson["transform"] = _enuToEcef(scene)
    tileJson["refine"] = "REPLACE"
    attribution = [s["attribution"] for s in scene.metadata.get("sources", [])]
    if appearance.attribution is not None and appearance.attribution not in attribution:
        attribution.append(appearance.attribution)
    tileset = {
        "asset": {
            "version": "1.1",
            "generator": f"map3d {__version__}",
            "extras": {
                "attribution": attribution,
                "request": scene.metadata.get("request", {}),
                "style": style,
                "buildingAppearance": {
                    "facades": "procedural; windows and floor bands are illustrative, not measured",
                    "imagerySource": appearance.sourceId,
                    "imageryColoredBuildings": appearance.coloredCount,
                    "mappedColorBuildings": appearance.mappedCount,
                    "buildingCount": len(scene.models.buildings),
                },
            },
        },
        "geometricError": 2.0 * tileJson["geometricError"] + 1.0,
        "root": tileJson,
    }
    path = root / "tileset.json"
    path.write_text(json.dumps(tileset, indent=1), encoding="utf-8")
    return path


def _writeTile(
    scene: Scene,
    root: Path,
    params: _Params,
    buildingXY: FloatArray,
    level: int,
    ix: int,
    iy: int,
) -> dict[str, Any]:
    n = 2**level
    r = params.rootExtent
    sx, sy = (r.x1 - r.x0) / n, (r.y1 - r.y0) / n
    extent = TileExtent(r.x0 + ix * sx, r.x0 + (ix + 1) * sx, r.y0 + iy * sy, r.y0 + (iy + 1) * sy)
    depth = params.depth
    terrainError = params.leafTerrainError_m * 2.0 ** (depth - level)
    minHeight = minBuildingHeight_m(level, depth)

    b = scene.models.buildings
    inTile = _inTile(buildingXY, r, sx, sy, ix, iy, n)
    selB = (
        np.flatnonzero(inTile & (b.height_m >= minHeight))
        if len(b)
        else np.array([], dtype=np.int64)
    )
    # Decks are drawn in every tile they intersect, clipped to it (see deckParts).
    selD = (
        np.flatnonzero(
            shapely.intersects(scene.models.decks.pixelShapes, extentBoxPx(scene, extent))
        )
        if level >= depth - 1 and len(scene.models.decks)
        else np.array([], dtype=np.int64)
    )
    terrain = terrainPart(scene, extent, params.terrainGridSize, terrainError)
    roofs, walls = buildingParts(scene, selB, extent, params.buildingAppearance)
    tops, sides = deckParts(scene, selD, extent)
    textured = MeshPart.concatenate([terrain, roofs, tops])
    colored = sides
    instanced: list[InstancedPart] = []
    treeCorners: list[FloatArray] = []
    if params.trees is not None:
        selT = _treesInTile(params.trees, r, sx, sy, ix, iy, n, level, depth)
        instanced = params.trees.select(selT)
        treeCorners = list(params.trees.extents(selT))
    uri = f"tiles/{level}/{ix}_{iy}.glb"
    writeGlb(
        root / uri,
        textured,
        renderTexture(scene, extent, params.buildingAppearance, params.textureSize),
        colored,
        instanced=instanced,
        facades=walls,
        facadeTexture=params.facadePng,
    )

    corners = [p.positions_ENU_m for p in (textured, colored, walls) if not p.isEmpty]
    positions = np.concatenate(corners + treeCorners)
    lo, hi = positions.min(axis=0) - _BOX_PAD_m, positions.max(axis=0) + _BOX_PAD_m
    tile: dict[str, Any] = {
        "boundingVolume": {"box": _box(lo, hi)},
        "geometricError": 0.0 if level >= depth else max(terrainError, minHeight),
        "content": {"uri": uri},
    }
    if level < depth:
        children = [
            _writeTile(
                scene,
                root,
                params,
                buildingXY,
                level + 1,
                2 * ix + dx,
                2 * iy + dy,
            )
            for dy in (0, 1)
            for dx in (0, 1)
        ]
        for child in children:  # parents must enclose their children
            clo, chi = _boxBounds(child["boundingVolume"]["box"])
            lo, hi = np.minimum(lo, clo), np.maximum(hi, chi)
        tile["boundingVolume"]["box"] = _box(lo, hi)
        tile["children"] = children
    return tile


def _treesInTile(
    trees: TreeSet,
    r: TileExtent,
    sx: float,
    sy: float,
    ix: int,
    iy: int,
    n: int,
    level: int,
    depth: int,
) -> npt.NDArray[np.int64]:
    if len(trees) == 0 or level < depth - 1:
        return np.array([], dtype=np.int64)
    inTile = _inTile(np.column_stack([trees.gx_m, trees.gy_m]), r, sx, sy, ix, iy, n)
    if level < depth:
        inTile &= trees.height_m >= PARENT_TREE_MIN_m
    return np.flatnonzero(inTile)


def _inTile(
    xy: FloatArray, r: TileExtent, sx: float, sy: float, ix: int, iy: int, n: int
) -> npt.NDArray[np.bool_]:
    if xy.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    cx = np.clip(np.floor((xy[:, 0] - r.x0) / sx), 0, n - 1)
    cy = np.clip(np.floor((xy[:, 1] - r.y0) / sy), 0, n - 1)
    result: npt.NDArray[np.bool_] = (cx == ix) & (cy == iy)
    return result


def _centroidsGrid(scene: Scene, pixelGeoms: npt.NDArray[np.object_]) -> FloatArray:
    if pixelGeoms.size == 0:
        return np.zeros((0, 2))
    c = shapely.centroid(pixelGeoms)
    return np.column_stack(scene.geometry.pixelToGrid(shapely.get_x(c), shapely.get_y(c)))


def _box(lo: FloatArray, hi: FloatArray) -> list[float]:
    c = 0.5 * (lo + hi)
    h = 0.5 * (hi - lo)
    return [float(v) for v in (c[0], c[1], c[2], h[0], 0.0, 0.0, 0.0, h[1], 0.0, 0.0, 0.0, h[2])]


def _boxBounds(box: list[float]) -> tuple[FloatArray, FloatArray]:
    c = np.array(box[:3])
    h = np.array([box[3], box[7], box[11]])
    return c - h, c + h


def _enuToEcef(scene: Scene) -> list[float]:
    frame = scene.geometry.frame
    e, n, u = frame.rotation  # rows: E, N, U unit vectors in ECEF
    o = frame.origin_ECEF_m
    return [float(v) for v in (*e, 0.0, *n, 0.0, *u, 0.0, *o, 1.0)]  # column-major
