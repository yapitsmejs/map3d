"""Individual trees for the 3D Tiles mesh (SPEC section 7).

The canopy layer is a height field; drawn as a surface it looks like green
mounds. Instead, the mesh keeps the true ground and places one rounded crown
on a short trunk per tree:

1. **Mapped trees first:** tree points (``trees`` role, e.g. NParks) keep their
   position and height.
2. **Trees from the canopy raster:** the raster is lightly smoothed; treetops
   are pixels at least :data:`MIN_TREE_HEIGHT_m` tall that are the highest
   within :data:`TREETOP_WINDOW_m` (so crown edges do not start trees of their
   own). Treetops closer than :data:`TREETOP_SPACING` of a crown are merged
   (tallest wins). Every canopy pixel then joins the nearest treetop it is
   connected to through canopy, and each crown's radius is that of a disc with
   its region's area (1.5-6 m, and at most 0.6 h + 1.5 m). Canopy no treetop
   reaches within 6 m (smooth woods) is seeded again at its tallest points, a
   crown apart. Crowns therefore add up to the canopy: a wood becomes a clump
   of rounded crowns, a street tree a single one.
3. **Placement:** trees stand on the scene's ground model (smoothed and
   residue-corrected, like everything else), inside the swath, and not on
   buildings or decks.

Crowns and trunks are written with ``EXT_mesh_gpu_instancing``: one shape per
tile plus a position, size and yaw per tree (the yaw varies which side of the
crown's fixed lump pattern each tree shows). Crowns are slightly lumpy spheres,
not smooth ones, so they read as foliage rather than balls; trunks grow with
both the crown and the tree height, and taller trees carry their crowns higher
off the ground.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pyproj
import shapely

from map3d.layers import _kernels
from map3d.layers.landCover import CLASS_BUILDING, CLASS_DECK
from map3d.layers.terrain import pixelSize_m, smoothWindow
from map3d.mesh.gltf import InstancedPart, MeshPart
from map3d.pack.reader import RasterWindow

if TYPE_CHECKING:
    from map3d.scene import Scene

__all__ = [
    "MAX_CROWN_RADIUS_m",
    "MIN_CROWN_RADIUS_m",
    "TREETOP_SPACING",
    "CROWN_COLORS",
    "MIN_TREE_HEIGHT_m",
    "TREETOP_WINDOW_m",
    "TRUNK_COLOR",
    "TreeSet",
    "crownRadius_m",
]

FloatArray = npt.NDArray[np.float64]

MIN_TREE_HEIGHT_m = 2.0
CROWN_COLORS: tuple[tuple[int, int, int], ...] = ((84, 150, 64), (100, 166, 76), (70, 134, 56))
TRUNK_COLOR = (122, 100, 80)
_SMOOTH_FWHM_m = 2.0  # removes single-pixel noise in 1 m canopy rasters
TREETOP_WINDOW_m = 1.5  # a treetop is the highest pixel within this distance
TREETOP_SPACING = 0.5  # treetops closer than this x allometric radius are one tree
MIN_CROWN_RADIUS_m = 1.5
MAX_CROWN_RADIUS_m = 6.0  # also the reach of a crown region
_FILL_ROUNDS = 4  # gap-filling rounds for canopy no treetop reaches
_FILL_TOLERANCE = 0.01  # stop when less than this fraction of canopy is unclaimed
_RADIUS_PER_HEIGHT_MAX = 0.6  # crowns are never wider than 0.6 h + 1.5 m
_RADIUS_AT_ZERO_HEIGHT_m = 1.5
_CROWN_RADIUS_BASE_m = 1.0
_CROWN_RADIUS_PER_HEIGHT = 0.3
_CROWN_RADIUS_RANGE_m = (2.0, 8.0)
_MAX_VALID_m = 150.0  # uint8 canopy rasters may use 255 as a fill value
_CROWN_DEPTH_PER_HEIGHT = 0.4  # crown half-height at most this fraction of the tree
_CROWN_DEPTH_PER_RADIUS = 0.85  # ... and slightly flattened
_TRUNK_RADIUS_PER_CROWN = 0.08
_TRUNK_RADIUS_PER_HEIGHT = 0.02  # trunks grow with the tree, not just the crown
_TRUNK_RADIUS_MAX_m = 0.7
_CROWN_LUMPINESS = 0.12  # radial noise amplitude of the crown surface
_GOLDEN_ANGLE = np.pi * (3.0 - 5.0**0.5)  # per-instance yaw spread


@dataclass(frozen=True, eq=False)
class TreeSet:
    """Trees to draw, in scene coordinates.

    Attributes:
        lat_deg: Latitude of each trunk.
        lon_deg: Longitude of each trunk.
        e_m: ENU east of the trunk base.
        n_m: ENU north of the trunk base.
        baseU_m: ENU up of the trunk base (on the ground).
        gx_m: Grid-frame x (for tiling).
        gy_m: Grid-frame y.
        height_m: Tree height above ground.
        crownRadius_m: Crown radius.
    """

    lat_deg: FloatArray
    lon_deg: FloatArray
    e_m: FloatArray
    n_m: FloatArray
    baseU_m: FloatArray
    gx_m: FloatArray
    gy_m: FloatArray
    height_m: FloatArray
    crownRadius_m: FloatArray

    def __len__(self) -> int:
        return int(self.e_m.size)

    @classmethod
    def fromScene(cls, scene: Scene) -> TreeSet:
        """Find the scene's trees (mapped points plus crowns packed into the canopy raster).

        Args:
            scene: A scene from :func:`map3d.scene.buildScene` (needs its models).

        Returns:
            The trees (possibly none).
        """
        assert scene.models is not None
        lat, lon, height, radius = _treesFromSources(scene)
        g = scene.geometry
        ground = scene.models.ground
        hEll = ground.orthometricAt(lat, lon) + ground.undulation_m(lat, lon)
        e, n, u = g.frame.geodeticToEnu(lat, lon, hEll)
        a = g.gridAxes_ENU
        gx, gy = e * a[0, 0] + n * a[0, 1], e * a[1, 0] + n * a[1, 1]
        s = g.gridSpacing_m
        halfW, halfH = 0.5 * g.nCols * s, 0.5 * g.nRows * s
        keep = np.isfinite(u) & (np.abs(gx) <= halfW) & (np.abs(gy) <= halfH)
        col = np.clip(((gx + halfW) / s).astype(np.int64), 0, g.nCols - 1)
        row = np.clip(((gy + halfH) / s).astype(np.int64), 0, g.nRows - 1)
        classes = np.asarray(scene.layers["landClass"])[row, col]
        keep &= ~np.isin(classes, [CLASS_BUILDING, CLASS_DECK])
        return cls(
            lat[keep],
            lon[keep],
            e[keep],
            n[keep],
            u[keep],
            gx[keep],
            gy[keep],
            height[keep],
            radius[keep],
        )

    def select(self, indices: npt.NDArray[np.int64]) -> list[InstancedPart]:
        """Instanced crowns (three shades) and trunks for the given trees.

        Args:
            indices: Tree indices.

        Returns:
            The instanced parts (empty when there are no trees).
        """
        idx = np.asarray(indices, dtype=np.int64)
        if idx.size == 0:
            return []
        h, r = self.height_m[idx], self.crownRadius_m[idx]
        e, n, base = self.e_m[idx], self.n_m[idx], self.baseU_m[idx]
        depth = np.minimum(_CROWN_DEPTH_PER_HEIGHT * h, _CROWN_DEPTH_PER_RADIUS * r)
        centre = base + h - depth
        yaw = (idx * _GOLDEN_ANGLE) % (2.0 * np.pi)
        half = 0.5 * yaw
        yawQuat = np.column_stack(
            [np.zeros_like(half), np.zeros_like(half), np.sin(half), np.cos(half)]
        )
        parts: list[InstancedPart] = []
        for k, colour in enumerate(CROWN_COLORS):
            mine = idx % len(CROWN_COLORS) == k
            if mine.any():
                parts.append(
                    InstancedPart(
                        _CROWN,
                        colour,
                        np.column_stack([e[mine], n[mine], centre[mine]]),
                        np.column_stack([r[mine], r[mine], depth[mine]]),
                        yawQuat[mine],
                    )
                )
        trunkR = np.clip(
            _TRUNK_RADIUS_PER_CROWN * r + _TRUNK_RADIUS_PER_HEIGHT * h, 0.15, _TRUNK_RADIUS_MAX_m
        )
        parts.append(
            InstancedPart(
                _TRUNK,
                TRUNK_COLOR,
                np.column_stack([e, n, base]),
                np.column_stack([trunkR, trunkR, h - depth]),
            )
        )
        return parts

    def extents(self, indices: npt.NDArray[np.int64]) -> tuple[FloatArray, FloatArray]:
        """ENU bounding corners ``(lo, hi)`` of the given trees, ``(k, 3)`` each."""
        idx = np.asarray(indices, dtype=np.int64)
        r = self.crownRadius_m[idx] * (1.0 + _CROWN_LUMPINESS)  # the lumps stick out
        lo = np.column_stack([self.e_m[idx] - r, self.n_m[idx] - r, self.baseU_m[idx]])
        hi = np.column_stack(
            [self.e_m[idx] + r, self.n_m[idx] + r, self.baseU_m[idx] + self.height_m[idx]]
        )
        return lo, hi


def _treesFromSources(scene: Scene) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """``(lat, lon, height, crown radius)``: mapped tree points, then raster treetops."""
    assert scene.models is not None
    canopy = scene.models.canopy
    g = scene.geometry
    pLat = pLon = pH = np.zeros(0)
    if canopy.treeDiscs.size:
        centres = shapely.centroid(canopy.treeDiscs)
        s = g.gridSpacing_m
        gx = shapely.get_x(centres) * s - 0.5 * g.nCols * s
        gy = shapely.get_y(centres) * s - 0.5 * g.nRows * s
        a = g.gridAxes_ENU
        e, n = gx * a[0, 0] + gy * a[1, 0], gx * a[0, 1] + gy * a[1, 1]
        pLat, pLon, _ = g.frame.columnPointAtHeight(e, n, 0.0)
        pH = np.asarray(canopy.treeHeights_m, dtype=np.float64)
    if canopy.window is None:
        return pLat, pLon, pH, crownRadius_m(pH)
    return _segmentRaster(canopy.window, pLat, pLon, pH)


def _segmentRaster(
    window: RasterWindow, pLat: FloatArray, pLon: FloatArray, pH: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Mapped points plus raster treetops, with crowns sized from the canopy they cover.

    Treetops are deduplicated (:func:`packCrowns` with :data:`TREETOP_SPACING` of
    the allometric radius), then every canopy pixel joins the nearest seed it is
    connected to (:func:`growRegions`, within :data:`MAX_CROWN_RADIUS_m`); a
    raster tree's crown radius is that of a disc with its region's area. Mapped
    points keep the allometric radius (the raster may not show them).
    """
    raw = window.values
    clean = np.where(np.isfinite(raw) & (raw >= 0) & (raw <= _MAX_VALID_m), raw, 0.0)
    smooth = smoothWindow(
        RasterWindow(clean, window.transform, window.crs, window.sourceId), _SMOOTH_FWHM_m
    ).values
    nR, nC = smooth.shape
    flat = smooth.ravel()
    pxX, pxY = pixelSize_m(window)
    peak = _maxFilter(smooth, int(TREETOP_WINDOW_m / pxY + 0.5), int(TREETOP_WINDOW_m / pxX + 0.5))
    canopyMask = smooth >= MIN_TREE_HEIGHT_m
    cand = np.flatnonzero(canopyMask.ravel() & (flat >= peak.ravel()))
    cand = cand[np.argsort(-flat[cand], kind="stable")]

    pc, pr = window.pixelCoords(pLat, pLon)
    pCol, pRow = np.floor(pc).astype(np.int64), np.floor(pr).astype(np.int64)
    inWindow = (pCol >= 0) & (pCol < nC) & (pRow >= 0) & (pRow < nR)
    nPoints = int(inWindow.sum())

    rows = np.concatenate([pRow[inWindow], cand // nC])
    cols = np.concatenate([pCol[inWindow], cand % nC])
    heights = np.concatenate([pH[inWindow], flat[cand]])
    claim = TREETOP_SPACING * crownRadius_m(heights)
    accepted = np.zeros(rows.size, dtype=np.bool_)
    _kernels.packCrowns(rows, cols, claim / pxY, claim / pxX, nR, nC, accepted)

    seeds = np.flatnonzero(accepted)
    seedRows, seedCols, seedH = rows[seeds], cols[seeds], heights[seeds]
    isRaster = seeds >= nPoints
    labels = np.full((nR, nC), -1, dtype=np.int32)
    _kernels.growRegions(seedRows, seedCols, canopyMask, pxY, pxX, MAX_CROWN_RADIUS_m, labels)
    # Fill canopy that no treetop reaches (smooth woods): seed its tallest
    # points a crown apart and grow again.
    total = max(int(canopyMask.sum()), 1)
    for _ in range(_FILL_ROUNDS):
        free = np.flatnonzero(canopyMask.ravel() & (labels.ravel() < 0))
        if free.size <= _FILL_TOLERANCE * total:
            break
        free = free[np.argsort(-flat[free], kind="stable")]
        fr, fc = free // nC, free % nC
        claim = np.full(free.size, MAX_CROWN_RADIUS_m)
        take = np.zeros(free.size, dtype=np.bool_)
        _kernels.packCrowns(fr, fc, claim / pxY, claim / pxX, nR, nC, take)
        seedRows = np.concatenate([seedRows, fr[take]])
        seedCols = np.concatenate([seedCols, fc[take]])
        seedH = np.concatenate([seedH, flat[free[take]]])
        isRaster = np.concatenate([isRaster, np.ones(int(take.sum()), dtype=bool)])
        _kernels.growRegions(seedRows, seedCols, canopyMask, pxY, pxX, MAX_CROWN_RADIUS_m, labels)
    area = np.bincount(labels[labels >= 0], minlength=seedRows.size) * pxX * pxY
    widest = _RADIUS_AT_ZERO_HEIGHT_m + _RADIUS_PER_HEIGHT_MAX * seedH
    radius = np.clip(
        np.minimum(np.sqrt(area / np.pi), widest), MIN_CROWN_RADIUS_m, MAX_CROWN_RADIUS_m
    )
    radius[~isRaster] = crownRadius_m(seedH[~isRaster])

    t = window.transform
    c, r = seedCols[isRaster] + 0.5, seedRows[isRaster] + 0.5
    x, y = t.a * c + t.b * r + t.c, t.d * c + t.e * r + t.f
    toGeo = pyproj.Transformer.from_crs(
        pyproj.CRS.from_user_input(window.crs.to_wkt()), "EPSG:4326", always_xy=True
    )
    lon, lat = toGeo.transform(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    outside = ~inWindow
    keptInside = np.flatnonzero(inWindow)[accepted[:nPoints]]
    order = np.concatenate([keptInside, np.flatnonzero(outside)])
    pointRadius = np.concatenate([radius[~isRaster], crownRadius_m(pH[outside])])
    return (
        np.concatenate([pLat[order], np.asarray(lat)]),
        np.concatenate([pLon[order], np.asarray(lon)]),
        np.concatenate([pH[order], seedH[isRaster]]),
        np.concatenate([pointRadius, radius[isRaster]]),
    )


def crownRadius_m(height_m: npt.ArrayLike) -> FloatArray:
    """Drawn crown radius for a tree of a given height: ``0.3 h + 1``, within 2-8 m."""
    h = np.asarray(height_m, dtype=np.float64)
    lo, hi = _CROWN_RADIUS_RANGE_m
    return np.clip(_CROWN_RADIUS_BASE_m + _CROWN_RADIUS_PER_HEIGHT * h, lo, hi)


def _maxFilter(values: FloatArray, radiusRows: int, radiusCols: int) -> FloatArray:
    """Maximum over a ``(2 rr + 1) x (2 rc + 1)`` window (separable; edges see less)."""
    out = values
    for axis, radius in ((0, radiusRows), (1, radiusCols)):
        if radius <= 0:
            continue
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, constant_values=-np.inf)
        n = out.shape[axis]
        acc = np.full_like(out, -np.inf)
        for k in range(2 * radius + 1):
            sl = [slice(None), slice(None)]
            sl[axis] = slice(k, k + n)
            acc = np.maximum(acc, padded[tuple(sl)])
        out = acc
    return out


def _icosphere(subdivisions: int = 1) -> MeshPart:
    """Unit sphere: an icosahedron subdivided ``subdivisions`` times."""
    t = (1.0 + 5.0**0.5) / 2.0
    verts = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ]  # fmt: skip
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]  # fmt: skip
    points = [np.array(v, dtype=np.float64) / np.linalg.norm(v) for v in verts]
    cache: dict[tuple[int, int], int] = {}

    def midpoint(i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        if key not in cache:
            m = points[i] + points[j]
            points.append(m / np.linalg.norm(m))
            cache[key] = len(points) - 1
        return cache[key]

    fine = list(faces)
    for _ in range(subdivisions):
        fine = [
            tri
            for a, b, c in fine
            for tri in (
                (a, midpoint(a, b), midpoint(c, a)),
                (b, midpoint(b, c), midpoint(a, b)),
                (c, midpoint(c, a), midpoint(b, c)),
                (midpoint(a, b), midpoint(b, c), midpoint(c, a)),
            )
        ]
    pos = np.array(points)
    tris = np.array(fine, dtype=np.uint32)
    p = pos[tris]
    outward = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p.mean(axis=1))
    tris[outward < 0] = tris[outward < 0][:, ::-1]
    return MeshPart(pos, tris)


def _cylinder(sides: int = 6) -> MeshPart:
    """Unit open cylinder from z = 0 to 1, radius 1, outward-facing."""
    angle = np.arange(sides) * 2.0 * np.pi / sides
    ring = np.column_stack([np.cos(angle), np.sin(angle)])
    pos = np.vstack(
        [np.column_stack([ring, np.zeros(sides)]), np.column_stack([ring, np.ones(sides)])]
    )
    i = np.arange(sides)
    j = (i + 1) % sides
    tris = np.concatenate(
        [np.column_stack([i, j, j + sides]), np.column_stack([i, j + sides, i + sides])]
    ).astype(np.uint32)
    return MeshPart(pos, tris)


def _lumpyCrown() -> MeshPart:
    """Unit sphere with low-frequency radial bumps, so crowns read as foliage.

    The displacement is a fixed sum of direction-space sine waves (the same for
    every crown); per-instance yaw rotations (:meth:`TreeSet.select`) spread the
    pattern, so a wood does not look like rows of identical balls.
    """
    sphere = _icosphere(2)
    pos = sphere.positions_ENU_m
    waves = [
        ([0.82, 0.31, -0.24], 3.1, 0.0),
        ([-0.26, 0.71, 0.42], 4.7, 2.1),
        ([0.13, -0.47, 0.79], 5.9, 4.4),
        ([0.55, 0.48, 0.61], 7.3, 1.3),
    ]  # fmt: skip
    bump = np.zeros(pos.shape[0])
    for axis, freq, phase in waves:
        bump += np.sin(freq * (pos @ np.asarray(axis)) + phase)
    bump /= len(waves)
    lumped = pos * (1.0 + _CROWN_LUMPINESS * bump)[:, None]
    # Keep the top and bottom poles (where the trunk meets the crown) round.
    polar = np.abs(pos[:, 2]) > 0.92
    lumped[polar] = pos[polar]
    return MeshPart(lumped, sphere.triangles)


_CROWN = _lumpyCrown()
_TRUNK = _cylinder()
