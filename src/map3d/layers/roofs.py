"""Roof shapes of buildings and building parts (SPEC section 8).

A roof is a height field over its footprint: :meth:`Roof.rise` gives the height
above the eaves at any point of a local metric plane (x east, y north, metres)
centred on the footprint. The grid layers (top surface per node) and the 3D
Tiles mesh (a lifted triangulation, :meth:`Roof.triangles`) evaluate the same
field, so they agree.

Shapes follow OpenStreetMap's Simple 3D Buildings tags as carried by Overture
(``roof_shape``, ``roof_height``, ``roof_direction``, ``roof_orientation``):

- **Axis shapes** use the footprint's frame: ``u`` across the ridge, ``v``
  along it. The ridge runs along the longer side of the footprint's minimum
  rotated rectangle; ``roof_orientation = across`` turns it, and
  ``roof_direction`` (the bearing a roof slope faces) sets ``u`` explicitly.

  - skillion: one plane, highest on the side opposite ``roof_direction``.
  - gabled (also gambrel, saltbox): two planes meeting at the ridge.
  - round on a near-rectangular footprint: a barrel vault.
  - hipped (also half-hipped, mansard) on a near-rectangular footprint: four
    planes, as the straight skeleton of the rectangle.

- **Radial shapes** rise along rays from the outline to an apex at the
  centroid: pyramidal (linear), dome (quarter circle) and onion (a dome with a
  pointed tip).
- **Inset shapes** follow the distance to the outline. They stand in for round
  and hipped roofs on footprints that are not near-rectangular, and for radial
  shapes whose apex is not inside the footprint. Round gives a shell that comes
  down to the eaves on every side (e.g. a conservatory dome).

Where a roof does not come down to the eaves at the outline (gable and barrel
ends), walls rise to meet it.

**Roof height:** the mapped ``roof_height`` when present, otherwise a default
per shape (see :func:`makeRoof`). It never exceeds the part's vertical extent,
and defaults for pitched and round roofs take at most half of it, so walls
remain (only domes and onions may default to the full height).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import shapely
from shapely.ops import polylabel

__all__ = [
    "CURVED_SHAPES",
    "RECTANGULAR_FILL",
    "Roof",
    "RoofShape",
    "makeRoof",
    "roofShapeCodes",
]

FloatArray = npt.NDArray[np.float64]


class RoofShape(IntEnum):
    """Roof shape codes (``uint8``)."""

    FLAT = 0
    SKILLION = 1
    GABLED = 2
    HIPPED = 3
    PYRAMIDAL = 4
    DOME = 5
    ONION = 6
    ROUND = 7


_ALIASES = {
    "flat": RoofShape.FLAT,
    "skillion": RoofShape.SKILLION,
    "gabled": RoofShape.GABLED,
    "gambrel": RoofShape.GABLED,
    "saltbox": RoofShape.GABLED,
    "hipped": RoofShape.HIPPED,
    "half_hipped": RoofShape.HIPPED,
    "mansard": RoofShape.HIPPED,
    "pyramidal": RoofShape.PYRAMIDAL,
    "dome": RoofShape.DOME,
    "onion": RoofShape.ONION,
    "round": RoofShape.ROUND,
}
CURVED_SHAPES = frozenset({RoofShape.DOME, RoofShape.ONION, RoofShape.ROUND})
_PITCHED_SHAPES = frozenset(
    {RoofShape.SKILLION, RoofShape.GABLED, RoofShape.HIPPED, RoofShape.PYRAMIDAL}
)
RECTANGULAR_FILL = 0.85  # footprint area / its (u, v) box at least this: axis shapes
_PITCH_TAN = math.tan(math.radians(30.0))  # default pitch of gabled and hipped roofs
_SKILLION_TAN = math.tan(math.radians(15.0))
_MIN_RISE_m = 0.1  # roofs lower than this are flat
_RADIAL_STEP_rad = math.radians(10.0)  # widest wedge of a radial roof
_CURVE_STEPS = 10  # rings / strips approximating a quarter circle
_CELL_PAD_m = 0.01  # mesh cells reach this far past the footprint's box
_A_m = 6378137.0
_E2 = 6.69437999014e-3

Mode = Literal["axis", "radial", "inset"]


def roofShapeCodes(names: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """Roof shape codes from Overture / OSM ``roof_shape`` names.

    Args:
        names: Shape names (``None`` or unknown names are flat).

    Returns:
        :class:`RoofShape` codes.
    """
    values = np.asarray(names, dtype=object).ravel()
    return np.array(
        [_ALIASES.get(str(v).strip().lower(), RoofShape.FLAT) if v else 0 for v in values],
        dtype=np.uint8,
    )


@dataclass(frozen=True, eq=False)
class Roof:
    """A roof height field over one footprint (build with :func:`makeRoof`).

    Attributes:
        shape: Roof shape.
        rise_m: Height of the roof's highest point above the eaves.
        mode: How the shape is evaluated (``"axis"``, ``"radial"`` or ``"inset"``).
        origin_deg: ``(lon, lat)`` of the local plane's origin.
        scale_m: Metres per degree of longitude and latitude at the origin.
        footprint: Footprint in the local plane (metres).
        axisU: Unit vector across the ridge (downhill for skillion roofs).
        uRange: Footprint extent along ``axisU``.
        vRange: Footprint extent along the ridge.
        apex: Apex of radial shapes.
        wedgeAngles: Radial wedge boundaries (ascending, spanning 2 pi).
        wedgeA: Outline point at the start of each wedge, relative to the apex.
        wedgeB: Outline point at the end of each wedge, relative to the apex.
        maxInset_m: Largest distance from the outline (inset shapes).
    """

    shape: RoofShape
    rise_m: float
    mode: Mode
    origin_deg: tuple[float, float]
    scale_m: tuple[float, float]
    footprint: Any
    axisU: FloatArray
    uRange: tuple[float, float]
    vRange: tuple[float, float]
    apex: FloatArray
    wedgeAngles: FloatArray
    wedgeA: FloatArray
    wedgeB: FloatArray
    maxInset_m: float

    @property
    def curved(self) -> bool:
        """Whether the roof surface is curved (meshes share vertices for smooth shading)."""
        return self.shape in CURVED_SHAPES

    def toLocal(
        self, lon_deg: npt.ArrayLike, lat_deg: npt.ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        """Local plane coordinates (metres) of geographic positions."""
        lon0, lat0 = self.origin_deg
        kx, ky = self.scale_m
        x = (np.asarray(lon_deg, dtype=np.float64) - lon0) * kx
        y = (np.asarray(lat_deg, dtype=np.float64) - lat0) * ky
        return x, y

    def toLonLat(self, x_m: npt.ArrayLike, y_m: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Geographic ``(lon, lat)`` of local plane coordinates."""
        lon0, lat0 = self.origin_deg
        kx, ky = self.scale_m
        return (
            np.asarray(x_m, dtype=np.float64) / kx + lon0,
            np.asarray(y_m, dtype=np.float64) / ky + lat0,
        )

    def riseAt(self, lon_deg: npt.ArrayLike, lat_deg: npt.ArrayLike) -> FloatArray:
        """Roof height above the eaves at geographic positions."""
        return self.rise(*self.toLocal(lon_deg, lat_deg))

    def rise(self, x_m: npt.ArrayLike, y_m: npt.ArrayLike) -> FloatArray:
        """Roof height above the eaves at local plane positions.

        Args:
            x_m: Local east coordinate(s).
            y_m: Local north coordinate(s).

        Returns:
            Height(s) in ``[0, rise_m]``; points off the roof get the height of
            the nearest part of the field (0 beyond the outline for radial and
            inset shapes).
        """
        x = np.asarray(x_m, dtype=np.float64)
        y = np.asarray(y_m, dtype=np.float64)
        if self.mode == "axis":
            t = self._axisFraction(x, y)
        elif self.mode == "radial":
            rho = self.radialFraction(x, y)
            t = _radialProfile(self.shape, rho)
        else:
            t = self._insetFraction(x, y)
        result: FloatArray = self.rise_m * np.clip(t, 0.0, 1.0)
        return result

    def radialFraction(self, x_m: FloatArray, y_m: FloatArray) -> FloatArray:
        """Fraction of the way from the apex (0) to the outline (1), clamped to ``[0, 1]``."""
        dx = np.asarray(x_m, dtype=np.float64) - self.apex[0]
        dy = np.asarray(y_m, dtype=np.float64) - self.apex[1]
        theta0 = self.wedgeAngles[0]
        phi = np.mod(np.arctan2(dy, dx) - theta0, 2.0 * np.pi) + theta0
        k = np.clip(
            np.searchsorted(self.wedgeAngles, phi, side="right") - 1, 0, self.wedgeA.shape[0] - 1
        )
        a, b = self.wedgeA[k], self.wedgeB[k]
        det = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        alpha = (dx * b[..., 1] - dy * b[..., 0]) / det
        beta = (a[..., 0] * dy - a[..., 1] * dx) / det
        rho: FloatArray = np.clip(alpha + beta, 0.0, 1.0)
        return rho

    def _uv(self, x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
        ux, uy = self.axisU
        return x * ux + y * uy, -x * uy + y * ux

    def _axisFraction(self, x: FloatArray, y: FloatArray) -> FloatArray:
        u, v = self._uv(x, y)
        u0, u1 = self.uRange
        v0, v1 = self.vRange
        a, b = 0.5 * (u1 - u0), 0.5 * (v1 - v0)
        du = u - 0.5 * (u0 + u1)
        if self.shape == RoofShape.SKILLION:
            return (u1 - u) / (u1 - u0)
        if self.shape == RoofShape.ROUND:
            return np.sqrt(np.clip(1.0 - (du / a) ** 2, 0.0, 1.0))
        if self.shape == RoofShape.HIPPED:
            dv = v - 0.5 * (v0 + v1)
            return np.minimum(a - np.abs(du), b - np.abs(dv)) / min(a, b)
        return 1.0 - np.abs(du) / a  # gabled

    def _insetFraction(self, x: FloatArray, y: FloatArray) -> FloatArray:
        inside = shapely.contains_xy(self.footprint, x, y)
        d = shapely.distance(shapely.boundary(self.footprint), shapely.points(x, y))
        t = np.where(inside, np.clip(d / self.maxInset_m, 0.0, 1.0), 0.0)
        if self.shape in CURVED_SHAPES:
            return np.sqrt(np.clip(1.0 - (1.0 - t) ** 2, 0.0, 1.0))
        return t

    def cells(self) -> npt.NDArray[np.object_]:
        """Convex polygons (local plane) inside which the mesh is lifted linearly.

        Their edges follow the roof's creases (ridges, hips, rings), so a
        triangulation of the footprint clipped to them reproduces the shape.
        """
        if self.mode == "radial":
            return self._radialCells()
        if self.mode == "inset":
            return self._gridCells()
        return self._axisCells()

    def _uvPolygons(self, quads: list[list[tuple[float, float]]]) -> npt.NDArray[np.object_]:
        ux, uy = self.axisU
        polys = []
        for quad in quads:
            uv = np.asarray(quad, dtype=np.float64)
            x = uv[:, 0] * ux - uv[:, 1] * uy
            y = uv[:, 0] * uy + uv[:, 1] * ux
            polys.append(shapely.Polygon(np.column_stack([x, y])))
        return np.array(polys, dtype=object)

    def _axisCells(self) -> npt.NDArray[np.object_]:
        u0, u1 = self.uRange
        v0, v1 = self.vRange
        uc, a = 0.5 * (u0 + u1), 0.5 * (u1 - u0)
        pad = 1.0 + 0.01 * max(u1 - u0, v1 - v0)
        lo, hi = v0 - pad, v1 + pad
        if self.shape == RoofShape.SKILLION:
            breaks = [u0 - pad, u1 + pad]
        elif self.shape == RoofShape.GABLED:
            breaks = [u0 - pad, uc, u1 + pad]
        elif self.shape == RoofShape.ROUND:
            phi = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 2 * _CURVE_STEPS + 1)[1:-1]
            breaks = [u0 - pad, *(uc + a * np.sin(phi)).tolist(), u1 + pad]
        else:
            # Pad so outline edges lying on the box stay inside the facets.
            e = _CELL_PAD_m
            return self._uvPolygons(_hippedFacets(u0 - e, u1 + e, v0 - e, v1 + e))
        strips = [
            [(p, lo), (q, lo), (q, hi), (p, hi)]
            for p, q in zip(breaks[:-1], breaks[1:], strict=True)
        ]
        return self._uvPolygons(strips)

    def _gridCells(self) -> npt.NDArray[np.object_]:
        u0, u1 = self.uRange
        v0, v1 = self.vRange
        step = float(np.clip(self.maxInset_m / 6.0, 1.0, 4.0))
        us = np.arange(u0 - _CELL_PAD_m, u1 + step, step)
        vs = np.arange(v0 - _CELL_PAD_m, v1 + step, step)
        quads = [
            [(p, q), (p + step, q), (p + step, q + step), (p, q + step)]
            for p in us[:-1]
            for q in vs[:-1]
        ]
        return self._uvPolygons(quads)

    def _radialCells(self) -> npt.NDArray[np.object_]:
        rings = _radialRings(self.shape)
        far = 10.0 * max(np.ptp(self.uRange), np.ptp(self.vRange), 1.0)
        cells = []
        c = self.apex
        for a, b in zip(self.wedgeA, self.wedgeB, strict=True):
            for r0, r1 in zip(rings[:-1], rings[1:], strict=True):
                pts = (
                    [c + r0 * a, c + r0 * b, c + r1 * b, c + r1 * a]
                    if r0 > 0
                    else [c, c + r1 * b, c + r1 * a]
                )
                cells.append(shapely.Polygon(pts))
            # Beyond the first outline crossing (non-star-shaped footprints): flat.
            ua, ub = a / np.linalg.norm(a), b / np.linalg.norm(b)
            cells.append(shapely.Polygon([c + a, c + b, c + far * ub, c + far * ua]))
        return np.array(cells, dtype=object)

    def triangles(self) -> FloatArray:
        """The footprint triangulated along :meth:`cells`, as ``(m, 3, 2)`` local corners."""
        pieces = shapely.intersection(self.footprint, self.cells())
        polys = shapely.get_parts(shapely.get_parts(pieces))
        polys = polys[(shapely.get_type_id(polys) == 3) & (shapely.area(polys) > 1e-9)]
        if polys.size == 0:
            return np.zeros((0, 3, 2))
        tris = shapely.get_parts(shapely.constrained_delaunay_triangles(polys))
        coords: FloatArray = shapely.get_coordinates(tris).reshape(-1, 4, 2)[:, :3]
        a, b, c = coords[:, 0], coords[:, 1], coords[:, 2]
        cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (
            c[:, 0] - a[:, 0]
        )
        return coords[np.abs(cross) > 1e-9]  # no slivers

    def outlineSegments(self) -> tuple[FloatArray, FloatArray]:
        """Outline edges split where they cross :meth:`cells`, as ``(start, end)`` local points.

        Wall tops sampled at these points meet the roof mesh's outline vertices.
        """
        # Union dissolves outline pieces shared by two cells (an outline on a cell edge).
        lines = shapely.union_all(
            shapely.intersection(shapely.boundary(self.footprint), self.cells())
        )
        parts = shapely.get_parts(shapely.get_parts(lines))
        parts = parts[shapely.get_type_id(parts) == 1]
        if parts.size == 0:
            return np.zeros((0, 2)), np.zeros((0, 2))
        coords, owner = shapely.get_coordinates(parts, return_index=True)
        a = np.flatnonzero(owner[:-1] == owner[1:])
        start, end = coords[a], coords[a + 1]
        keep = np.linalg.norm(end - start, axis=1) > 1e-6
        return start[keep], end[keep]


def makeRoof(
    footprint_deg: Any,
    shape: int,
    available_m: float,
    roofHeight_m: float = math.nan,
    direction_deg: float = math.nan,
    orientation: str | None = None,
) -> Roof | None:
    """Build a building's roof from its mapped attributes.

    Default roof heights (when ``roofHeight_m`` is missing): dome and pyramid,
    the radius of a circle of the footprint's area (onion 1.5x); round, half the
    footprint's width; gabled and hipped, a 30 degree pitch over half the width;
    skillion, a 15 degree pitch over the width. Inset shapes use the largest
    distance to the outline in place of half the width.

    Args:
        footprint_deg: Footprint polygon in lon/lat.
        shape: :class:`RoofShape` code.
        available_m: Vertical extent of the part (top minus bottom).
        roofHeight_m: Mapped roof height, NaN if unknown.
        direction_deg: Bearing the roof slope faces (clockwise from north), NaN if unknown.
        orientation: ``"along"`` or ``"across"`` the longest side, if mapped.

    Returns:
        The roof, or ``None`` when it is flat (or too low to matter).
    """
    code = RoofShape(int(shape))
    if code == RoofShape.FLAT or not available_m > _MIN_RISE_m:
        return None
    polygon = _largestPolygon(footprint_deg)
    if polygon is None:
        return None
    lon0, lat0 = (float(v) for v in shapely.get_coordinates(shapely.centroid(polygon))[0])
    scale = _metresPerDegree(lat0)
    footprint = shapely.transform(polygon, lambda xy: (xy - (lon0, lat0)) * scale)
    axisU = _acrossAxis(footprint, direction_deg, orientation)
    ext = shapely.get_coordinates(shapely.get_exterior_ring(footprint))
    u = ext @ axisU
    v = ext @ np.array([-axisU[1], axisU[0]])
    uRange, vRange = (float(u.min()), float(u.max())), (float(v.min()), float(v.max()))
    a, b = 0.5 * np.ptp(u), 0.5 * np.ptp(v)
    area = float(shapely.area(footprint))
    fill = area / max(4.0 * a * b, 1e-9)
    maxInset = float(shapely.distance(polylabel(footprint, 0.1), shapely.boundary(footprint)))
    apex, angles, wedgeA, wedgeB = _radialWedges(footprint)

    radial = code in (RoofShape.PYRAMIDAL, RoofShape.DOME, RoofShape.ONION)
    mode: Mode
    if radial and angles.size:
        mode = "radial"
    elif code in (RoofShape.SKILLION, RoofShape.GABLED) or (
        code in (RoofShape.ROUND, RoofShape.HIPPED) and fill >= RECTANGULAR_FILL
    ):
        mode = "axis"
    else:
        mode = "inset"
    if min(a, b) <= 0.0 or maxInset <= 0.0:
        return None

    rise = _defaultRise(code, mode, area, a, b, maxInset)
    if np.isfinite(roofHeight_m) and roofHeight_m > 0.0:
        rise = float(roofHeight_m)
    elif code in _PITCHED_SHAPES or code == RoofShape.ROUND:
        rise = min(rise, 0.5 * available_m)
    rise = min(rise, available_m)
    if rise < _MIN_RISE_m:
        return None
    return Roof(
        shape=code,
        rise_m=float(rise),
        mode=mode,
        origin_deg=(lon0, lat0),
        scale_m=(float(scale[0]), float(scale[1])),
        footprint=footprint,
        axisU=axisU,
        uRange=uRange,
        vRange=vRange,
        apex=apex,
        wedgeAngles=angles,
        wedgeA=wedgeA,
        wedgeB=wedgeB,
        maxInset_m=maxInset,
    )


def _defaultRise(
    code: RoofShape, mode: Mode, area: float, a: float, b: float, inset: float
) -> float:
    if mode == "radial":
        radius = math.sqrt(area / math.pi)
        return 1.5 * radius if code == RoofShape.ONION else radius
    half = inset if mode == "inset" else min(a, b) if code == RoofShape.HIPPED else a
    if code == RoofShape.SKILLION:
        return 2.0 * a * _SKILLION_TAN
    if code in CURVED_SHAPES:
        return half
    return half * _PITCH_TAN


def _radialProfile(code: RoofShape, rho: FloatArray) -> FloatArray:
    if code == RoofShape.PYRAMIDAL:
        return 1.0 - rho
    dome = np.sqrt(np.clip(1.0 - rho**2, 0.0, 1.0))
    if code == RoofShape.ONION:
        return 0.7 * dome + 0.3 * np.clip(1.0 - rho / 0.3, 0.0, 1.0)
    return dome


def _radialRings(code: RoofShape) -> FloatArray:
    if code == RoofShape.PYRAMIDAL:
        return np.array([0.0, 1.0])
    rings = np.sin(np.linspace(0.0, 0.5 * np.pi, _CURVE_STEPS + 1))
    if code == RoofShape.ONION:
        rings = np.union1d(rings, [0.3])
    return rings


def _hippedFacets(u0: float, u1: float, v0: float, v1: float) -> list[list[tuple[float, float]]]:
    """The four planes of a hipped roof over the box (its straight skeleton)."""
    a, b = 0.5 * (u1 - u0), 0.5 * (v1 - v0)
    uc, vc = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
    if b >= a:
        lo, hi = (uc, vc - (b - a)), (uc, vc + (b - a))
        return [
            [(u0, v0), (u0, v1), hi, lo],
            [(u1, v0), lo, hi, (u1, v1)],
            [(u0, v0), lo, (u1, v0)],
            [(u0, v1), (u1, v1), hi],
        ]
    lo, hi = (uc - (a - b), vc), (uc + (a - b), vc)
    return [
        [(u0, v0), (u1, v0), hi, lo],
        [(u0, v1), lo, hi, (u1, v1)],
        [(u0, v0), lo, (u0, v1)],
        [(u1, v0), (u1, v1), hi],
    ]


def _metresPerDegree(lat_deg: float) -> FloatArray:
    phi = math.radians(lat_deg)
    w = math.sqrt(1.0 - _E2 * math.sin(phi) ** 2)
    n = _A_m / w
    m = _A_m * (1.0 - _E2) / w**3
    return np.array([math.radians(1.0) * n * math.cos(phi), math.radians(1.0) * m])


def _largestPolygon(geom: Any) -> Any:
    parts = shapely.get_parts(geom)
    parts = parts[shapely.get_type_id(parts) == 3]
    if parts.size == 0:
        return None
    return parts[int(np.argmax(shapely.area(parts)))]


def _acrossAxis(footprint: Any, direction_deg: float, orientation: str | None) -> FloatArray:
    if np.isfinite(direction_deg):
        b = math.radians(direction_deg)
        return np.array([math.sin(b), math.cos(b)])
    rect = shapely.get_coordinates(shapely.oriented_envelope(footprint))
    edges = np.diff(rect[:3], axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    if lengths.size < 2 or lengths.min() <= 0.0:
        return np.array([1.0, 0.0])
    longSide: FloatArray = edges[int(np.argmax(lengths))] / lengths.max()
    along = np.array([-longSide[1], longSide[0]])  # across the long side: ridge along it
    if orientation is not None and str(orientation).lower() == "across":
        return longSide
    return along


def _radialWedges(footprint: Any) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Apex and wedges between rays through every outline vertex (and every 10 degrees)."""
    apex = shapely.get_coordinates(shapely.centroid(footprint))[0]
    none = (apex, np.zeros(0), np.zeros((0, 2)), np.zeros((0, 2)))
    if not shapely.contains_xy(footprint, *apex):
        return none
    ring = shapely.get_exterior_ring(footprint)
    pts = shapely.get_coordinates(ring)[:-1] - apex
    angles = np.unique(np.round(np.mod(np.arctan2(pts[:, 1], pts[:, 0]), 2.0 * np.pi), 12))
    gaps = np.diff(np.append(angles, angles[0] + 2.0 * np.pi))
    filled = [angles]
    for start, gap in zip(angles, gaps, strict=True):
        n = int(math.ceil(gap / _RADIAL_STEP_rad))
        if n > 1:
            filled.append(start + gap * np.arange(1, n) / n)
    theta = np.sort(np.mod(np.concatenate(filled), 2.0 * np.pi))
    far = 4.0 * float(np.max(np.linalg.norm(pts, axis=1)))
    hits = np.array([_rayHit(ring, apex, t, far) for t in theta])
    if not np.all(np.isfinite(hits)) or np.min(np.linalg.norm(hits, axis=1)) < 1e-6:
        return none
    bounds = np.append(theta, theta[0] + 2.0 * np.pi)
    return apex, bounds, hits, np.roll(hits, -1, axis=0)


def _rayHit(ring: Any, apex: FloatArray, theta: float, far: float) -> FloatArray:
    d = np.array([math.cos(theta), math.sin(theta)])
    ray = shapely.LineString([apex, apex + far * d])
    pts = shapely.get_coordinates(shapely.intersection(ray, ring))
    if pts.shape[0] == 0:
        return np.full(2, np.nan)
    rel = pts - apex
    dist = rel @ d
    rel, dist = rel[dist > 1e-9], dist[dist > 1e-9]
    if dist.size == 0:
        return np.full(2, np.nan)
    return d * float(dist.min())
