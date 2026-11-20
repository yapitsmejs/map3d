"""Scene request: the user-facing inputs and their strict validation (SPEC section 4)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

__all__ = ["SceneRequest", "VerticalDatum"]

VerticalDatum = Literal["msl", "ellipsoid"]

_VERTICAL_DATUMS: tuple[str, ...] = ("msl", "ellipsoid")
_RATIO_REL_TOL = 1e-9


@dataclass(frozen=True, eq=False)
class SceneRequest:
    """Inputs describing one scene. Validated on construction.

    Attributes:
        sceneCenter_deg: ``(lat, lon)`` WGS84 geodetic coordinates of the grid center.
        gridSpacing_m: Node spacing, identical on both grid axes.
        gridAxes_ENU: 3x3 matrix whose **rows** are the grid x, y, z axes expressed
            as unit vectors in the scene-center ENU frame. Stored read-only.
        swathHorizontal_m: Swath extent along grid x.
        swathVertical_m: Swath extent along grid y (an in-plane extent, not altitude).
        verticalDatum: ``"msl"`` (EGM2008 geoid) or ``"ellipsoid"`` (WGS84).
        maxDeckLevels: Number of elevated-structure levels stored in the grid.
        axesTol: Tolerance for the orthonormality / handedness / z-Up checks.

    Raises:
        ValueError: On any invalid input, with a message naming the problem.
    """

    sceneCenter_deg: tuple[float, float]
    gridSpacing_m: float
    gridAxes_ENU: npt.NDArray[np.float64]
    swathHorizontal_m: float
    swathVertical_m: float
    verticalDatum: VerticalDatum = "msl"
    maxDeckLevels: int = 5
    axesTol: float = 1e-6
    nCols: int = field(init=False)
    nRows: int = field(init=False)

    def __post_init__(self) -> None:
        self._validateCenter()
        if not (math.isfinite(self.axesTol) and self.axesTol > 0):
            raise ValueError(f"axesTol must be a positive finite number, got {self.axesTol!r}")
        object.__setattr__(self, "gridAxes_ENU", _validateAxes(self.gridAxes_ENU, self.axesTol))
        _requirePositiveFinite("gridSpacing_m", self.gridSpacing_m)
        nCols = _nodeCount("swathHorizontal_m", self.swathHorizontal_m, self.gridSpacing_m)
        nRows = _nodeCount("swathVertical_m", self.swathVertical_m, self.gridSpacing_m)
        object.__setattr__(self, "nCols", nCols)
        object.__setattr__(self, "nRows", nRows)
        if self.verticalDatum not in _VERTICAL_DATUMS:
            raise ValueError(
                f"verticalDatum must be one of {_VERTICAL_DATUMS}, got {self.verticalDatum!r}"
            )
        if (
            isinstance(self.maxDeckLevels, bool)
            or not isinstance(self.maxDeckLevels, int)
            or self.maxDeckLevels < 0
        ):
            raise ValueError(
                f"maxDeckLevels must be a non-negative integer, got {self.maxDeckLevels!r}"
            )

    def _validateCenter(self) -> None:
        if len(self.sceneCenter_deg) != 2:
            raise ValueError(
                f"sceneCenter_deg must be (lat, lon), got {len(self.sceneCenter_deg)} values"
            )
        lat_deg, lon_deg = (float(v) for v in self.sceneCenter_deg)
        if not (math.isfinite(lat_deg) and -90.0 < lat_deg < 90.0):
            raise ValueError(
                f"latitude must be finite and strictly within (-90, 90), got {lat_deg}"
            )
        if not (math.isfinite(lon_deg) and -180.0 <= lon_deg <= 180.0):
            raise ValueError(f"longitude must be finite and within [-180, 180], got {lon_deg}")
        object.__setattr__(self, "sceneCenter_deg", (lat_deg, lon_deg))


def _requirePositiveFinite(name: str, value: float) -> None:
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")


def _nodeCount(name: str, swath_m: float, spacing_m: float) -> int:
    _requirePositiveFinite(name, swath_m)
    ratio = swath_m / spacing_m
    count = round(ratio)
    if count < 1 or abs(ratio - count) > _RATIO_REL_TOL * max(1.0, ratio):
        raise ValueError(
            f"{name}={swath_m!r} must be a whole multiple of gridSpacing_m={spacing_m!r} "
            f"(ratio {ratio!r})"
        )
    return int(count)


def _validateAxes(axes: npt.ArrayLike, tol: float) -> npt.NDArray[np.float64]:
    r = np.array(axes, dtype=np.float64)
    if r.shape != (3, 3):
        raise ValueError(f"gridAxes_ENU must be a 3x3 matrix (rows = axes), got shape {r.shape}")
    if not np.all(np.isfinite(r)):
        raise ValueError("gridAxes_ENU must contain only finite values")
    lengths = np.linalg.norm(r, axis=1)
    for name, length in zip("xyz", lengths, strict=True):
        if abs(length - 1.0) > tol:
            raise ValueError(f"gridAxes_ENU {name} axis must have unit length, got {length!r}")
    for (i, j), names in zip(((0, 1), (0, 2), (1, 2)), ("xy", "xz", "yz"), strict=True):
        dot = float(r[i] @ r[j])
        if abs(dot) > tol:
            raise ValueError(
                f"gridAxes_ENU axes {names[0]} and {names[1]} must be orthogonal (dot = {dot!r})"
            )
    det = float(np.linalg.det(r))
    if abs(det - 1.0) > tol:
        raise ValueError(f"gridAxes_ENU must be right-handed (det = +1), got det = {det!r}")
    tilt = float(np.linalg.norm(r[2] - np.array([0.0, 0.0, 1.0])))
    if tilt > tol:
        raise ValueError(
            f"gridAxes_ENU z axis must point Up (0, 0, 1) in v1 (tilted grids are not "
            f"supported); |z - Up| = {tilt!r}"
        )
    r.setflags(write=False)
    return r
