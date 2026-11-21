"""Per-node provenance codes for the ``heightSource`` layer (SPEC section 6.1)."""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import numpy.typing as npt

__all__ = [
    "HeightSource",
    "HEIGHT_UNCERTAINTY_m",
    "heightSourceTable",
    "heightUncertaintyFor",
    "heightUncertaintyTable",
]


class HeightSource(IntEnum):
    """What set a node's ``surfaceHeight`` (stored as uint8)."""

    NONE = 0
    GROUND = 1
    AUTHORITATIVE_HEIGHT = 2
    AUTHORITATIVE_FLOORS = 3
    EXPLICIT_HEIGHT = 4
    BUILDING_HEIGHT_RASTER = 5
    FLOORS = 6
    DEFAULT_HEIGHT = 7
    CANOPY = 8
    DECK_ESTIMATED = 9


def heightSourceTable() -> dict[str, int]:
    """Name-to-code table, recorded in ``scene.json``.

    Returns:
        Mapping of enum names to their uint8 codes.
    """
    return {member.name: int(member.value) for member in HeightSource}


#: Conservative 1-sigma-ish error per height source, in metres.
HEIGHT_UNCERTAINTY_m: dict[HeightSource, float] = {
    HeightSource.NONE: float("nan"),
    HeightSource.GROUND: 1.0,
    HeightSource.AUTHORITATIVE_HEIGHT: 1.0,
    HeightSource.AUTHORITATIVE_FLOORS: 1.5,
    HeightSource.EXPLICIT_HEIGHT: 3.0,
    HeightSource.BUILDING_HEIGHT_RASTER: 4.0,
    HeightSource.FLOORS: 3.0,
    HeightSource.DEFAULT_HEIGHT: 10.0,
    HeightSource.CANOPY: 2.0,
    HeightSource.DECK_ESTIMATED: 4.0,
}


def heightUncertaintyFor(codes: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Uncertainty of each node's height, looked up by its source code.

    Args:
        codes: ``heightSource`` codes (uint8).

    Returns:
        Uncertainties in metres (NaN for unknown codes).
    """
    lookup = np.full(256, np.nan)
    for source, uncertainty in HEIGHT_UNCERTAINTY_m.items():
        lookup[int(source)] = uncertainty
    return lookup[np.asarray(codes, dtype=np.uint8)]


def heightUncertaintyTable() -> dict[str, float]:
    """Name-to-uncertainty table, recorded in ``scene.json``.

    Returns:
        Mapping of height-source names to their uncertainties in metres.
    """
    return {k.name: v for k, v in HEIGHT_UNCERTAINTY_m.items() if k is not HeightSource.NONE}
