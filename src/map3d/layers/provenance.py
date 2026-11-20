"""Per-node provenance codes for the ``heightSource`` layer (SPEC section 6.1)."""

from __future__ import annotations

from enum import IntEnum

__all__ = ["HeightSource", "heightSourceTable"]


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


def heightSourceTable() -> dict[str, int]:
    """Name-to-code table, recorded in ``scene.json``.

    Returns:
        Mapping of enum names to their uint8 codes.
    """
    return {member.name: int(member.value) for member in HeightSource}
