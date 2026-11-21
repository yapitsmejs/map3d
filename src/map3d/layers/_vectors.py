"""Helpers shared by the vector layers: Arrow columns, overlap and outline sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import shapely

if TYPE_CHECKING:
    from map3d.layers.terrain import GroundModel

FloatArray = npt.NDArray[np.float64]

DUPLICATE_OVERLAP = 0.5  # a feature this much covered by accepted ones is a duplicate
_OUTLINE_STEP_deg = 5e-5  # ~5.5 m: outline densification for ground sampling


def objectColumn(table: Any, name: str) -> npt.NDArray[np.object_]:
    """An Arrow table column as an object array (nulls are ``None``)."""
    return np.array(table.column(name).to_pylist(), dtype=object)


def textColumn(table: Any, name: str) -> npt.NDArray[np.object_]:
    """An Arrow table column as an object array of ``str``."""
    return np.array([str(v) for v in table.column(name).to_pylist()], dtype=object)


def floatColumn(table: Any, name: str) -> FloatArray:
    """An Arrow table column as float64 (nulls are NaN)."""
    return np.array(
        [np.nan if v is None else float(v) for v in table.column(name).to_pylist()],
        dtype=np.float64,
    )


def mostlyCovered(
    candidates: npt.NDArray[np.object_], cover: npt.NDArray[np.object_]
) -> npt.NDArray[np.bool_]:
    """Whether more than :data:`DUPLICATE_OVERLAP` of each candidate's area is under ``cover``.

    Args:
        candidates: Polygons to test.
        cover: Polygons already accepted (same coordinate system).

    Returns:
        One flag per candidate.
    """
    candIdx, covIdx = shapely.STRtree(cover).query(candidates, predicate="intersects")
    overlap = np.zeros(candidates.size)
    if candIdx.size:
        areas = shapely.area(shapely.intersection(candidates[candIdx], cover[covIdx]))
        np.add.at(overlap, candIdx, areas)
    covered: npt.NDArray[np.bool_] = overlap > DUPLICATE_OVERLAP * shapely.area(candidates)
    return covered


def outlineGround(
    geoms: npt.NDArray[np.object_], ground: GroundModel
) -> tuple[FloatArray, npt.NDArray[np.int64]]:
    """Ground sampled along each lon/lat polygon's densified outline.

    Args:
        geoms: Polygons in lon/lat.
        ground: Ground model.

    Returns:
        ``(orthometric heights, owner index)`` per outline sample.
    """
    if geoms.size == 0:
        return np.zeros(0), np.zeros(0, dtype=np.int64)
    outline = shapely.segmentize(shapely.boundary(geoms), _OUTLINE_STEP_deg)
    coords, owner = shapely.get_coordinates(outline, return_index=True)
    return ground.orthometricAt(coords[:, 1], coords[:, 0]), owner
