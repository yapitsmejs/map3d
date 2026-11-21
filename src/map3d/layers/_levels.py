"""Stacked elevated levels per node: decks and raised building parts (SPEC section 6).

Elevated structures with open space beneath them, such as bridge decks and
raised building parts (a sky bridge, a cantilevered sky park), are reported
as levels: per node, the top and underside of each, lowest first.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["LevelEntries", "stackLevels"]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class LevelEntries:
    """Elevated surfaces at chunk nodes, one entry per (node, structure).

    Attributes:
        flat: Flat node index within the chunk (``row * cols + col``).
        top: Orthometric top.
        bottom: Orthometric underside.
    """

    flat: npt.NDArray[np.int64]
    top: FloatArray
    bottom: FloatArray

    @classmethod
    def empty(cls) -> LevelEntries:
        """No entries."""
        return cls(np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0))

    @classmethod
    def concatenate(cls, parts: Sequence[LevelEntries]) -> LevelEntries:
        """Entries of all ``parts``."""
        if not parts:
            return cls.empty()
        return cls(
            np.concatenate([p.flat for p in parts]).astype(np.int64),
            np.concatenate([p.top for p in parts]),
            np.concatenate([p.bottom for p in parts]),
        )


def stackLevels(
    entries: LevelEntries, shape: tuple[int, int], maxLevels: int, mergeTolerance_m: float
) -> tuple[FloatArray, FloatArray, int]:
    """Sort elevated surfaces into per-node levels, lowest first.

    Per node, entries are sorted by top; a top at least ``mergeTolerance_m``
    above the previous one starts a new level, so overlapping pieces of one
    structure don't create phantom levels. A level's top is its highest
    entry's top and its underside the lowest entry's underside.

    Args:
        entries: Elevated surfaces.
        shape: Chunk shape ``(rows, cols)``.
        maxLevels: ``K``, the number of levels to store.
        mergeTolerance_m: Tops closer than this merge into one level.

    Returns:
        ``(tops, bottoms, overflowNodes)``: ``(K, rows, cols)`` arrays (NaN
        where absent) and the number of nodes with more than ``K`` levels.
    """
    tops = np.full((maxLevels, *shape), np.nan)
    bottoms = np.full((maxLevels, *shape), np.nan)
    if entries.flat.size == 0:
        return tops, bottoms, 0
    order = np.lexsort((entries.top, entries.flat))
    flat, top, bottom = entries.flat[order], entries.top[order], entries.bottom[order]
    isFirst = np.ones(flat.size, dtype=bool)
    isFirst[1:] = flat[1:] != flat[:-1]
    newLevel = isFirst.copy()
    newLevel[1:] |= (top[1:] - top[:-1]) >= mergeTolerance_m
    groupStart = np.maximum.accumulate(np.where(isFirst, np.arange(flat.size), 0))
    levelCum = np.cumsum(newLevel)
    level = levelCum - levelCum[groupStart]
    stored = level < maxLevels
    for k in range(maxLevels):
        sel = stored & (level == k)
        np.fmax.at(tops[k].ravel(), flat[sel], top[sel])
        np.fmin.at(bottoms[k].ravel(), flat[sel], bottom[sel])
    overflow = int(np.unique(flat[level >= maxLevels]).size)
    return tops, bottoms, overflow
