"""Online data acquisition for ``map3d prepare`` (requires the ``prepare`` extra).

This package root needs no extra, so the CLI can import its defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["DEFAULT_BUILDINGS_YEAR", "defaultCacheDir"]

DEFAULT_BUILDINGS_YEAR = 2023  # latest Google Open Buildings 2.5D Temporal year


def defaultCacheDir(packDir: str | os.PathLike[str]) -> Path:
    """Default download/work cache for a pack: ``<packDir>.cache`` beside it.

    Args:
        packDir: Pack directory.

    Returns:
        The cache directory path.
    """
    pack = Path(packDir)
    return pack.with_name(pack.name + ".cache")
