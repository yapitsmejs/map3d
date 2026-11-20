"""``.npz`` + JSON sidecar writer/reader (SPEC section 6.2).

Layers are stored under their names, and feature-table columns under
``features.<column>``. Everything loads with ``allow_pickle=False``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from map3d.scene import Scene

__all__ = ["readNpz", "writeNpz"]

_FEATURE_PREFIX = "features."


def writeNpz(
    scene: Scene, path: str | os.PathLike[str], compress: bool = False
) -> tuple[Path, Path]:
    """Write a scene as ``<path>.npz`` plus a ``<path>.json`` sidecar.

    Args:
        scene: The scene.
        path: Output path prefix; a ``.npz`` suffix is optional.
        compress: Use zlib compression (smaller, slower).

    Returns:
        The ``.npz`` and ``.json`` paths.
    """
    base = Path(path)
    if base.suffix == ".npz":
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    npzPath = base.parent / (base.name + ".npz")
    jsonPath = base.parent / (base.name + ".json")
    arrays: dict[str, npt.NDArray[Any]] = dict(scene.layers)
    arrays.update({_FEATURE_PREFIX + k: v for k, v in scene.features.items()})
    save: Any = np.savez_compressed if compress else np.savez
    save(npzPath, **arrays)
    metadata = {**scene.metadata, "files": {"npz": npzPath.name}}
    jsonPath.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return npzPath, jsonPath


def readNpz(
    npzPath: str | os.PathLike[str],
) -> tuple[dict[str, npt.NDArray[Any]], dict[str, npt.NDArray[Any]], dict[str, Any]]:
    """Read a scene written by :func:`writeNpz`.

    Args:
        npzPath: Path to the ``.npz`` file (the ``.json`` must sit beside it).

    Returns:
        ``(layers, features, metadata)``.
    """
    path = Path(npzPath)
    layers: dict[str, npt.NDArray[Any]] = {}
    features: dict[str, npt.NDArray[Any]] = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            if key.startswith(_FEATURE_PREFIX):
                features[key[len(_FEATURE_PREFIX) :]] = data[key]
            else:
                layers[key] = data[key]
    metadata: dict[str, Any] = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return layers, features, metadata
