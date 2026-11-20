"""Build offline, georeferenced 3D scene grids and 3D Tiles meshes from open data."""

from __future__ import annotations

import logging

__version__ = "0.1.0"

from map3d.frames import LocalFrame
from map3d.geoid import GeoidGrid
from map3d.gridGeometry import GridGeometry
from map3d.request import SceneRequest
from map3d.scene import Scene, buildScene

# Library convention: never emit log output unless the application configures logging.
logging.getLogger("map3d").addHandler(logging.NullHandler())

__all__ = [
    "GeoidGrid",
    "GridGeometry",
    "LocalFrame",
    "Scene",
    "SceneRequest",
    "__version__",
    "buildScene",
]
