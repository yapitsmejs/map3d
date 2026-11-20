"""Offline data packs: catalogue, manifest, ingest and reading (SPEC section 9)."""

from __future__ import annotations

from map3d.pack.ingest import ingestFiles
from map3d.pack.manifest import Manifest
from map3d.pack.reader import Pack, RasterWindow, VectorSelection
from map3d.pack.synthetic import SyntheticScene, makeSyntheticPack

__all__ = [
    "Manifest",
    "Pack",
    "RasterWindow",
    "SyntheticScene",
    "VectorSelection",
    "ingestFiles",
    "makeSyntheticPack",
]
