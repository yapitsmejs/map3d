"""Scene building: request + pack -> multi-layer grid (SPEC sections 5-6).

All layer maths happens in orthometric height (MSL, EGM2008) and is converted at
the end of each chunk:

- ``*_datum_m``: ``H`` for ``verticalDatum="msl"``, else ``H + N`` (ellipsoidal).
- ``*_ENU_m``: Up coordinate of the column point at ellipsoidal height ``H + N``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from map3d import __version__
from map3d.gridGeometry import GridGeometry
from map3d.layers.buildings import BuildingSet, loadBuildings, rasterizeBuildings
from map3d.layers.columns import ChunkColumns
from map3d.layers.provenance import HeightSource, heightSourceTable
from map3d.layers.terrain import GroundModel
from map3d.pack.manifest import MANIFEST_NAME, SourceEntry, fileSha256
from map3d.pack.reader import Pack
from map3d.pack.registry import sourcesForRole
from map3d.request import SceneRequest

__all__ = ["LAYER_INFO", "Scene", "buildScene"]

_log = logging.getLogger("map3d")

FloatArray = npt.NDArray[np.float64]

LAYER_INFO: dict[str, dict[str, str]] = {
    "groundHeight_datum_m": {
        "units": "m",
        "description": "Bare-earth ground height above the vertical datum (cell-max).",
    },
    "groundHeight_ENU_m": {
        "units": "m",
        "description": "Bare-earth ground: Up coordinate in the scene-center ENU frame (cell-max).",
    },
    "surfaceHeight_datum_m": {
        "units": "m",
        "description": "Top surface (terrain + buildings) above the vertical datum (cell-max DSM).",
    },
    "surfaceHeight_ENU_m": {
        "units": "m",
        "description": "Top surface: Up coordinate in the scene-center ENU frame (cell-max DSM).",
    },
    "buildingHeight_m": {
        "units": "m",
        "description": "Building roof height above the node's ground; NaN where there is no building.",
    },
    "featureId": {
        "units": "1",
        "description": "Id of the top-most feature (indexes the features table); 0 = none.",
    },
    "heightSource": {
        "units": "1",
        "description": "Code of the source that set surfaceHeight (see heightSourceCodes).",
    },
    "nativeResolution_m": {
        "units": "m",
        "description": "Native resolution of that source; 0 for vector-derived heights.",
    },
}

_ARRAY_ORDER = "[row, col]; row 0 at -y, column 0 at -x of the grid frame; plot with origin='lower'"


@dataclass(eq=False)
class Scene:
    """A built scene grid.

    Attributes:
        request: The request it was built from.
        geometry: Grid geometry.
        layers: Layer name to array (see :data:`LAYER_INFO`).
        features: Columns of the features table.
        metadata: Everything written to ``scene.json``.
    """

    request: SceneRequest
    geometry: GridGeometry
    layers: dict[str, npt.NDArray[Any]]
    features: dict[str, npt.NDArray[Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def warnings(self) -> list[str]:
        """Warnings raised while building (also in ``metadata``)."""
        result: list[str] = self.metadata["warnings"]
        return result

    def writeNpz(self, path: str | os.PathLike[str], compress: bool = False) -> tuple[Path, Path]:
        """Write ``<path>.npz`` and ``<path>.json``.

        Args:
            path: Output path prefix (``.npz`` suffix optional).
            compress: Use zlib compression.

        Returns:
            The ``.npz`` and ``.json`` paths.
        """
        from map3d.io.npzWriter import writeNpz

        return writeNpz(self, path, compress=compress)

    def writeGeoTiff(self, outDir: str | os.PathLike[str]) -> list[Path]:
        """Write one GeoTIFF per layer into ``outDir``.

        Args:
            outDir: Output directory.

        Returns:
            Paths written.
        """
        from map3d.io.geotiffWriter import writeGeoTiff

        return writeGeoTiff(self, outDir)


def buildScene(
    request: SceneRequest, pack: str | os.PathLike[str] | Pack, chunkSize: int = 1024
) -> Scene:
    """Build the scene grid for a request from an offline pack.

    Args:
        request: Validated scene request.
        pack: Pack directory or opened pack.
        chunkSize: Rows/columns per processing chunk.

    Returns:
        The scene.

    Raises:
        ValueError: If the pack cannot serve the scene (outside coverage, or a
            required role is missing).
    """
    opened = pack if isinstance(pack, Pack) else Pack.open(pack)
    geometry = GridGeometry.fromRequest(request)
    warnings = opened.checkCoverage(geometry.footprintBounds_deg())
    ground = GroundModel.load(opened, geometry)
    buildings = loadBuildings(opened, geometry, ground)
    warnings += _resolutionWarnings(opened, request.gridSpacing_m)
    for message in warnings:
        _log.warning(message)

    layers = _allocate(geometry.shape)
    for rows, cols in geometry.iterChunks(chunkSize, chunkSize):
        _buildChunk(layers, request, geometry, ground, buildings, rows, cols)

    metadata = _metadata(request, geometry, ground, opened, warnings, layers)
    return Scene(request, geometry, layers, buildings.featureTable(), metadata)


def _allocate(shape: tuple[int, int]) -> dict[str, npt.NDArray[Any]]:
    layers: dict[str, npt.NDArray[Any]] = {}
    for name in LAYER_INFO:
        if name == "featureId":
            layers[name] = np.zeros(shape, dtype=np.int64)
        elif name == "heightSource":
            layers[name] = np.zeros(shape, dtype=np.uint8)
        else:
            layers[name] = np.full(shape, np.nan, dtype=np.float32)
    return layers


def _buildChunk(
    layers: dict[str, npt.NDArray[Any]],
    request: SceneRequest,
    geometry: GridGeometry,
    ground: GroundModel,
    buildings: BuildingSet,
    rows: slice,
    cols: slice,
) -> None:
    columns = ChunkColumns(geometry, rows, cols)
    groundH = ground.cellMaxOrthometric(columns, geometry.gridSpacing_m)
    ids = rasterizeBuildings(buildings, geometry, rows, cols)
    hasBuilding = ids > 0
    index = np.maximum(ids - 1, 0)
    roofH = np.where(
        hasBuilding, buildings.roofOrthometric_m[index] if len(buildings) else np.nan, np.nan
    )
    surfaceH = np.fmax(groundH, roofH)
    groundKnown = np.isfinite(groundH)
    buildingOnTop = hasBuilding & (~groundKnown | (roofH >= groundH))

    if len(buildings):
        bldgSource = buildings.heightSource[index]
        bldgRes = buildings.nativeResolution_m[index]
    else:
        bldgSource = np.zeros(ids.shape, dtype=np.uint8)
        bldgRes = np.zeros(ids.shape)
    heightSource = np.where(
        buildingOnTop,
        bldgSource,
        np.where(groundKnown, np.uint8(HeightSource.GROUND), np.uint8(HeightSource.NONE)),
    )
    nativeRes = np.where(
        buildingOnTop, bldgRes, np.where(groundKnown, ground.nativeResolution_m, np.nan)
    )

    lat, lon = columns.nodeLatLon()
    undulation = ground.undulation_m(lat, lon)
    offset = undulation if request.verticalDatum == "ellipsoid" else 0.0
    layers["groundHeight_datum_m"][rows, cols] = groundH + offset
    layers["groundHeight_ENU_m"][rows, cols] = columns.toEnuUp(groundH + undulation)
    layers["surfaceHeight_datum_m"][rows, cols] = surfaceH + offset
    layers["surfaceHeight_ENU_m"][rows, cols] = columns.toEnuUp(surfaceH + undulation)
    layers["buildingHeight_m"][rows, cols] = np.where(
        hasBuilding, np.maximum(roofH - groundH, 0.0), np.nan
    )
    layers["featureId"][rows, cols] = ids
    layers["heightSource"][rows, cols] = heightSource
    layers["nativeResolution_m"][rows, cols] = nativeRes


def _resolutionWarnings(pack: Pack, gridSpacing_m: float) -> list[str]:
    warnings = []
    for role in ("ground", "buildingHeightRaster"):
        entry = pack.primarySource(role)
        if entry is not None and gridSpacing_m < entry.nativeResolution_m:
            warnings.append(
                f"grid spacing {gridSpacing_m} m is finer than {entry.sourceId!r} "
                f"({entry.nativeResolution_m} m); values are interpolated"
            )
    return warnings


def _usedSources(pack: Pack) -> list[SourceEntry]:
    used: list[SourceEntry] = []
    for role in ("ground", "geoid", "buildingHeightRaster"):
        entry = pack.primarySource(role)
        if entry is not None:
            used.append(entry)
    used += [
        pack.manifest.sources[s] for s in sourcesForRole("buildings") if s in pack.manifest.sources
    ]
    return used


def _metadata(
    request: SceneRequest,
    geometry: GridGeometry,
    ground: GroundModel,
    pack: Pack,
    warnings: list[str],
    layers: dict[str, npt.NDArray[Any]],
) -> dict[str, Any]:
    lat0, lon0 = request.sceneCenter_deg
    frame = geometry.frame
    affine = geometry.affine()
    return {
        "map3dVersion": __version__,
        "request": {
            "sceneCenter_deg": [lat0, lon0],
            "gridSpacing_m": request.gridSpacing_m,
            "gridAxes_ENU": request.gridAxes_ENU.tolist(),
            "swathHorizontal_m": request.swathHorizontal_m,
            "swathVertical_m": request.swathVertical_m,
            "verticalDatum": request.verticalDatum,
            "maxDeckLevels": request.maxDeckLevels,
            "imageryPath": None if request.imageryPath is None else str(request.imageryPath),
        },
        "frame": {
            "sceneCenter_deg": [lat0, lon0],
            "origin_ECEF_m": frame.origin_ECEF_m.tolist(),
            "enuAxes_ECEF": frame.rotation.tolist(),
            "gridAxes_ENU": geometry.gridAxes_ENU.tolist(),
            "nodeConvention": "cell-centered",
            "arrayOrder": _ARRAY_ORDER,
            "nRows": geometry.nRows,
            "nCols": geometry.nCols,
            "gridSpacing_m": geometry.gridSpacing_m,
            "orthoCrs": geometry.orthoCrs().to_wkt(),
            "affine": [affine.a, affine.b, affine.c, affine.d, affine.e, affine.f],
        },
        "verticalDatum": {
            "name": "EGM2008" if request.verticalDatum == "msl" else "WGS84 ellipsoid",
            "geoidUndulationAtCenter_m": float(ground.undulation_m(lat0, lon0)),
            "enuLayers": "Up coordinate in the scene-center ENU frame (includes Earth curvature)",
        },
        "layers": {
            name: {**LAYER_INFO[name], "dtype": str(arr.dtype), "shape": list(arr.shape)}
            for name, arr in layers.items()
        },
        "heightSourceCodes": heightSourceTable(),
        "sources": [
            {
                "sourceId": e.sourceId,
                "role": e.role,
                "dataset": e.dataset,
                "licence": e.licence,
                "attribution": e.attribution.replace(
                    "{accessed}", e.accessedDate or "(unknown date)"
                ),
                "accessedDate": e.accessedDate,
                "nativeResolution_m": e.nativeResolution_m,
            }
            for e in _usedSources(pack)
        ],
        "pack": {"path": str(pack.root), "manifestSha256": fileSha256(pack.root / MANIFEST_NAME)},
        "warnings": warnings,
    }
