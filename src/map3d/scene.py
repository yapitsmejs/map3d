"""Scene building: request + pack -> multi-layer grid (SPEC sections 5-6).

All layer maths happens in orthometric height (MSL, EGM2008) and is converted at
the end of each chunk:

- ``*_datum_m``: ``H`` for ``verticalDatum="msl"``, else ``H + N`` (ellipsoidal).
- ``*_ENU_m``: Up coordinate of the column point at ellipsoidal height ``H + N``.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import rasterio

from map3d import __version__
from map3d.gridGeometry import GridGeometry
from map3d.layers import decks as deckModel
from map3d.layers.buildings import BuildingSet, loadBuildings, rasterizeBuildings
from map3d.layers.canopy import CanopyModel
from map3d.layers.columns import ChunkColumns
from map3d.layers.decks import DeckSet, loadDecks, rasterizeDecks
from map3d.layers.landCover import CLASS_BUILDING, CLASS_DECK, LandCoverModel, classTable
from map3d.layers.provenance import HeightSource, heightSourceTable
from map3d.layers.terrain import DEFAULT_GROUND_SMOOTHING_m, GroundModel
from map3d.pack.manifest import MANIFEST_NAME, SourceEntry, fileSha256
from map3d.pack.reader import Pack
from map3d.pack.registry import sourcesForRole
from map3d.request import SceneRequest

if TYPE_CHECKING:
    from map3d.mesh.tileset import Style

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
        "description": "Top surface (terrain, canopy, decks, buildings) above the vertical datum (cell-max DSM).",
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
    "canopyHeight_m": {
        "units": "m",
        "description": "Vegetation canopy height above ground (cell-max); 0 = none, NaN = no canopy source.",
    },
    "landClass": {
        "units": "1",
        "description": "Class of the top-most surface: building > deck > road/rail > land cover (see landClassCodes).",
    },
    "deckTopHeight_datum_m": {
        "units": "m",
        "description": "Top of elevated-structure level k (k = 0 lowest) above the vertical datum; shape (K, rows, cols).",
    },
    "deckTopHeight_ENU_m": {
        "units": "m",
        "description": "Top of elevated-structure level k: Up coordinate in the scene-center ENU frame.",
    },
    "deckBottomHeight_datum_m": {
        "units": "m",
        "description": "Underside of elevated-structure level k above the vertical datum.",
    },
    "deckBottomHeight_ENU_m": {
        "units": "m",
        "description": "Underside of elevated-structure level k: Up coordinate in the scene-center ENU frame.",
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
        storageDir: Directory of memory-mapped layers, if any.
        models: Source models used to build the grid (ground, buildings, canopy,
            land cover, decks), reused to build the 3D Tiles mesh.
        pack: The pack the scene was built from.
    """

    request: SceneRequest
    geometry: GridGeometry
    layers: dict[str, npt.NDArray[Any]]
    features: dict[str, npt.NDArray[Any]]
    metadata: dict[str, Any] = field(default_factory=dict)
    storageDir: Path | None = None
    _ownsStorage: bool = field(default=False, repr=False)
    models: Any = field(default=None, repr=False)
    pack: Pack | None = field(default=None, repr=False)

    def __enter__(self) -> Scene:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release disk-backed layers; delete their directory if the scene created it.

        A no-op for in-memory scenes. Safe to call more than once.
        """
        if self.storageDir is None:
            return
        for array in self.layers.values():
            mm = getattr(array, "_mmap", None)
            if mm is not None:
                mm.close()
        self.layers = {}
        if self._ownsStorage:
            shutil.rmtree(self.storageDir, ignore_errors=True)
        self.storageDir = None

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

    def write3dTiles(
        self,
        outDir: str | os.PathLike[str],
        withViewer: bool = True,
        style: Style = "conceptual",
        buildingImageryPath: str | os.PathLike[str] | None = None,
    ) -> Path:
        """Write the scene as a 3D Tiles 1.1 tileset (plus the offline viewer).

        Args:
            outDir: Output directory.
            withViewer: Also install the offline viewer (``map3d view``).
            style: ``"conceptual"`` (flat map colours, lit), the default.
            buildingImageryPath: Optional high-resolution RGB orthophoto for
                building palettes. Default: the pack's imagery, with neutral
                fallback. Facade windows and floor bands are procedural.

        Returns:
            Path of ``tileset.json``.
        """
        from map3d.mesh.tileset import write3dTiles
        from map3d.viewer import installViewer

        path = write3dTiles(self, outDir, style=style, buildingImageryPath=buildingImageryPath)
        if withViewer:
            installViewer(outDir)
        return path


@dataclass(frozen=True, eq=False)
class _Models:
    ground: GroundModel
    buildings: BuildingSet
    canopy: CanopyModel
    landCover: LandCoverModel
    decks: DeckSet


def buildScene(
    request: SceneRequest,
    pack: str | os.PathLike[str] | Pack,
    chunkSize: int = 1024,
    workers: int | None = None,
    workDir: str | os.PathLike[str] | None = None,
    memoryBudget_GB: float = 2.0,
    correctGround: bool = True,
    groundSmoothing_m: float = DEFAULT_GROUND_SMOOTHING_m,
) -> Scene:
    """Build the scene grid for a request from an offline pack.

    Chunks are processed in parallel threads (the hot loops are numba ``nogil``
    kernels). Results are bitwise identical for any chunking or worker count.

    Args:
        request: Validated scene request.
        pack: Pack directory or opened pack.
        chunkSize: Rows/columns per processing chunk.
        workers: Parallel chunk workers (default: min(8, CPU count)).
        workDir: Store layers as memory-mapped ``.npy`` files here (kept).
        memoryBudget_GB: If the layers would exceed this and no ``workDir`` is
            given, they are memory-mapped in a temporary directory that
            :meth:`Scene.close` deletes.
        correctGround: Remove building residue (failed building removal in
            FABDEM) from the ground under footprints; see
            :mod:`map3d.layers.groundFix`.
        groundSmoothing_m: Gaussian low-pass width (FWHM) applied to the
            ground source before anything samples it, removing bumps narrower
            than about this (FABDEM noise and building residue). 0 disables.

    Returns:
        The scene.

    Raises:
        ValueError: If the pack cannot serve the scene (outside coverage, or a
            required role is missing).
    """
    opened = pack if isinstance(pack, Pack) else Pack.open(pack)
    geometry = GridGeometry.fromRequest(request)
    warnings = opened.checkCoverage(geometry.footprintBounds_deg())
    ground = GroundModel.load(opened, geometry, smoothing_m=groundSmoothing_m)
    buildings = loadBuildings(opened, geometry, ground, correctGround=correctGround)
    if buildings.groundCorrection is not None:
        ground = dataclasses.replace(ground, correction=buildings.groundCorrection)
    models = _Models(
        ground=ground,
        buildings=buildings,
        canopy=CanopyModel.load(opened, geometry),
        landCover=LandCoverModel.load(opened, geometry),
        decks=loadDecks(opened, geometry, ground),
    )
    warnings += _resolutionWarnings(opened, request.gridSpacing_m)

    storageDir, ownsStorage = _chooseStorage(
        geometry.shape, request.maxDeckLevels, workDir, memoryBudget_GB
    )
    layers = _allocate(geometry.shape, request.maxDeckLevels, storageDir)
    chunks = list(geometry.iterChunks(chunkSize, chunkSize))
    nWorkers = workers if workers is not None else min(8, os.cpu_count() or 1)
    if nWorkers > 1 and len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=nWorkers) as pool:
            overflow = sum(
                pool.map(
                    lambda rc: _buildChunkInThread(layers, request, geometry, models, *rc), chunks
                )
            )
    else:
        overflow = sum(
            _buildChunk(layers, request, geometry, models, rows, cols) for rows, cols in chunks
        )
    if overflow:
        warnings.append(
            f"{overflow} nodes have more than maxDeckLevels={request.maxDeckLevels} deck levels; "
            "the lowest are stored and the highest still sets surfaceHeight"
        )
    for message in warnings:
        _log.warning(message)

    features = _concatFeatures(
        models.buildings.featureTable(),
        models.decks.featureTable(firstFeatureId=len(models.buildings) + 1),
    )
    metadata = _metadata(request, geometry, ground, opened, warnings, layers)
    metadata["storage"] = "memory" if storageDir is None else "memmap"
    metadata["groundSmoothing_m"] = float(groundSmoothing_m)
    metadata["groundCorrection"] = (
        buildings.groundCorrection.summary()
        if buildings.groundCorrection is not None
        else {"buildingsCorrected": 0, "maxLowering_m": 0.0, "method": "disabled"}
    )
    return Scene(
        request,
        geometry,
        layers,
        features,
        metadata,
        storageDir=storageDir,
        _ownsStorage=ownsStorage,
        models=models,
        pack=opened,
    )


def _concatFeatures(*tables: dict[str, npt.NDArray[Any]]) -> dict[str, npt.NDArray[Any]]:
    return {key: np.concatenate([t[key] for t in tables]) for key in tables[0]}


_DECK_LAYERS = (
    "deckTopHeight_datum_m",
    "deckTopHeight_ENU_m",
    "deckBottomHeight_datum_m",
    "deckBottomHeight_ENU_m",
)


def _layerSpecs(
    shape: tuple[int, int], maxDeckLevels: int
) -> dict[str, tuple[tuple[int, ...], type]]:
    specs: dict[str, tuple[tuple[int, ...], type]] = {}
    for name in LAYER_INFO:
        if name == "featureId":
            specs[name] = (shape, np.int64)
        elif name in ("heightSource", "landClass"):
            specs[name] = (shape, np.uint8)
        elif name in _DECK_LAYERS:
            specs[name] = ((maxDeckLevels, *shape), np.float32)
        else:
            specs[name] = (shape, np.float32)
    return specs


def _chooseStorage(
    shape: tuple[int, int],
    maxDeckLevels: int,
    workDir: str | os.PathLike[str] | None,
    budget_GB: float,
) -> tuple[Path | None, bool]:
    if workDir is not None:
        path = Path(workDir)
        path.mkdir(parents=True, exist_ok=True)
        return path, False
    total = sum(
        int(np.prod(sh)) * np.dtype(dt).itemsize
        for sh, dt in _layerSpecs(shape, maxDeckLevels).values()
    )
    if total > budget_GB * 1e9:
        return Path(tempfile.mkdtemp(prefix="map3d-scene-")), True
    return None, False


def _allocate(
    shape: tuple[int, int], maxDeckLevels: int, storageDir: Path | None = None
) -> dict[str, npt.NDArray[Any]]:
    # No prefill: _buildChunk assigns every element of every layer.
    layers: dict[str, npt.NDArray[Any]] = {}
    for name, (layerShape, dtype) in _layerSpecs(shape, maxDeckLevels).items():
        if storageDir is None:
            layers[name] = np.empty(layerShape, dtype=dtype)
        else:
            layers[name] = np.lib.format.open_memmap(
                storageDir / f"{name}.npy", mode="w+", dtype=dtype, shape=layerShape
            )
    return layers


def _buildChunkInThread(*args: Any) -> int:
    # rasterio's GDAL environment (Env) is thread-local; give each worker its own.
    with rasterio.Env():
        return _buildChunk(*args)


def _buildChunk(
    layers: dict[str, npt.NDArray[Any]],
    request: SceneRequest,
    geometry: GridGeometry,
    models: _Models,
    rows: slice,
    cols: slice,
) -> int:
    columns = ChunkColumns(geometry, rows, cols)
    ground, buildings = models.ground, models.buildings
    groundH = ground.cellMaxOrthometric(columns, geometry.gridSpacing_m)
    canopy = models.canopy.cellMax(columns)
    ids = rasterizeBuildings(buildings, geometry, rows, cols)
    decks = rasterizeDecks(models.decks, geometry, columns, request.maxDeckLevels)

    hasBuilding = ids > 0
    index = np.maximum(ids - 1, 0)
    if len(buildings):
        roofH = np.where(hasBuilding, buildings.roofOrthometric_m[index], np.nan)
        bldgSource = buildings.heightSource[index]
        bldgRes = buildings.nativeResolution_m[index]
    else:
        roofH = np.full(ids.shape, np.nan)
        bldgSource = np.zeros(ids.shape, dtype=np.uint8)
        bldgRes = np.zeros(ids.shape)
    canopyTop = np.where(canopy > 0, groundH + canopy, np.nan)
    hasDeck = decks.maxDeck > 0

    # Candidates in tie-break order: building > deck > canopy > ground.
    stack = np.stack([roofH, decks.maxTop, canopyTop, groundH])
    known = np.isfinite(stack)
    anyKnown = known.any(axis=0)
    winner = np.argmax(np.where(known, stack, -np.inf), axis=0)
    surfaceH = np.where(anyKnown, np.nanmax(np.where(known, stack, -np.inf), axis=0), np.nan)
    sourceByWinner = np.stack(
        [
            bldgSource,
            np.full(ids.shape, HeightSource.DECK_ESTIMATED, dtype=np.uint8),
            np.full(ids.shape, HeightSource.CANOPY, dtype=np.uint8),
            np.full(ids.shape, HeightSource.GROUND, dtype=np.uint8),
        ]
    )
    heightSource = np.take_along_axis(sourceByWinner, winner[None], axis=0)[0]
    heightSource = np.where(anyKnown, heightSource, np.uint8(HeightSource.NONE)).astype(np.uint8)
    resByWinner = np.stack(
        [
            bldgRes,
            np.zeros(ids.shape),
            np.full(ids.shape, models.canopy.nativeResolution_m),
            np.full(ids.shape, ground.nativeResolution_m),
        ]
    )
    nativeRes = np.where(anyKnown, np.take_along_axis(resByWinner, winner[None], axis=0)[0], np.nan)

    deckOnTop = hasDeck & (~hasBuilding | (decks.maxTop > roofH))
    featureId = np.where(deckOnTop, len(buildings) + decks.maxDeck, ids)
    landClass = models.landCover.classes(columns)
    landClass = np.where(hasDeck, np.uint8(CLASS_DECK), landClass)
    landClass = np.where(hasBuilding, np.uint8(CLASS_BUILDING), landClass).astype(np.uint8)

    (undulation,) = columns.mapNodes(lambda la, lo: (ground.undulation_m(la, lo),))
    offset = undulation if request.verticalDatum == "ellipsoid" else 0.0
    layers["groundHeight_datum_m"][rows, cols] = groundH + offset
    layers["groundHeight_ENU_m"][rows, cols] = columns.toEnuUp(groundH + undulation)
    layers["surfaceHeight_datum_m"][rows, cols] = surfaceH + offset
    layers["surfaceHeight_ENU_m"][rows, cols] = columns.toEnuUp(surfaceH + undulation)
    layers["buildingHeight_m"][rows, cols] = np.where(
        hasBuilding, np.maximum(roofH - groundH, 0.0), np.nan
    )
    layers["canopyHeight_m"][rows, cols] = canopy
    layers["featureId"][rows, cols] = featureId
    layers["heightSource"][rows, cols] = heightSource
    layers["nativeResolution_m"][rows, cols] = nativeRes
    layers["landClass"][rows, cols] = landClass
    layers["deckTopHeight_datum_m"][:, rows, cols] = decks.tops + offset
    layers["deckTopHeight_ENU_m"][:, rows, cols] = columns.toEnuUp(decks.tops + undulation)
    layers["deckBottomHeight_datum_m"][:, rows, cols] = decks.bottoms + offset
    layers["deckBottomHeight_ENU_m"][:, rows, cols] = columns.toEnuUp(decks.bottoms + undulation)
    return decks.overflowNodes


def _resolutionWarnings(pack: Pack, gridSpacing_m: float) -> list[str]:
    warnings = []
    for role in ("ground", "buildingHeightRaster", "canopyHeight", "landCover"):
        entry = pack.primarySource(role)
        if entry is not None and gridSpacing_m < entry.nativeResolution_m:
            how = "sampled nearest" if role == "landCover" else "interpolated"
            warnings.append(
                f"grid spacing {gridSpacing_m} m is finer than {entry.sourceId!r} "
                f"({entry.nativeResolution_m} m); values are {how}"
            )
    return warnings


def _usedSources(pack: Pack) -> list[SourceEntry]:
    used: list[SourceEntry] = []
    for role in ("ground", "geoid", "buildingHeightRaster", "canopyHeight", "trees", "landCover"):
        entry = pack.primarySource(role)
        if entry is not None:
            used.append(entry)
    for role in ("buildings", "transport", "elevatedStructures"):
        used += [
            pack.manifest.sources[s] for s in sourcesForRole(role) if s in pack.manifest.sources
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
        "landClassCodes": classTable(),
        "deckModel": {
            "roadLevelHeight_m": deckModel.ROAD_LEVEL_HEIGHT_m,
            "railLevelHeight_m": deckModel.RAIL_LEVEL_HEIGHT_m,
            "footbridgeLevelHeight_m": deckModel.FOOTBRIDGE_LEVEL_HEIGHT_m,
            "deckThickness_m": deckModel.DECK_THICKNESS_m,
            "mergeTolerance_m": deckModel.MERGE_TOLERANCE_m,
            "note": "Deck heights are estimated: ground interpolated between bridge ends + level x height.",
        },
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
