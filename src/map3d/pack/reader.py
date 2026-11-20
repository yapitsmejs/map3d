"""Read-only access to an offline data pack (SPEC section 9).

Scene builds never touch the network: everything comes through :class:`Pack`.
"""

from __future__ import annotations

import math
import os
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyproj
import rasterio
import shapely
from rasterio.merge import merge
from rasterio.warp import transform_bounds

from map3d.geoid import GeoidGrid
from map3d.pack.geoparquet import readGeoParquet
from map3d.pack.manifest import (
    Bounds,
    Manifest,
    PackFile,
    SourceEntry,
    fileSha256,
    validateBounds,
)
from map3d.pack.registry import ROLES, sourcesForRole

__all__ = ["Pack", "RasterWindow", "VectorSelection"]

FloatArray = npt.NDArray[np.float64]
SampleMethod = Literal["bilinear", "nearest"]


@dataclass(frozen=True, eq=False)
class RasterWindow:
    """A block of raster data read from a pack, in the source's own CRS.

    Attributes:
        values: Pixel values as float64, nodata as NaN, shape ``(rows, cols)``.
        transform: Affine map from pixel-corner ``(col, row)`` to CRS ``(x, y)``.
        crs: The source CRS.
        sourceId: Source the data came from.
    """

    values: FloatArray
    transform: Any
    crs: Any
    sourceId: str

    def sample(
        self, lat_deg: npt.ArrayLike, lon_deg: npt.ArrayLike, method: SampleMethod = "bilinear"
    ) -> FloatArray:
        """Sample the window at geographic positions.

        Bilinear interpolation is between pixel centers; it returns NaN when any
        of the four neighbours is NaN or outside the window.

        Args:
            lat_deg: Latitude, in degrees.
            lon_deg: Longitude, in degrees.
            method: ``"bilinear"`` or ``"nearest"``.

        Returns:
            Sampled values, broadcast to the inputs' common shape.

        Raises:
            ValueError: For an unknown method.
        """
        lat, lon = np.broadcast_arrays(
            np.asarray(lat_deg, dtype=np.float64), np.asarray(lon_deg, dtype=np.float64)
        )
        x, y = self._toCrs(lon, lat)
        inv = ~self.transform
        col = inv.a * x + inv.b * y + inv.c
        row = inv.d * x + inv.e * y + inv.f
        nRows, nCols = self.values.shape
        if method == "nearest":
            i = np.floor(row).astype(np.intp)
            j = np.floor(col).astype(np.intp)
            inside = (i >= 0) & (i < nRows) & (j >= 0) & (j < nCols)
            out = np.full(lat.shape, np.nan)
            out[inside] = self.values[i[inside], j[inside]]
            return out
        if method != "bilinear":
            raise ValueError(f"unknown sampling method {method!r}")
        fr = row - 0.5
        fc = col - 0.5
        i0 = np.floor(fr).astype(np.intp)
        j0 = np.floor(fc).astype(np.intp)
        inside = (i0 >= 0) & (i0 + 1 < nRows) & (j0 >= 0) & (j0 + 1 < nCols)
        out = np.full(lat.shape, np.nan)
        ii, jj = i0[inside], j0[inside]
        ti, tj = fr[inside] - ii, fc[inside] - jj
        v = self.values
        out[inside] = (
            (1 - ti) * (1 - tj) * v[ii, jj]
            + (1 - ti) * tj * v[ii, jj + 1]
            + ti * (1 - tj) * v[ii + 1, jj]
            + ti * tj * v[ii + 1, jj + 1]
        )
        return out

    def _toCrs(self, lon: FloatArray, lat: FloatArray) -> tuple[FloatArray, FloatArray]:
        crs = pyproj.CRS.from_wkt(self.crs.to_wkt())
        if crs.equals(pyproj.CRS("OGC:CRS84"), ignore_axis_order=True):
            return lon, lat
        toCrs = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x, y = toCrs.transform(lon, lat)
        return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


@dataclass(frozen=True, eq=False)
class VectorSelection:
    """Vector features selected from a pack.

    Attributes:
        table: Canonical attribute columns (no geometry column).
        geometries: Shapely geometries in lon/lat, aligned with ``table`` rows.
    """

    table: pa.Table
    geometries: npt.NDArray[np.object_]

    def __len__(self) -> int:
        return int(self.table.num_rows)


class Pack:
    """An opened offline data pack."""

    def __init__(self, root: Path, manifest: Manifest) -> None:
        """Wrap an already-loaded manifest; prefer :meth:`open`.

        Args:
            root: Pack root directory.
            manifest: Its manifest.
        """
        self.root = root
        self.manifest = manifest

    @classmethod
    def open(cls, packDir: str | os.PathLike[str]) -> Pack:
        """Open a pack.

        Args:
            packDir: Pack root directory.

        Returns:
            The pack.
        """
        root = Path(packDir)
        return cls(root, Manifest.load(root))

    @property
    def bounds_deg(self) -> Bounds:
        """Declared pack coverage ``(minLon, minLat, maxLon, maxLat)``."""
        return self.manifest.bounds_deg

    @property
    def sourceIds(self) -> list[str]:
        """Identifiers of the sources present in the pack."""
        return list(self.manifest.sources)

    def primarySource(self, role: str) -> SourceEntry | None:
        """Highest-priority source present for a role.

        Args:
            role: Role identifier.

        Returns:
            The source entry, or ``None`` if the pack has no source for the role.
        """
        for sourceId in sourcesForRole(role):
            if sourceId in self.manifest.sources:
                return self.manifest.sources[sourceId]
        return None

    def checkCoverage(self, bounds_deg: Bounds) -> list[str]:
        """Check that the pack can serve a scene footprint.

        Args:
            bounds_deg: Scene footprint ``(minLon, minLat, maxLon, maxLat)``.

        Raster coverage is checked against file extents. Vector sources are
        only checked for presence: a vector file's extent is where its features
        lie, not the area it was clipped to (packs clip vectors to the pack).

        Returns:
            Warnings about missing or partial optional data.

        Raises:
            ValueError: If the footprint is outside the pack, or a required role
                is missing or does not cover it.
        """
        box = shapely.box(*validateBounds(bounds_deg))
        if not shapely.box(*self.bounds_deg).contains(box):
            raise ValueError(f"scene footprint {bounds_deg} is outside the pack {self.bounds_deg}")
        warnings: list[str] = []
        for role, spec in ROLES.items():
            entry = self.primarySource(role)
            if entry is None:
                if spec.required:
                    raise ValueError(f"pack has no source for required role {role!r}")
                warnings.append(f"no source for role {role!r}; its layers will be empty")
                continue
            if entry.kind == "raster" and not _fileUnion(entry.files).contains(box):
                if spec.required:
                    raise ValueError(f"{entry.sourceId!r} does not cover the scene footprint")
                warnings.append(f"{entry.sourceId!r} only partially covers the scene footprint")
        return warnings

    def readRaster(
        self, sourceId: str, bounds_deg: Bounds, band: int = 1, marginPixels: int = 2
    ) -> RasterWindow:
        """Read a raster source over a lon/lat box, mosaicking tiles.

        The window is snapped to the source pixel grid (no resampling) and
        padded by ``marginPixels`` so bilinear sampling works up to the box edge.

        Args:
            sourceId: Raster source in the pack.
            bounds_deg: ``(minLon, minLat, maxLon, maxLat)``.
            band: 1-based band index.
            marginPixels: Padding around the box, in source pixels.

        Returns:
            The raster window.

        Raises:
            ValueError: If the source is absent, not raster, or has no files there.
        """
        entry = self._entry(sourceId, "raster")
        query = shapely.box(*validateBounds(bounds_deg))
        files = [f for f in entry.files if f.bounds_deg and _extent(f.bounds_deg).intersects(query)]
        if not files:
            raise ValueError(f"{sourceId!r} has no files intersecting {bounds_deg}")
        with ExitStack() as stack:
            srcs = [stack.enter_context(rasterio.open(self.root / f.path)) for f in files]
            first = srcs[0]
            left, bottom, right, top = transform_bounds("EPSG:4326", first.crs, *bounds_deg)
            snapped = _snapToGrid(first.transform, (left, bottom, right, top), marginPixels)
            data, transform = merge(
                srcs, bounds=snapped, indexes=[band], dtype="float64", nodata=np.nan
            )
            crs = first.crs
        return RasterWindow(np.asarray(data[0], dtype=np.float64), transform, crs, sourceId)

    def readVectors(
        self, sourceId: str, bounds_deg: Bounds, columns: list[str] | None = None
    ) -> VectorSelection:
        """Read vector features intersecting a lon/lat box.

        Args:
            sourceId: Vector source in the pack.
            bounds_deg: ``(minLon, minLat, maxLon, maxLat)``.
            columns: Attribute columns to read (default: all).

        Returns:
            The selected features.

        Raises:
            ValueError: If the source is absent or not vector.
        """
        entry = self._entry(sourceId, "vector")
        query = shapely.box(*validateBounds(bounds_deg))
        tables: list[pa.Table] = []
        geoms: list[npt.NDArray[np.object_]] = []
        for f in entry.files:
            if f.bounds_deg is not None and not _extent(f.bounds_deg).intersects(query):
                continue
            table, fileGeoms = readGeoParquet(self.root / f.path, columns)
            mask = shapely.intersects(fileGeoms, query)
            tables.append(table.filter(pa.array(mask)))
            geoms.append(fileGeoms[mask])
        if not tables:
            table, fileGeoms = readGeoParquet(self.root / entry.files[0].path, columns)
            return VectorSelection(table.slice(0, 0), fileGeoms[:0])
        return VectorSelection(pa.concat_tables(tables), np.concatenate(geoms))

    def geoid(self) -> GeoidGrid:
        """Load the geoid grid of the highest-priority geoid source.

        Returns:
            The geoid grid.

        Raises:
            ValueError: If the pack has no geoid source.
        """
        entry = self.primarySource("geoid")
        if entry is None:
            raise ValueError("pack has no source for required role 'geoid'")
        return GeoidGrid.fromGeoTiff(self.root / entry.files[0].path)

    def verify(self) -> list[str]:
        """Check every pack file against its recorded checksum.

        Returns:
            Problems found (empty when the pack is intact).
        """
        problems: list[str] = []
        for entry in self.manifest.sources.values():
            for f in entry.files:
                path = self.root / f.path
                if not path.is_file():
                    problems.append(f"missing file: {f.path}")
                elif fileSha256(path) != f.sha256:
                    problems.append(f"checksum mismatch: {f.path}")
        return problems

    def _entry(self, sourceId: str, kind: str) -> SourceEntry:
        entry = self.manifest.sources.get(sourceId)
        if entry is None:
            raise ValueError(f"source {sourceId!r} is not in pack {self.root}")
        if entry.kind != kind:
            raise ValueError(f"source {sourceId!r} is not a {kind} source")
        return entry


def _fileUnion(files: list[PackFile]) -> Any:
    return shapely.union_all([_extent(f.bounds_deg) for f in files if f.bounds_deg])


def _extent(bounds: Bounds) -> Any:
    """Extent as a geometry: polygon, or line/point when the box is degenerate."""
    minX, minY, maxX, maxY = bounds
    return shapely.envelope(shapely.multipoints([(minX, minY), (maxX, maxY)]))


def _snapToGrid(
    transform: Any, bounds: tuple[float, float, float, float], marginPixels: int
) -> tuple[float, float, float, float]:
    left, bottom, right, top = bounds
    x0, resX = transform.c, transform.a
    y0, resY = transform.f, -transform.e
    snappedLeft = x0 + (math.floor((left - x0) / resX) - marginPixels) * resX
    snappedRight = x0 + (math.ceil((right - x0) / resX) + marginPixels) * resX
    snappedTop = y0 - (math.floor((y0 - top) / resY) - marginPixels) * resY
    snappedBottom = y0 - (math.ceil((y0 - bottom) / resY) + marginPixels) * resY
    return (snappedLeft, snappedBottom, snappedRight, snappedTop)
