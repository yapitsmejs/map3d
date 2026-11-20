"""Bare-earth ground layer (SPEC sections 5.4 and 6.1).

Internally all heights are **orthometric (EGM2008)**, like FABDEM/GLO-30. The
scene converts them to the requested datum and to ENU Up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from map3d.geoid import GeoidGrid
from map3d.gridGeometry import GridGeometry
from map3d.layers.columns import ChunkColumns, cellMax, supersampleFactor
from map3d.pack.reader import Pack, RasterWindow

if TYPE_CHECKING:
    from map3d.layers.groundFix import GroundCorrection

__all__ = ["GroundModel"]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class GroundModel:
    """Ground heights from the pack's highest-priority ground source.

    Attributes:
        sourceId: Ground source used.
        nativeResolution_m: Its nominal resolution.
        verticalReference: Its vertical reference (``"EGM2008"`` or ``"WGS84"``).
        window: Raster data covering the scene footprint.
        geoid: Geoid grid for datum conversions.
        correction: Optional cap removing building residue under buildings
            (see :mod:`map3d.layers.groundFix`); applied to every query.
    """

    sourceId: str
    nativeResolution_m: float
    verticalReference: str | None
    window: RasterWindow
    geoid: GeoidGrid
    correction: GroundCorrection | None = None

    @classmethod
    def load(cls, pack: Pack, geometry: GridGeometry) -> GroundModel:
        """Read the ground source and geoid over the scene footprint.

        Args:
            pack: Opened data pack.
            geometry: Scene grid geometry.

        Returns:
            The ground model.

        Raises:
            ValueError: If the pack has no ground source.
        """
        entry = pack.primarySource("ground")
        if entry is None:
            raise ValueError("pack has no source for required role 'ground'")
        bounds = geometry.footprintBounds_deg(margin_m=geometry.gridSpacing_m)
        return cls(
            sourceId=entry.sourceId,
            nativeResolution_m=entry.nativeResolution_m,
            verticalReference=entry.verticalReference,
            window=pack.readRaster(entry.sourceId, bounds),
            geoid=pack.geoid(),
        )

    def undulation_m(self, lat_deg: npt.ArrayLike, lon_deg: npt.ArrayLike) -> FloatArray:
        """Geoid undulation ``N`` (``h_ellipsoid = H_msl + N``)."""
        return self.geoid.undulation_m(lat_deg, lon_deg)

    def orthometricAt(self, lat_deg: FloatArray, lon_deg: FloatArray) -> FloatArray:
        """Bilinear ground height above MSL (EGM2008) at geographic positions.

        Args:
            lat_deg: Latitude, in degrees.
            lon_deg: Longitude, in degrees.

        Returns:
            Orthometric height (NaN outside the source).

        Raises:
            ValueError: If the source's vertical reference is unsupported.
        """
        values = self.window.sample(lat_deg, lon_deg)
        if self.verticalReference == "WGS84":
            values = values - self.undulation_m(lat_deg, lon_deg)
        elif self.verticalReference != "EGM2008":
            raise ValueError(
                f"ground source {self.sourceId!r} has unsupported vertical reference "
                f"{self.verticalReference!r}"
            )
        if self.correction is not None:
            values = self.correction.correctPoints(lat_deg, lon_deg, values)
        return values

    def cellMaxOrthometric(self, columns: ChunkColumns, gridSpacing_m: float) -> FloatArray:
        """Cell-max ground height above MSL for a chunk.

        Samples a sub-lattice that includes all cell corners, subdivided when
        the source is finer than the grid, and max-pools it per cell.

        Args:
            columns: Chunk column geometry.
            gridSpacing_m: Grid spacing.

        Returns:
            Orthometric cell-max ground height, chunk-shaped.
        """
        ss = supersampleFactor(gridSpacing_m, self.nativeResolution_m)
        col, row = columns.mapLattice(ss, self.window.pixelCoords)
        values = self.window.sampleAtPixels(col, row)
        if self.verticalReference == "WGS84":
            lat, lon = columns.latticeLatLon(ss)
            values = values - self.undulation_m(lat, lon)
        elif self.verticalReference != "EGM2008":
            raise ValueError(
                f"ground source {self.sourceId!r} has unsupported vertical reference "
                f"{self.verticalReference!r}"
            )
        ground = cellMax(values, ss)
        if self.correction is not None:
            ground = self.correction.correctChunk(columns.rows, columns.cols, ground)
        return ground
