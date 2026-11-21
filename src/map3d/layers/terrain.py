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

__all__ = ["DEFAULT_GROUND_SMOOTHING_m", "GroundModel", "pixelSize_m", "smoothWindow"]

FloatArray = npt.NDArray[np.float64]

DEFAULT_GROUND_SMOOTHING_m = 90.0  # FWHM of the ground low-pass in buildScene (3 FABDEM pixels)
_FWHM_PER_SIGMA = 2.0 * float(np.sqrt(2.0 * np.log(2.0)))
_TRUNCATE_SIGMAS = 4.0
_METERS_PER_DEG = 111_320.0


def pixelSize_m(window: RasterWindow) -> tuple[float, float]:
    """Approximate pixel width and height in metres (geographic: at the window centre)."""
    t = window.transform
    pxX, pxY = abs(t.a), abs(t.e)
    if window.crs is not None and window.crs.is_geographic:
        lat = t.f + t.e * window.values.shape[0] / 2.0
        pxX *= _METERS_PER_DEG * float(np.cos(np.radians(lat)))
        pxY *= _METERS_PER_DEG
    return float(pxX), float(pxY)


def smoothWindow(window: RasterWindow, fwhm_m: float) -> RasterWindow:
    """Gaussian low-pass of a raster window, lowering its effective resolution.

    The filter is separable and normalised over valid pixels, so no-data (NaN)
    neither spreads nor drags neighbours toward zero; NaN pixels stay NaN.
    Linear surfaces are preserved away from the window edge. Pixel sizes of
    geographic rasters are converted to metres at the window's centre latitude.

    Args:
        window: Raster window (heights).
        fwhm_m: Filter full width at half maximum, in metres; 0 returns the window.

    Returns:
        A new window with smoothed values (same transform and CRS).

    Raises:
        ValueError: If ``fwhm_m`` is negative.
    """
    if fwhm_m < 0:
        raise ValueError(f"ground smoothing must be >= 0 m, got {fwhm_m}")
    if fwhm_m == 0:
        return window
    pxX, pxY = pixelSize_m(window)
    sigma_m = fwhm_m / _FWHM_PER_SIGMA
    valid = np.isfinite(window.values)
    num = np.where(valid, window.values, 0.0)
    den = valid.astype(np.float64)
    for axis, px in ((1, pxX), (0, pxY)):
        kernel = _gaussianKernel(sigma_m / px)
        num = _convolve(num, kernel, axis)
        den = _convolve(den, kernel, axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = np.where(valid & (den > 0), num / den, np.nan)
    return RasterWindow(smoothed, window.transform, window.crs, window.sourceId)


def _gaussianKernel(sigma_px: float) -> FloatArray:
    radius = max(1, int(np.ceil(_TRUNCATE_SIGMAS * sigma_px)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel: FloatArray = np.exp(-0.5 * (x / sigma_px) ** 2)
    return kernel / kernel.sum()


def _convolve(a: FloatArray, kernel: FloatArray, axis: int) -> FloatArray:
    """``a`` convolved with a symmetric ``kernel`` along ``axis`` (zero outside)."""
    r = kernel.size // 2
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r, r)
    padded = np.pad(a, pad)
    n = a.shape[axis]
    out = np.zeros_like(a)
    for i, k in enumerate(kernel):
        out += k * np.take(padded, np.arange(i, i + n), axis=axis)
    return out


@dataclass(frozen=True, eq=False)
class GroundModel:
    """Ground heights from the pack's highest-priority ground source.

    Attributes:
        sourceId: Ground source used.
        nativeResolution_m: Its effective resolution: the source's nominal
            resolution, or the smoothing width if coarser.
        verticalReference: Its vertical reference (``"EGM2008"`` or ``"WGS84"``).
        window: Raster data covering the scene footprint (smoothed, if requested).
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
    def load(cls, pack: Pack, geometry: GridGeometry, smoothing_m: float = 0.0) -> GroundModel:
        """Read the ground source and geoid over the scene footprint.

        Args:
            pack: Opened data pack.
            geometry: Scene grid geometry.
            smoothing_m: Gaussian low-pass width (FWHM) applied to the source,
                in metres; removes bumps narrower than about this (see
                :func:`smoothWindow`). 0 keeps the source as is.

        Returns:
            The ground model.

        Raises:
            ValueError: If the pack has no ground source, or ``smoothing_m`` < 0.
        """
        entry = pack.primarySource("ground")
        if entry is None:
            raise ValueError("pack has no source for required role 'ground'")
        if smoothing_m < 0:
            raise ValueError(f"ground smoothing must be >= 0 m, got {smoothing_m}")
        # Read beyond the footprint by the filter radius, so the scene is not
        # affected by the window edge.
        radius_m = _TRUNCATE_SIGMAS * smoothing_m / _FWHM_PER_SIGMA
        bounds = geometry.footprintBounds_deg(margin_m=geometry.gridSpacing_m + radius_m)
        window = smoothWindow(pack.readRaster(entry.sourceId, bounds), smoothing_m)
        return cls(
            sourceId=entry.sourceId,
            nativeResolution_m=max(entry.nativeResolution_m, smoothing_m),
            verticalReference=entry.verticalReference,
            window=window,
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

    def cellMaxOrthometric(self, columns: ChunkColumns) -> FloatArray:
        """Cell-max ground height above MSL for a chunk.

        Samples a sub-lattice that includes all cell corners, subdivided when
        the source is finer than the grid, and max-pools it per cell.

        Args:
            columns: Chunk column geometry.

        Returns:
            Orthometric cell-max ground height, chunk-shaped.
        """
        ss = supersampleFactor(columns.geometry.gridSpacing_m, self.nativeResolution_m)
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
