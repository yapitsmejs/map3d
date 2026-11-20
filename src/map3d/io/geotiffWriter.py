"""GeoTIFF writer (SPEC section 6.3).

One file per layer; 3-D layers ``(K, rows, cols)`` become K-band files. The
CRS is the scene's ellipsoidal orthographic CRS, and the affine encodes the
grid rotation and the row-0-at-minus-y order, so pixel values and array order
are identical to the ``.npz`` and the georeferencing is exact.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import rasterio
from rasterio.windows import Window

if TYPE_CHECKING:
    from map3d.scene import Scene

__all__ = ["writeGeoTiff"]

_TILE = 256
_BAND_ROWS = 2048  # a multiple of the tile size


def writeGeoTiff(scene: Scene, outDir: str | os.PathLike[str]) -> list[Path]:
    """Write every layer of a scene as a GeoTIFF.

    Args:
        scene: The scene.
        outDir: Output directory (created if needed).

    Returns:
        Paths written, one per layer (``<layerName>.tif``).
    """
    root = Path(outDir)
    root.mkdir(parents=True, exist_ok=True)
    crsWkt = scene.geometry.orthoCrs().to_wkt()
    transform = scene.geometry.affine()
    attribution = "; ".join(s["attribution"] for s in scene.metadata.get("sources", []))
    datum = scene.metadata.get("verticalDatum", {}).get("name", "")
    layerInfo = scene.metadata.get("layers", {})
    paths: list[Path] = []
    for name, array in scene.layers.items():
        bands = array if array.ndim == 3 else array[np.newaxis, ...]
        count, rows, cols = bands.shape
        if count == 0:
            continue  # e.g. deck layers with maxDeckLevels=0
        isFloat = np.issubdtype(bands.dtype, np.floating)
        profile: dict[str, Any] = {
            "driver": "GTiff",
            "height": rows,
            "width": cols,
            "count": count,
            "dtype": str(bands.dtype),
            "crs": crsWkt,
            "transform": transform,
            "compress": "deflate",
            "num_threads": "ALL_CPUS",  # parallel compression (GDAL GTiff option)
            "nodata": np.nan if isFloat else None,
        }
        if rows >= _TILE and cols >= _TILE:
            profile.update(tiled=True, blockxsize=_TILE, blockysize=_TILE)
        path = root / f"{name}.tif"
        info = layerInfo.get(name, {})
        with rasterio.open(path, "w", **profile) as dst:
            # Write in row bands so huge (memory-mapped) layers never load whole.
            for row0 in range(0, rows, _BAND_ROWS):
                row1 = min(rows, row0 + _BAND_ROWS)
                dst.write(
                    np.asarray(bands[:, row0:row1, :]), window=Window(0, row0, cols, row1 - row0)
                )
            dst.update_tags(
                units=info.get("units", ""),
                description=info.get("description", ""),
                verticalDatum=datum,
                attribution=attribution,
            )
        paths.append(path)
    return paths
