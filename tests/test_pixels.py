"""Tests for ``map3d.layers.pixels``."""

from __future__ import annotations

import numpy as np
import shapely
from _support import headingAxes

from map3d.gridGeometry import GridGeometry
from map3d.layers.pixels import toPixelGeometries
from map3d.request import SceneRequest


def _geom(heading: float = 0.0) -> GridGeometry:
    axes = headingAxes(heading)
    return GridGeometry.fromRequest(SceneRequest((1.3, 103.8), 2.0, axes, 1000.0, 600.0))


def testToPixel_centerAndCorners():
    geom = _geom()
    lat, lon, _ = geom.frame.columnPointAtHeight(
        np.array([0.0, -500.0]), np.array([0.0, -300.0]), 0.0
    )
    pts = shapely.points(lon, lat)
    px = toPixelGeometries(pts, 0.0, geom)
    np.testing.assert_allclose(shapely.get_coordinates(px), [[250.0, 150.0], [0.0, 0.0]], atol=1e-6)


def testToPixel_rotatedGridAndHeightLean():
    geom = _geom(90.0)  # grid y points east
    lat, lon, _ = geom.frame.columnPointAtHeight(np.array([100.0]), np.array([0.0]), 200.0)
    px = toPixelGeometries(shapely.points(lon, lat), np.array([200.0]), geom)
    # E = +100 m along grid y -> row offset +50 px; column unchanged at the center.
    np.testing.assert_allclose(shapely.get_coordinates(px), [[250.0, 200.0]], atol=1e-6)


def testToPixel_emptyAndNanHeight():
    geom = _geom()
    empty = np.array([], dtype=object)
    assert toPixelGeometries(empty, 0.0, geom).size == 0
    pt = shapely.points([103.8], [1.3])
    np.testing.assert_allclose(
        shapely.get_coordinates(toPixelGeometries(pt, np.array([np.nan]), geom)),
        shapely.get_coordinates(toPixelGeometries(pt, 0.0, geom)),
    )


def testRasterizePixels_chunkOffsetsAndNoIdentityTransformWarning():
    import warnings

    from map3d.layers.pixels import rasterizePixels

    square = shapely.box(1.0, 1.0, 3.0, 2.0)  # cols 1..3, rows 1..2 in pixel space
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for r0, c0 in [(0, 0), (1, 1), (1, 0), (5, 5)]:
            out = rasterizePixels([(square, 7)], (4, 4), r0, c0, allTouched=False, dtype="uint8")
            expected = np.zeros((4, 4), dtype=np.uint8)
            rows, cols = np.mgrid[r0 : r0 + 4, c0 : c0 + 4]
            expected[(rows == 1) & (cols >= 1) & (cols < 3)] = 7
            np.testing.assert_array_equal(out, expected)


def testBurnChunk_priorityOrderAndChunkOffset():
    from map3d.layers.columns import ChunkColumns
    from map3d.layers.pixels import burnChunk

    geom = _geom()
    shapes = np.array([shapely.box(10, 10, 14, 14), shapely.box(12, 12, 16, 16)], dtype=object)
    values = np.array([5, 9])
    chunk = ChunkColumns(geom, slice(8, 16), slice(8, 16))
    out = burnChunk(shapes, values, chunk, allTouched=False, dtype="uint8")
    assert out[4, 4] == 9 and out[2, 2] == 5 and out[0, 0] == 0  # higher value drawn last
    low = burnChunk(shapes, values, chunk, allTouched=False, dtype="uint8", priority=-values)
    assert low[4, 4] == 5  # an explicit priority overrides the value order
    far = ChunkColumns(geom, slice(100, 110), slice(100, 110))
    assert not burnChunk(shapes, values, far, allTouched=False, dtype="uint8").any()
