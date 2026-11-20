"""Design-point benchmark (SPEC section 11): 5 km x 5 km at 0.5 m, K = 5.

Skipped unless ``MAP3D_BENCH_PACK`` points at a pack covering the scene
(``MAP3D_BENCH_CENTER="lat,lon"``, default Marina Bay). Run with
``uv run pytest -m slow``.
"""

from __future__ import annotations

import os
import time

import pytest

from map3d import SceneRequest, buildScene

PACK = os.environ.get("MAP3D_BENCH_PACK")


@pytest.mark.slow
@pytest.mark.skipif(PACK is None, reason="set MAP3D_BENCH_PACK to run the design-point benchmark")
def testDesignPoint_buildsUnderBudget(tmp_path):
    lat, lon = (float(v) for v in os.environ.get("MAP3D_BENCH_CENTER", "1.285,103.86").split(","))
    request = SceneRequest((lat, lon), 0.5, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 5000.0, 5000.0)
    t0 = time.perf_counter()
    with buildScene(request, PACK) as scene:
        build_s = time.perf_counter() - t0
        assert scene.geometry.nRows * scene.geometry.nCols == 100_000_000
        scene.writeNpz(tmp_path / "scene")
        scene.writeGeoTiff(tmp_path / "geotiff")
    total_s = time.perf_counter() - t0
    assert build_s < 120.0
    assert total_s < 300.0  # grid share of the 10-minute budget (3D Tiles is M7)
