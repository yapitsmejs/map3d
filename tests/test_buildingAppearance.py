"""Tests for imagery-derived building palettes and procedural facade materials."""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

from map3d.mesh.buildingAppearance import BuildingAppearance, facadeTexture
from map3d.pack.synthetic import SyntheticScene, makeSyntheticPack
from map3d.request import SceneRequest
from map3d.scene import Scene, buildScene


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> SyntheticScene:
    return makeSyntheticPack(tmp_path_factory.mktemp("appearance") / "scene.pack")


@pytest.fixture(scope="module")
def scene(synthetic: SyntheticScene) -> Scene:
    return buildScene(
        SceneRequest(
            (synthetic.frame.lat0_deg, synthetic.frame.lon0_deg),
            5.0,
            np.eye(3),
            1000.0,
            1000.0,
        ),
        synthetic.packDir,
    )


def _ortho(
    scene: Scene, path: Path, colour: tuple[int, int, int], size: int = 100, bands: int = 3
) -> Path:
    bounds = scene.geometry.footprintBounds_deg(margin_m=50.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=bands,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_bounds(*bounds, size, size),
    ) as dst:
        data = np.zeros((bands, size, size), dtype=np.uint8)
        data[:3] = np.array(colour, dtype=np.uint8)[:bands, None, None]
        dst.write(data)
    return path


def testBuildingAppearance_neutralWhenPackHasNoImagery(scene: Scene):
    appearance = BuildingAppearance.fromScene(scene)
    assert appearance.sourceId is None
    assert appearance.attribution is None
    assert appearance.coloredCount == 0
    np.testing.assert_array_equal(appearance.roofColors[0], [214, 214, 212])
    np.testing.assert_array_equal(appearance.wallColors[0], [240, 240, 244])


def testBuildingAppearance_unavailablePackCoverageFallsBack(scene: Scene, monkeypatch):
    entry = SimpleNamespace(
        sourceId="sentinel2Composite",
        attribution="Sentinel-2",
        accessedDate="2026-01-01",
    )
    monkeypatch.setattr(scene.pack, "primarySource", lambda role: entry)

    def unavailable(*args, **kwargs):
        raise ValueError("imagery has no files intersecting the scene")

    monkeypatch.setattr(scene.pack, "readRaster", unavailable)
    appearance = BuildingAppearance.fromScene(scene)
    assert appearance.coloredCount == 0
    assert appearance.sourceId is None


def testBuildingAppearance_orthoDeterminesRoofAndCoordinatedWall(scene: Scene, tmp_path: Path):
    path = _ortho(scene, tmp_path / "ortho.tif", (160, 100, 80))
    first = BuildingAppearance.fromScene(scene, buildingImageryPath=path)
    second = BuildingAppearance.fromScene(scene, buildingImageryPath=path)
    assert first.sourceId == "byoBuildingImagery"
    assert first.attribution is not None and "ortho.tif" in first.attribution
    assert first.coloredCount == len(scene.models.buildings)
    np.testing.assert_array_equal(first.roofColors, second.roofColors)
    np.testing.assert_array_equal(first.wallColors, second.wallColors)
    np.testing.assert_array_equal(first.roofColors[0], [160, 100, 80])
    assert np.all(first.wallColors[0].astype(int) > first.roofColors[0])


@pytest.mark.parametrize("colour", [(20, 180, 20), (10, 10, 10)])
def testBuildingAppearance_rejectsVegetationAndShadows(
    scene: Scene, tmp_path: Path, colour: tuple[int, int, int]
):
    path = _ortho(scene, tmp_path / "ortho.tif", colour)
    appearance = BuildingAppearance.fromScene(scene, buildingImageryPath=path)
    assert appearance.coloredCount == 0
    np.testing.assert_array_equal(appearance.roofColors[0], [214, 214, 212])


def testBuildingAppearance_coarseImageryFallsBack(scene: Scene, tmp_path: Path):
    path = _ortho(scene, tmp_path / "coarse.tif", (160, 100, 80), size=1)
    appearance = BuildingAppearance.fromScene(scene, buildingImageryPath=path)
    assert appearance.coloredCount == 0


def testBuildingAppearance_requiresRgbImagery(scene: Scene, tmp_path: Path):
    path = _ortho(scene, tmp_path / "grey.tif", (100, 100, 100), bands=2)
    with pytest.raises(ValueError, match="RGB"):
        BuildingAppearance.fromScene(scene, buildingImageryPath=path)
    with pytest.raises(ValueError, match="not found"):
        BuildingAppearance.fromScene(scene, buildingImageryPath=tmp_path / "missing.tif")


def testFacadeTexture_hasRepeatedWindowAndFloorBand():
    with warnings.catch_warnings(), MemoryFile(facadeTexture()) as mem:
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with mem.open() as src:
            image = np.moveaxis(src.read(), 0, -1)
    assert image.shape == (96, 96, 3)
    assert np.all(image[0, 0] == 255)
    assert image[30, 35].mean() < 100  # glass
    assert image[-2, 0].mean() < 240  # floor band
