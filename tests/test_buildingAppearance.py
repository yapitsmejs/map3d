"""Tests for imagery-derived building palettes and procedural facade materials."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from _support import decodePng, sceneOver, writeOrtho

from map3d.mesh.buildingAppearance import BuildingAppearance, facadeTexture
from map3d.pack.synthetic import SyntheticScene, makeSyntheticPack
from map3d.scene import Scene


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> SyntheticScene:
    return makeSyntheticPack(tmp_path_factory.mktemp("appearance") / "scene.pack")


@pytest.fixture(scope="module")
def scene(synthetic: SyntheticScene) -> Scene:
    return sceneOver(synthetic)


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
    path = writeOrtho(scene, tmp_path / "ortho.tif", (160, 100, 80))
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
    path = writeOrtho(scene, tmp_path / "ortho.tif", colour)
    appearance = BuildingAppearance.fromScene(scene, buildingImageryPath=path)
    assert appearance.coloredCount == 0
    np.testing.assert_array_equal(appearance.roofColors[0], [214, 214, 212])


def testBuildingAppearance_coarseImageryFallsBack(scene: Scene, tmp_path: Path):
    path = writeOrtho(scene, tmp_path / "coarse.tif", (160, 100, 80), size=1)
    appearance = BuildingAppearance.fromScene(scene, buildingImageryPath=path)
    assert appearance.coloredCount == 0


def testBuildingAppearance_requiresRgbImagery(scene: Scene, tmp_path: Path):
    path = writeOrtho(scene, tmp_path / "grey.tif", (100, 100, 100), bands=2)
    with pytest.raises(ValueError, match="RGB"):
        BuildingAppearance.fromScene(scene, buildingImageryPath=path)
    with pytest.raises(ValueError, match="not found"):
        BuildingAppearance.fromScene(scene, buildingImageryPath=tmp_path / "missing.tif")


def testFacadeTexture_hasRepeatedWindowAndFloorBand():
    image = decodePng(facadeTexture())
    assert image.shape == (96, 96, 3)
    assert np.all(image[0, 0] == 255)
    assert image[30, 35].mean() < 100  # glass
    assert image[-2, 0].mean() < 240  # floor band


def testParseColor_hexShortHexNamesAndJunk():
    from map3d.mesh.buildingAppearance import parseColor

    np.testing.assert_array_equal(parseColor("#5D7877"), [93, 120, 119])
    np.testing.assert_array_equal(parseColor("5d7877"), [93, 120, 119])
    np.testing.assert_array_equal(parseColor("#fa0"), [255, 170, 0])
    np.testing.assert_array_equal(parseColor(" Light Grey "), [211, 211, 211])
    for junk in (None, "", "#12345", "sparkly", 42):
        assert parseColor(junk) is None


def testBuildingAppearance_mappedColoursWinOverImagery(scene: Scene, tmp_path: Path):
    import dataclasses

    b = scene.models.buildings
    n = len(b)
    facade = np.array(["#506674"] + [None] * (n - 1), dtype=object)
    roof = np.array([None, "white"] + [None] * (n - 2), dtype=object)
    mapped = dataclasses.replace(
        scene,
        models=dataclasses.replace(
            scene.models, buildings=dataclasses.replace(b, facadeColor=facade, roofColor=roof)
        ),
    )
    ortho = writeOrtho(scene, tmp_path / "ortho.tif", (170, 90, 60), size=400)
    imagery = BuildingAppearance.fromScene(scene, buildingImageryPath=ortho)
    appearance = BuildingAppearance.fromScene(mapped, buildingImageryPath=ortho)
    assert appearance.mappedCount == 2
    np.testing.assert_array_equal(appearance.wallColors[0], [80, 102, 116])  # mapped facade
    np.testing.assert_array_equal(appearance.roofColors[0], imagery.roofColors[0])  # imagery roof
    np.testing.assert_array_equal(appearance.roofColors[1], [255, 255, 255])  # mapped roof
    assert not np.array_equal(appearance.wallColors[1], imagery.wallColors[1])  # re-coordinated
    np.testing.assert_array_equal(appearance.wallColors[2], imagery.wallColors[2])  # unmapped
