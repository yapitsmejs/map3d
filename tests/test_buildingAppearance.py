"""Tests for imagery-derived building palettes and procedural facade materials."""

from __future__ import annotations

import colorsys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from _support import decodePng, sceneOver, writeOrtho

from map3d.mesh.buildingAppearance import (
    MAPPED_MAX_SATURATION,
    BuildingAppearance,
    facadeTexture,
)
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
    # Walls take a stable tint from a palette of plausible facade colours.
    from map3d.mesh.buildingAppearance import _NEUTRAL_WALLS

    palette = {tuple(c) for c in _NEUTRAL_WALLS.tolist()}
    assert all(tuple(w) in palette for w in appearance.wallColors.tolist())
    again = BuildingAppearance.fromScene(scene)
    np.testing.assert_array_equal(again.wallColors, appearance.wallColors)


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
    facade = np.array(["#506674"] + [None] * (n - 2) + ["#FF3333"], dtype=object)
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
    assert appearance.mappedCount == 3
    np.testing.assert_array_equal(appearance.wallColors[0], [80, 102, 116])  # mapped facade
    np.testing.assert_array_equal(appearance.roofColors[0], imagery.roofColors[0])  # imagery roof
    np.testing.assert_array_equal(appearance.roofColors[1], [242, 242, 242])  # mapped, toned
    assert not np.array_equal(appearance.wallColors[1], imagery.wallColors[1])  # re-coordinated
    # Saturated mapped colours keep their hue but are toned to the style: muted red.
    red = appearance.wallColors[n - 1].astype(int)
    assert red[0] > red[1] + 60 and red[1] == red[2]
    hue, light, sat = colorsys.rgb_to_hls(*(red / 255.0))
    assert sat == pytest.approx(MAPPED_MAX_SATURATION, abs=0.01)


def testFacadeStyles_glassMetalAndLowRaisedParts(scene: Scene):
    import dataclasses

    from map3d.mesh.buildingAppearance import FacadeStyle, facadeStyles

    b = scene.models.buildings
    n = len(b)
    assert n == 3
    styled = dataclasses.replace(
        b,
        facadeMaterial=np.array(["glass", " Metal", None], dtype=object),
        minHeight_m=np.array([0.0, 0.0, b.height_m[2] - 4.0]),
    )
    assert list(facadeStyles(styled)) == [FacadeStyle.GLASS, FacadeStyle.PLAIN, FacadeStyle.PLAIN]
    assert list(facadeStyles(b)) == [FacadeStyle.WINDOWS] * 3
    appearance = BuildingAppearance.neutral(b).withMappedColors(styled)
    counts = appearance.styleCounts()
    assert counts["glass"] == 1 and counts["plain"] == 2
    assert sum(counts.values()) == 3


def testRoofMaterial_givesATypicalColourUnlessAColourIsMapped(scene: Scene):
    import dataclasses

    b = scene.models.buildings
    tiled = dataclasses.replace(
        b,
        roofMaterial=np.array(["roof_tiles", "roof_tiles", "unobtainium"], dtype=object),
        roofColor=np.array([None, "#FFFFFF", None], dtype=object),
    )
    neutral = BuildingAppearance.neutral(b)
    appearance = neutral.withMappedColors(tiled)
    r, g, bl = appearance.roofColors[0].astype(int)
    assert r > g + 40 and r > bl + 40  # terracotta
    np.testing.assert_array_equal(appearance.roofColors[1], [242, 242, 242])  # colour wins
    np.testing.assert_array_equal(appearance.roofColors[2], neutral.roofColors[2])  # unknown
    assert appearance.mappedCount == 2


def testFacadeTextures_distinctRepeatingTiles():
    from map3d.mesh.buildingAppearance import FACADE_VARIANTS, FacadeStyle, facadeTextureFor

    tiles = {
        (s, v): decodePng(facadeTextureFor(s, v))
        for s in FacadeStyle
        for v in range(FACADE_VARIANTS)
    }
    assert all(t.shape == (96, 96, 3) for t in tiles.values())
    np.testing.assert_array_equal(tiles[(FacadeStyle.WINDOWS, 0)], decodePng(facadeTexture()))
    glass = tiles[(FacadeStyle.GLASS, 0)]
    assert glass[..., 2].mean() > glass[..., 0].mean()  # glass is bluish
    plain = tiles[(FacadeStyle.PLAIN, 0)]
    assert all(plain.std() < t.std() for k, t in tiles.items() if k[0] != FacadeStyle.PLAIN)
    lattice = tiles[(FacadeStyle.LATTICE, 0)]
    assert lattice.min() < 130 and lattice.max() > 220  # dark struts on a light ground
    # Every multi-variant style draws two different tiles, and styles differ.
    distinct = {t.tobytes() for t in tiles.values()}
    assert len(distinct) == len(FacadeStyle) * FACADE_VARIANTS - 2  # plain, lattice: one each


def testFacadeStyles_fromClassSourceHeightAndRoofMaterial(scene: Scene):
    import dataclasses

    import shapely

    from map3d.mesh.buildingAppearance import FacadeStyle, facadeStyles, facadeVariants

    b = scene.models.buildings
    box = b.geometries[0]
    round_ = shapely.buffer(shapely.centroid(box), 5e-5)  # ~5.5 m radius
    rows = [
        # (class, source, height, roof material, geometry, floors) -> style
        ("office", "overtureBuildings", 40.0, None, box, 10, FacadeStyle.RIBBON),
        ("office", "overtureBuildings", 150.0, None, box, 40, FacadeStyle.GLASS),
        ("apartments", "overtureBuildings", 40.0, None, box, 12, FacadeStyle.RESIDENTIAL),
        (None, "sgHdbBuildings", 40.0, None, box, 12, FacadeStyle.RESIDENTIAL),
        ("retail", "overtureBuildings", 12.0, None, box, 3, FacadeStyle.RETAIL),
        ("parking", "overtureBuildings", 15.0, None, box, 5, FacadeStyle.PARKING),
        ("warehouse", "overtureBuildings", 12.0, None, box, 1, FacadeStyle.INDUSTRIAL),
        ("church", "overtureBuildings", 20.0, None, box, 2, FacadeStyle.HERITAGE),
        ("church", "overtureBuildings", 60.0, None, box, 12, FacadeStyle.WINDOWS),
        (None, "overtureBuildings", 12.0, "roof_tiles", box, 3, FacadeStyle.HERITAGE),
        (None, "overtureBuildings", 150.0, None, box, None, FacadeStyle.GLASS),
        (None, "overtureBuildings", 30.0, None, box, None, FacadeStyle.WINDOWS),
        (None, "overtureBuildings", 30.0, None, round_, None, FacadeStyle.PLAIN),
        ("roof", "overtureBuildings", 5.0, None, box, None, FacadeStyle.PLAIN),
    ]
    n = len(rows)
    column = lambda k: np.array([r[k] for r in rows], dtype=object)  # noqa: E731
    styled = dataclasses.replace(
        b,
        geometries=column(4),
        sourceIds=np.array([r[1] for r in rows]),
        sourceFeatureIds=np.array([f"f{k}" for k in range(n)]),
        height_m=np.array([r[2] for r in rows]),
        minHeight_m=np.zeros(n),
        roofMaterial=column(3),
        facadeMaterial=np.full(n, None, dtype=object),
        buildingClass=column(0),
        numFloors=np.array([np.nan if r[5] is None else r[5] for r in rows], dtype=float),
        roofs=(None,) * n,
        heightSource=np.zeros(n, dtype=np.uint8),
        baseOrthometric_m=np.zeros(n),
        roofOrthometric_m=np.array([r[2] for r in rows]),
        anchorOrthometric_m=np.zeros(n),
    )
    assert [FacadeStyle(s) for s in facadeStyles(styled)] == [r[6] for r in rows]
    variants = facadeVariants(styled)
    assert set(variants.tolist()) == {0, 1}  # both variants in use
    np.testing.assert_array_equal(variants, facadeVariants(styled))  # stable
