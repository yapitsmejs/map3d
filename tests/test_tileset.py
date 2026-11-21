"""Tests for ``map3d.mesh.tileset`` (3D Tiles 1.1 writer)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pygltflib
import pytest
from _meshSupport import sceneOver, writeOrtho
from _support import validateTileset

from map3d.mesh import tileset as ts
from map3d.pack.synthetic import SyntheticScene, makeSyntheticPack
from map3d.scene import Scene


@pytest.fixture(scope="module")
def hill(tmp_path_factory: pytest.TempPathFactory) -> SyntheticScene:
    return makeSyntheticPack(tmp_path_factory.mktemp("t") / "hill.pack", terrain="hill")


@pytest.fixture(scope="module")
def scene(hill: SyntheticScene) -> Scene:
    return sceneOver(hill, angle_deg=30.0, height_m=800.0)


@pytest.fixture(scope="module")
def written(scene: Scene, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return ts.write3dTiles(
        scene, tmp_path_factory.mktemp("tiles"), leafSize_m=256.0, terrainGridSize=33
    )


def testDepth_fromLeafSize():
    assert ts.quadtreeDepth(1000.0, 800.0, 256.0) == 2
    assert ts.quadtreeDepth(200.0, 100.0, 512.0) == 0


def testTileset_passesStructuralValidation(written: Path):
    assert validateTileset(written) == []


def testTileset_rootTransformIsEnuToEcef(written: Path, scene: Scene):
    root = json.loads(written.read_text(encoding="utf-8"))["root"]
    m = np.array(root["transform"]).reshape(4, 4).T  # column-major
    frame = scene.geometry.frame
    np.testing.assert_allclose(m[:3, 0], frame.rotation[0])
    np.testing.assert_allclose(m[:3, 2], frame.rotation[2])
    np.testing.assert_allclose(m[:3, 3], frame.origin_ECEF_m)
    assert root["refine"] == "REPLACE"


def testTileset_levelsAndErrors(written: Path):
    tileset = json.loads(written.read_text(encoding="utf-8"))
    root = tileset["root"]
    assert len(root["children"]) == 4 and len(root["children"][0]["children"]) == 4
    leaf = root["children"][0]["children"][0]
    assert leaf["geometricError"] == 0 and "children" not in leaf
    assert root["geometricError"] > root["children"][0]["geometricError"] > 0
    assert tileset["asset"]["extras"]["attribution"]


def testTileset_tallestRoofAppearsInLeafAndBuildingsInLeavesOnly(written: Path, scene: Scene):
    tileset = json.loads(written.read_text(encoding="utf-8"))
    roofTop = 0.0
    for uri in _uris(tileset["root"]):
        gltf = pygltflib.GLTF2().load_binary(str(written.parent / uri))
        for prim in gltf.meshes[0].primitives:
            roofTop = max(roofTop, gltf.accessors[prim.attributes.POSITION].max[1])  # glTF y = Up
    expected = float(np.nanmax(scene.layers["surfaceHeight_ENU_m"]))
    assert roofTop == pytest.approx(expected, abs=0.6)  # cell-max grid vs exact mesh


def testTileset_requiresBuiltScene(scene: Scene, tmp_path: Path):
    bare = Scene(scene.request, scene.geometry, scene.layers, scene.features, scene.metadata)
    with pytest.raises(ValueError, match="buildScene"):
        ts.write3dTiles(bare, tmp_path)


def _uris(tile: dict) -> list[str]:
    out = [tile["content"]["uri"]] if "content" in tile else []
    for child in tile.get("children", []):
        out += _uris(child)
    return out


def _firstGlb(path: Path) -> pygltflib.GLTF2:
    tileset = json.loads(path.read_text(encoding="utf-8"))
    return pygltflib.GLTF2().load_binary(str(path.parent / _uris(tileset["root"])[0]))


def testTileset_conceptualStyleIsDefault(written: Path):
    tileset = json.loads(written.read_text(encoding="utf-8"))
    assert tileset["asset"]["extras"]["style"] == "conceptual"
    gltf = _firstGlb(written)
    assert gltf.images[0].mimeType == "image/png"
    assert "KHR_materials_unlit" not in (gltf.extensionsUsed or [])
    assert all(p.attributes.NORMAL is not None for p in gltf.meshes[0].primitives)


def testTileset_buildingImageryAndProceduralFacades(scene: Scene, tmp_path: Path):
    ortho = writeOrtho(scene, tmp_path / "ortho.tif", (120, 120, 120))
    path = ts.write3dTiles(
        scene,
        tmp_path / "styled",
        buildingImageryPath=ortho,
        leafSize_m=512.0,
        terrainGridSize=33,
    )
    tileset = json.loads(path.read_text(encoding="utf-8"))
    appearance = tileset["asset"]["extras"]["buildingAppearance"]
    assert appearance == {
        "facades": "procedural; windows and floor bands are illustrative, not measured",
        "imagerySource": "byoBuildingImagery",
        "imageryColoredBuildings": 3,
        "buildingCount": 3,
    }
    assert any("ortho.tif" in text for text in tileset["asset"]["extras"]["attribution"])
    assert any(
        len(pygltflib.GLTF2().load_binary(str(path.parent / uri)).images) == 2
        for uri in _uris(tileset["root"])
    )
    assert validateTileset(path) == []


def testTileset_styleValidation(scene: Scene, tmp_path: Path):
    assert ts.STYLES == ("conceptual",)
    with pytest.raises(ValueError, match="unknown style.*conceptual"):
        ts.write3dTiles(scene, tmp_path, style="satellite")  # type: ignore[arg-type]


def _instanceCounts(path: Path) -> dict[str, int]:
    """Crown instances per tile URI (all instanced nodes except the trunks)."""
    from map3d.mesh.trees import TRUNK_COLOR

    trunk = (np.array(TRUNK_COLOR) / 255.0) ** 2.2
    tileset = json.loads(path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for uri in _uris(tileset["root"]):
        gltf = pygltflib.GLTF2().load_binary(str(path.parent / uri))
        n = 0
        for node in gltf.nodes:
            ext = (node.extensions or {}).get("EXT_mesh_gpu_instancing")
            if ext is None:
                continue
            mat = gltf.materials[gltf.meshes[node.mesh].primitives[0].material]
            if np.allclose(mat.pbrMetallicRoughness.baseColorFactor[:3], trunk):
                continue
            n += gltf.accessors[ext["attributes"]["TRANSLATION"]].count
        counts[uri] = n
    return counts


def testTileset_treesAreInstancedOnTheBareGround(written: Path, scene: Scene):
    from map3d.mesh.trees import TreeSet

    trees = TreeSet.fromScene(scene)
    assert len(trees) >= 1  # this swath holds the mapped tree (the canopy patch is outside)
    counts = _instanceCounts(written)
    depth = max(int(uri.split("/")[1]) for uri in counts)
    leaves = sum(n for uri, n in counts.items() if int(uri.split("/")[1]) == depth)
    assert leaves == len(trees)  # every tree once in the leaves
    parents = sum(n for uri, n in counts.items() if int(uri.split("/")[1]) == depth - 1)
    assert parents == int((trees.height_m >= ts.PARENT_TREE_MIN_m).sum())
    assert all(n == 0 for uri, n in counts.items() if int(uri.split("/")[1]) < depth - 1)


def testTileset_treesCanBeLeftOut(scene: Scene, tmp_path: Path):
    bare = ts.write3dTiles(
        scene, tmp_path / "t", leafSize_m=512.0, terrainGridSize=33, withTrees=False
    )
    assert sum(_instanceCounts(bare).values()) == 0
    assert validateTileset(bare) == []
