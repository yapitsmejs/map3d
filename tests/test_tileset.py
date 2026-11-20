"""Tests for ``map3d.mesh.tileset`` (3D Tiles 1.1 writer)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pygltflib
import pytest
from _support import validateTileset

from map3d.mesh import tileset as ts
from map3d.pack.synthetic import SyntheticScene, makeSyntheticPack
from map3d.request import SceneRequest
from map3d.scene import Scene, buildScene


@pytest.fixture(scope="module")
def hill(tmp_path_factory: pytest.TempPathFactory) -> SyntheticScene:
    return makeSyntheticPack(tmp_path_factory.mktemp("t") / "hill.pack", terrain="hill")


@pytest.fixture(scope="module")
def scene(hill: SyntheticScene) -> Scene:
    h = math.radians(30.0)
    axes = [[math.cos(h), -math.sin(h), 0.0], [math.sin(h), math.cos(h), 0.0], [0.0, 0.0, 1.0]]
    return buildScene(
        SceneRequest((hill.frame.lat0_deg, hill.frame.lon0_deg), 5.0, axes, 1000.0, 800.0),
        hill.packDir,
    )


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


def testTileset_canopyCanBeLeftOut(scene: Scene, tmp_path: Path):
    def maxUp(path: Path) -> float:
        tileset = json.loads(path.read_text(encoding="utf-8"))
        top = -np.inf
        for uri in _uris(tileset["root"]):
            gltf = pygltflib.GLTF2().load_binary(str(path.parent / uri))
            for prim in gltf.meshes[0].primitives:
                top = max(top, gltf.accessors[prim.attributes.POSITION].max[1])
        return top

    withTrees = ts.write3dTiles(scene, tmp_path / "a", leafSize_m=512.0, terrainGridSize=33)
    bare = ts.write3dTiles(
        scene, tmp_path / "b", leafSize_m=512.0, terrainGridSize=33, withCanopy=False
    )
    assert validateTileset(withTrees) == [] and validateTileset(bare) == []
    assert maxUp(withTrees) >= maxUp(bare)
