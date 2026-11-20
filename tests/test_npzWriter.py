"""Tests for ``map3d.io.npzWriter``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from map3d.io.npzWriter import readNpz, writeNpz
from map3d.pack.synthetic import makeSyntheticPack
from map3d.request import SceneRequest
from map3d.scene import Scene, buildScene


@pytest.fixture(scope="module")
def scene(tmp_path_factory: pytest.TempPathFactory) -> Scene:
    syn = makeSyntheticPack(tmp_path_factory.mktemp("n") / "p.pack")
    req = SceneRequest((syn.frame.lat0_deg, syn.frame.lon0_deg), 10.0, np.eye(3), 500.0, 400.0)
    return buildScene(req, syn.packDir)


@pytest.mark.parametrize("compress", [False, True])
def testWriteRead_roundTrips(tmp_path: Path, scene: Scene, compress: bool):
    npzPath, jsonPath = writeNpz(scene, tmp_path / "out" / "scene", compress=compress)
    assert npzPath == tmp_path / "out" / "scene.npz"
    assert jsonPath == tmp_path / "out" / "scene.json"
    layers, features, metadata = readNpz(npzPath)
    assert set(layers) == set(scene.layers)
    for name, arr in scene.layers.items():
        np.testing.assert_array_equal(layers[name], arr)
        assert layers[name].dtype == arr.dtype
    np.testing.assert_array_equal(features["featureId"], scene.features["featureId"])
    assert list(features["sourceFeatureId"]) == list(scene.features["sourceFeatureId"])
    assert metadata["frame"] == json.loads(json.dumps(scene.metadata["frame"]))
    assert metadata["files"]["npz"] == "scene.npz"


def testWrite_loadsWithoutPickle(tmp_path: Path, scene: Scene):
    npzPath, _ = writeNpz(scene, tmp_path / "scene")
    with np.load(npzPath, allow_pickle=False) as data:
        assert "surfaceHeight_ENU_m" in data.files
        assert "features.kind" in data.files


def testWrite_acceptsExplicitNpzSuffix(tmp_path: Path, scene: Scene):
    npzPath, jsonPath = writeNpz(scene, tmp_path / "grid.npz")
    assert npzPath.name == "grid.npz" and jsonPath.name == "grid.json"
