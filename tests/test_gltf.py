"""Tests for ``map3d.mesh.gltf`` (GLB writer)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygltflib
import pytest

from map3d.mesh.gltf import MeshPart, enuToGltf, writeGlb


def _square(z: float = 0.0, uv: bool = True, color: bool = False) -> MeshPart:
    pos = np.array([[0, 0, z], [10, 0, z], [10, 10, z], [0, 10, z]], dtype=np.float64)
    return MeshPart(
        positions_ENU_m=pos,
        triangles=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
        uv=np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32) if uv else None,
        colors=np.full((4, 3), 200, dtype=np.uint8) if color else None,
    )


def testEnuToGltf_isYUp():
    np.testing.assert_array_equal(enuToGltf(np.array([[1.0, 2.0, 3.0]])), [[1.0, 3.0, -2.0]])


def testMeshPart_concatenateAndEmpty():
    a, b = _square(0.0), _square(5.0)
    both = MeshPart.concatenate([a, b, MeshPart.empty(withUv=True)])
    assert both.positions_ENU_m.shape == (8, 3)
    np.testing.assert_array_equal(both.triangles[2], [4, 5, 6])
    assert both.uv is not None and both.uv.shape == (8, 2)
    assert MeshPart.concatenate([]).isEmpty


def testWriteGlb_texturedAndColoredPrimitives(tmp_path: Path):
    jpeg = b"\xff\xd8\xff\xe0fakejpeg\xff\xd9"
    path = writeGlb(
        tmp_path / "t.glb",
        textured=_square(1.0),
        texture=jpeg,
        colored=_square(2.0, uv=False, color=True),
    )
    raw = path.read_bytes()
    assert raw[:4] == b"glTF" and int.from_bytes(raw[4:8], "little") == 2
    gltf = pygltflib.GLTF2().load_binary(str(path))
    prims = gltf.meshes[0].primitives
    assert len(prims) == 2
    pos = gltf.accessors[prims[0].attributes.POSITION]
    assert pos.count == 4
    assert pos.min == pytest.approx([0.0, 1.0, -10.0]) and pos.max == pytest.approx(
        [10.0, 1.0, 0.0]
    )
    assert prims[0].attributes.TEXCOORD_0 is not None
    assert prims[1].attributes.COLOR_0 is not None
    color = gltf.accessors[prims[1].attributes.COLOR_0]
    assert color.normalized is True
    assert color.type == "VEC4"  # 4-byte aligned elements (glTF requirement)
    assert "KHR_materials_unlit" in gltf.extensionsUsed
    assert gltf.images[0].mimeType == "image/jpeg"
    assert gltf.materials[prims[1].material].doubleSided is True
    blob = gltf.binary_blob()
    view = gltf.bufferViews[gltf.images[0].bufferView]
    assert blob[view.byteOffset : view.byteOffset + view.byteLength] == jpeg


def testWriteGlb_onlyColoredOrNothing(tmp_path: Path):
    path = writeGlb(
        tmp_path / "c.glb", textured=None, texture=None, colored=_square(uv=False, color=True)
    )
    gltf = pygltflib.GLTF2().load_binary(str(path))
    assert len(gltf.meshes[0].primitives) == 1 and not gltf.images
    with pytest.raises(ValueError, match="empty"):
        writeGlb(tmp_path / "e.glb", textured=None, texture=None, colored=None)


def testWriteGlb_texturedPartNeedsUvAndTexture(tmp_path: Path):
    with pytest.raises(ValueError, match="uv"):
        writeGlb(tmp_path / "x.glb", textured=_square(uv=False), texture=b"x", colored=None)
