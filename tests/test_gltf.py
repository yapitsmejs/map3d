"""Tests for ``map3d.mesh.gltf`` (GLB writer)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygltflib
import pytest

from map3d.mesh.gltf import MeshPart, enuToGltf, writeGlb

_PNG = b"\x89PNG\r\n\x1a\nfakepng"


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
    path = writeGlb(
        tmp_path / "t.glb",
        textured=_square(1.0),
        texture=_PNG,
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
    assert gltf.images[0].mimeType == "image/png"
    assert gltf.materials[prims[1].material].doubleSided is True
    blob = gltf.binary_blob()
    view = gltf.bufferViews[gltf.images[0].bufferView]
    assert blob[view.byteOffset : view.byteOffset + view.byteLength] == _PNG


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


def testVertexNormals_flatAndSmooth():
    from map3d.mesh.gltf import vertexNormals

    flat = vertexNormals(_square(3.0).positions_ENU_m, _square(3.0).triangles)
    np.testing.assert_allclose(flat, np.tile([0.0, 0.0, 1.0], (4, 1)))
    # A ridge: two faces sloping 45 degrees either way share the top edge.
    pos = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0]], float)
    tris = np.array([[0, 1, 3], [0, 3, 2], [3, 5, 4], [3, 4, 2]], dtype=np.uint32)
    normals = vertexNormals(pos, tris)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)
    np.testing.assert_allclose(normals[2], [0.0, 0.0, 1.0], atol=1e-12)  # averaged on the ridge
    np.testing.assert_allclose(normals[0], [0.0, -np.sqrt(0.5), np.sqrt(0.5)])
    # Degenerate (unused) vertices get "up".
    np.testing.assert_allclose(
        vertexNormals(np.zeros((1, 3)), np.zeros((0, 3), np.uint32)), [[0, 0, 1]]
    )


def testWriteGlb_everyPrimitiveIsLitWithNormals(tmp_path: Path):
    path = writeGlb(
        tmp_path / "lit.glb",
        textured=_square(1.0),
        texture=_PNG,
        colored=_square(2.0, uv=False, color=True),
    )
    gltf = pygltflib.GLTF2().load_binary(str(path))
    assert "KHR_materials_unlit" not in (gltf.extensionsUsed or [])
    blob = gltf.binary_blob()
    for prim in gltf.meshes[0].primitives:
        acc = gltf.accessors[prim.attributes.NORMAL]
        assert acc.type == "VEC3" and acc.componentType == pygltflib.FLOAT
        view = gltf.bufferViews[acc.bufferView]
        normals = np.frombuffer(
            blob[view.byteOffset : view.byteOffset + view.byteLength], np.float32
        ).reshape(-1, 3)
        np.testing.assert_allclose(normals, np.tile([0.0, 1.0, 0.0], (4, 1)))  # up is +y in glTF


def testWriteGlb_instancedPartsUseGpuInstancing(tmp_path: Path):
    from map3d.mesh.gltf import InstancedPart

    unit = MeshPart(
        positions_ENU_m=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
        triangles=np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.uint32),
    )
    part = InstancedPart(
        unit,
        color=(90, 160, 70),
        translations_ENU_m=np.array([[10.0, 20.0, 5.0], [-3.0, 4.0, 1.0]]),
        scales=np.array([[2.0, 2.0, 3.0], [1.0, 1.0, 1.5]]),
        rotations_ENU=np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.6, 0.8]]),
    )
    path = writeGlb(tmp_path / "i.glb", _square(1.0), _PNG, None, instanced=[part, part])
    gltf = pygltflib.GLTF2().load_binary(str(path))
    assert "EXT_mesh_gpu_instancing" in gltf.extensionsUsed
    assert "EXT_mesh_gpu_instancing" in gltf.extensionsRequired
    assert len(gltf.scenes[0].nodes) == 3 and len(gltf.meshes) == 3
    node = gltf.nodes[1]
    ext = node.extensions["EXT_mesh_gpu_instancing"]["attributes"]
    blob = gltf.binary_blob()

    def read(index: int) -> np.ndarray:
        acc = gltf.accessors[index]
        view = gltf.bufferViews[acc.bufferView]
        raw = blob[view.byteOffset : view.byteOffset + view.byteLength]
        return np.frombuffer(raw, np.float32).reshape(acc.count, -1)

    np.testing.assert_allclose(read(ext["TRANSLATION"]), [[10, 5, -20], [-3, 1, -4]])  # y-up
    np.testing.assert_allclose(read(ext["SCALE"]), [[2, 3, 2], [1, 1.5, 1]])
    # ENU yaw (x, y, z, w) -> glTF yaw about +y: (x, z, -y, w).
    np.testing.assert_allclose(read(ext["ROTATION"]), [[0, 0, 0, 1], [0, 0.6, 0, 0.8]], atol=1e-6)
    prim = gltf.meshes[node.mesh].primitives[0]
    assert prim.attributes.NORMAL is not None
    mat = gltf.materials[prim.material]
    linear = (np.array([90, 160, 70]) / 255) ** 2.2  # glTF colour factors are linear
    np.testing.assert_allclose(mat.pbrMetallicRoughness.baseColorFactor[:3], linear, rtol=0.05)


def testWriteGlb_texturedFacadesRepeatAndKeepVertexColors(tmp_path: Path):
    facade = _square(2.0, uv=True, color=True)
    path = writeGlb(
        tmp_path / "f.glb",
        textured=_square(1.0),
        texture=_PNG,
        colored=None,
        facades=facade,
        facadeTexture=_PNG,
    )
    gltf = pygltflib.GLTF2().load_binary(str(path))
    assert len(gltf.images) == 2
    assert len(gltf.meshes[0].primitives) == 2
    primitive = gltf.meshes[0].primitives[1]
    assert primitive.attributes.TEXCOORD_0 is not None
    assert primitive.attributes.COLOR_0 is not None
    assert primitive.attributes.NORMAL is not None
    material = gltf.materials[primitive.material]
    assert material.doubleSided is True
    sampler = gltf.samplers[
        gltf.textures[material.pbrMetallicRoughness.baseColorTexture.index].sampler
    ]
    assert sampler.wrapS == 10497 and sampler.wrapT == 10497
    with pytest.raises(ValueError, match="facade"):
        writeGlb(tmp_path / "bad.glb", None, None, None, facades=facade)


def testWriteGlb_onlyInstancedIsAllowed(tmp_path: Path):
    from map3d.mesh.gltf import InstancedPart

    part = InstancedPart(_square(0.0, uv=False), (1, 2, 3), np.zeros((1, 3)), np.ones((1, 3)))
    gltf = pygltflib.GLTF2().load_binary(
        str(writeGlb(tmp_path / "o.glb", None, None, None, instanced=[part]))
    )
    assert len(gltf.meshes) == 1 and gltf.nodes[0].extensions
    empty = InstancedPart(part.mesh, part.color, np.zeros((0, 3)), np.zeros((0, 3)))
    with pytest.raises(ValueError, match="empty"):
        writeGlb(tmp_path / "e.glb", None, None, None, instanced=[empty])


def testWriteGlb_otherFacadesGetTheirOwnTextures(tmp_path: Path):
    facade = MeshPart(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]]),
        np.array([[0, 1, 2]], dtype=np.uint32),
        uv=np.zeros((3, 2), dtype=np.float32),
        colors=np.full((3, 3), 200, dtype=np.uint8),
    )
    path = writeGlb(
        tmp_path / "o.glb",
        textured=None,
        texture=None,
        colored=None,
        otherFacades=[(facade, _PNG), (facade, _PNG), (MeshPart.empty(withUv=True), _PNG)],
    )
    gltf = pygltflib.GLTF2().load_binary(str(path))
    assert len(gltf.meshes[0].primitives) == 2  # the empty part is skipped
    assert len(gltf.images) == 2
    assert all(m.doubleSided for m in gltf.materials)
    noUv = MeshPart(facade.positions_ENU_m, facade.triangles)
    with pytest.raises(ValueError, match="facade part needs uv"):
        writeGlb(tmp_path / "x.glb", None, None, None, otherFacades=[(noUv, _PNG)])
