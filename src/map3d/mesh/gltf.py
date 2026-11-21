"""Minimal GLB writer for map3d tiles.

A tile's GLB holds one mesh with up to three primitives, all lit with vertex
normals so the shape reads through flat colours:

- **textured:** terrain, roofs and deck tops, draped with the tile's
  texture (an embedded PNG).
- **colored:** deck sides and piers, with per-vertex colours. Double-sided.
- **facades:** building walls with repeating procedural texture UVs and
  per-vertex tints. Double-sided.

Positions are ENU metres relative to the scene centre, written y-up as glTF
requires: ``(x, y, z) = (E, U, -N)``. 3D Tiles rotates glTF content y-up ->
z-up, giving back ``(E, N, U)``. The tileset root transform then maps ENU to
ECEF.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pygltflib
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile

__all__ = ["InstancedPart", "MeshPart", "encodePng", "enuToGltf", "vertexNormals", "writeGlb"]

FloatArray = npt.NDArray[np.float64]

_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963


@dataclass(frozen=True, eq=False)
class MeshPart:
    """Triangles in scene ENU metres, with optional UVs or vertex colours."""

    positions_ENU_m: FloatArray
    triangles: npt.NDArray[np.uint32]
    uv: npt.NDArray[np.float32] | None = None
    colors: npt.NDArray[np.uint8] | None = None

    @property
    def isEmpty(self) -> bool:
        """True when there are no triangles."""
        return int(self.triangles.shape[0]) == 0

    @classmethod
    def empty(cls, withUv: bool = False, withColors: bool = False) -> MeshPart:
        """A part with no geometry."""
        return cls(
            np.zeros((0, 3)),
            np.zeros((0, 3), dtype=np.uint32),
            np.zeros((0, 2), dtype=np.float32) if withUv else None,
            np.zeros((0, 3), dtype=np.uint8) if withColors else None,
        )

    @classmethod
    def concatenate(cls, parts: Sequence[MeshPart]) -> MeshPart:
        """Merge parts (all must agree on having UVs / colours)."""
        parts = [p for p in parts if p.positions_ENU_m.shape[0]]
        if not parts:
            return cls.empty()
        offsets = np.cumsum([0] + [p.positions_ENU_m.shape[0] for p in parts[:-1]])
        return cls(
            np.concatenate([p.positions_ENU_m for p in parts]),
            np.concatenate(
                [p.triangles + np.uint32(o) for p, o in zip(parts, offsets, strict=True)]
            ).astype(np.uint32),
            None
            if parts[0].uv is None
            else np.concatenate([p.uv for p in parts if p.uv is not None]),
            None
            if parts[0].colors is None
            else np.concatenate([p.colors for p in parts if p.colors is not None]),
        )


@dataclass(frozen=True, eq=False)
class InstancedPart:
    """One mesh drawn many times (``EXT_mesh_gpu_instancing``), e.g. tree crowns.

    Attributes:
        mesh: The shape in local ENU units (positions and triangles only).
        color: sRGB colour of its lit material.
        translations_ENU_m: Instance positions, ``(n, 3)``, scene ENU metres.
        scales: Per-instance scale along E, N, U, ``(n, 3)``.
        rotations_ENU: Optional per-instance unit quaternions ``(x, y, z, w)``
            in the ENU frame (typically a yaw about up), ``(n, 4)``.
    """

    mesh: MeshPart
    color: tuple[int, int, int]
    translations_ENU_m: FloatArray
    scales: FloatArray
    rotations_ENU: FloatArray | None = None

    @property
    def count(self) -> int:
        """Number of instances."""
        return int(self.translations_ENU_m.shape[0])


def enuToGltf(positions_ENU_m: FloatArray) -> FloatArray:
    """ENU (z-up) -> glTF (y-up): ``(E, N, U) -> (E, U, -N)``."""
    p = np.asarray(positions_ENU_m, dtype=np.float64)
    return np.column_stack([p[:, 0], p[:, 2], -p[:, 1]])


def vertexNormals(positions: FloatArray, triangles: npt.NDArray[np.uint32]) -> FloatArray:
    """Area-weighted unit vertex normals.

    Vertices shared between triangles get a smooth (averaged) normal; triangles
    with their own vertices come out flat. Unused or degenerate vertices point up.

    Args:
        positions: Vertex positions, ``(n, 3)``.
        triangles: Vertex indices, ``(m, 3)``.

    Returns:
        Unit normals, ``(n, 3)``, in the frame of ``positions``.
    """
    p = np.asarray(positions, dtype=np.float64)
    tris = np.asarray(triangles, dtype=np.int64)
    acc = np.zeros_like(p)
    if tris.shape[0]:
        face = np.cross(p[tris[:, 1]] - p[tris[:, 0]], p[tris[:, 2]] - p[tris[:, 0]])
        for k in range(3):
            np.add.at(acc, tris[:, k], face)
    length = np.linalg.norm(acc, axis=1)
    ok = length > 0
    out = np.tile(np.array([0.0, 0.0, 1.0]), (p.shape[0], 1))
    out[ok] = acc[ok] / length[ok, None]
    return out


def encodePng(rgb: npt.NDArray[np.uint8]) -> bytes:
    """Encode an RGB image as PNG bytes (for a glTF texture).

    Args:
        rgb: Image, ``(height, width, 3)`` ``uint8``.

    Returns:
        Lossless PNG bytes.
    """
    height, width, _ = rgb.shape
    with warnings.catch_warnings(), MemoryFile() as mem:
        warnings.simplefilter("ignore", NotGeoreferencedWarning)  # textures are not georeferenced
        with mem.open(driver="PNG", width=width, height=height, count=3, dtype="uint8") as dst:
            dst.write(np.moveaxis(rgb, -1, 0))
        return bytes(mem.read())


def writeGlb(
    path: str | os.PathLike[str],
    textured: MeshPart | None,
    texture: bytes | None,
    colored: MeshPart | None,
    instanced: Sequence[InstancedPart] = (),
    facades: MeshPart | None = None,
    facadeTexture: bytes | None = None,
) -> Path:
    """Write a GLB with a textured and/or a vertex-coloured primitive, plus instances.

    Args:
        path: Output path.
        textured: Part with UVs (drawn with ``texture``), or ``None``.
        texture: PNG bytes for the textured part.
        colored: Part with vertex colours, or ``None``.
        instanced: Meshes drawn once per instance, each as its own node with
            ``EXT_mesh_gpu_instancing`` (required by the file when present).
        facades: Optional textured wall part with UVs and optional vertex tints.
        facadeTexture: Repeating procedural facade PNG for ``facades``.

    Returns:
        The written path.

    Raises:
        ValueError: If everything is empty, or the textured part lacks UVs/texture.
    """
    hasTextured = textured is not None and not textured.isEmpty
    hasColored = colored is not None and not colored.isEmpty
    hasFacades = facades is not None and not facades.isEmpty
    instances = [p for p in instanced if p.count and not p.mesh.isEmpty]
    if not hasTextured and not hasColored and not hasFacades and not instances:
        raise ValueError("cannot write an empty GLB")
    if hasTextured and (textured is None or textured.uv is None or texture is None):
        raise ValueError("the textured part needs uv coordinates and a texture")
    if hasFacades and (facades is None or facades.uv is None or facadeTexture is None):
        raise ValueError("the facade part needs uv coordinates and a facade texture")

    blob = bytearray()
    gltf = pygltflib.GLTF2(scene=0, scenes=[pygltflib.Scene(nodes=[])])
    primitives: list[pygltflib.Primitive] = []

    def addView(data: bytes, target: int | None) -> int:
        while len(blob) % 4:
            blob.append(0)
        gltf.bufferViews.append(
            pygltflib.BufferView(
                buffer=0, byteOffset=len(blob), byteLength=len(data), target=target
            )
        )
        blob.extend(data)
        return len(gltf.bufferViews) - 1

    def addAccessor(
        array: npt.NDArray[np.generic],
        componentType: int,
        kind: str,
        target: int | None,
        normalized: bool = False,
        withBounds: bool = False,
    ) -> int:
        view = addView(np.ascontiguousarray(array).tobytes(), target)
        accessor = pygltflib.Accessor(
            bufferView=view,
            componentType=componentType,
            count=int(array.shape[0]),
            type=kind,
            normalized=normalized or None,
        )
        if withBounds:
            accessor.min = [float(v) for v in array.min(axis=0)]
            accessor.max = [float(v) for v in array.max(axis=0)]
        gltf.accessors.append(accessor)
        return len(gltf.accessors) - 1

    def addPrimitive(part: MeshPart, material: int) -> pygltflib.Primitive:
        attributes = pygltflib.Attributes(
            POSITION=addAccessor(
                enuToGltf(part.positions_ENU_m).astype(np.float32),
                pygltflib.FLOAT,
                pygltflib.VEC3,
                _ARRAY_BUFFER,
                withBounds=True,
            )
        )
        n = enuToGltf(vertexNormals(part.positions_ENU_m, part.triangles)).astype(np.float32)
        n /= np.linalg.norm(n, axis=1, keepdims=True)  # unit length after float32 rounding
        attributes.NORMAL = addAccessor(n, pygltflib.FLOAT, pygltflib.VEC3, _ARRAY_BUFFER)
        if part.uv is not None:
            attributes.TEXCOORD_0 = addAccessor(
                part.uv.astype(np.float32), pygltflib.FLOAT, pygltflib.VEC2, _ARRAY_BUFFER
            )
        if part.colors is not None:
            # RGBA, not RGB: glTF vertex attributes must be 4-byte aligned per element.
            rgba = np.column_stack(
                [part.colors.astype(np.uint8), np.full(part.colors.shape[0], 255, np.uint8)]
            )
            attributes.COLOR_0 = addAccessor(
                rgba,
                pygltflib.UNSIGNED_BYTE,
                pygltflib.VEC4,
                _ARRAY_BUFFER,
                normalized=True,
            )
        indices = addAccessor(
            part.triangles.astype(np.uint32).reshape(-1),
            pygltflib.UNSIGNED_INT,
            pygltflib.SCALAR,
            _ELEMENT_ARRAY_BUFFER,
        )
        return pygltflib.Primitive(attributes=attributes, indices=indices, material=material)

    def addMaterial(
        doubleSided: bool = False, texture: int | None = None, color: list[float] | None = None
    ) -> int:
        pbr = pygltflib.PbrMetallicRoughness(metallicFactor=0.0, roughnessFactor=1.0)
        if texture is not None:
            pbr.baseColorTexture = pygltflib.TextureInfo(index=texture)
        if color is not None:
            pbr.baseColorFactor = color
        gltf.materials.append(pygltflib.Material(pbrMetallicRoughness=pbr, doubleSided=doubleSided))
        return len(gltf.materials) - 1

    def addTextureMaterial(data: bytes, doubleSided: bool, repeat: bool) -> int:
        image = addView(data, None)
        imageIndex = len(gltf.images)
        gltf.images.append(pygltflib.Image(bufferView=image, mimeType="image/png"))
        samplerIndex = len(gltf.samplers)
        wrap = 10497 if repeat else 33071  # REPEAT or CLAMP_TO_EDGE
        gltf.samplers.append(
            pygltflib.Sampler(magFilter=9729, minFilter=9987, wrapS=wrap, wrapT=wrap)
        )
        textureIndex = len(gltf.textures)
        gltf.textures.append(pygltflib.Texture(source=imageIndex, sampler=samplerIndex))
        return addMaterial(doubleSided, texture=textureIndex)

    if hasTextured and textured is not None and texture is not None:
        material = addTextureMaterial(texture, doubleSided=False, repeat=False)
        primitives.append(addPrimitive(textured, material))
    if hasFacades and facades is not None and facadeTexture is not None:
        material = addTextureMaterial(facadeTexture, doubleSided=True, repeat=True)
        primitives.append(addPrimitive(facades, material))
    if hasColored and colored is not None:
        primitives.append(addPrimitive(colored, addMaterial(doubleSided=True)))

    if primitives:
        gltf.meshes.append(pygltflib.Mesh(primitives=primitives))
        gltf.nodes.append(pygltflib.Node(mesh=len(gltf.meshes) - 1))
        gltf.scenes[0].nodes.append(len(gltf.nodes) - 1)
    for part in instances:
        linear = (np.asarray(part.color, dtype=np.float64) / 255.0) ** 2.2  # factors are linear
        prim = addPrimitive(part.mesh, addMaterial(color=[*(float(c) for c in linear), 1.0]))
        gltf.meshes.append(pygltflib.Mesh(primitives=[prim]))
        s = np.asarray(part.scales, dtype=np.float64)
        attributes = {
            "TRANSLATION": addAccessor(
                enuToGltf(part.translations_ENU_m).astype(np.float32),
                pygltflib.FLOAT,
                pygltflib.VEC3,
                None,
            ),
            "SCALE": addAccessor(
                np.column_stack([s[:, 0], s[:, 2], s[:, 1]]).astype(np.float32),
                pygltflib.FLOAT,
                pygltflib.VEC3,
                None,
            ),
        }
        if part.rotations_ENU is not None:
            # ENU (x, y, z) -> glTF (x, z, -y); a yaw about ENU up is a yaw
            # about glTF +y. w is unchanged (the map is orthogonal).
            q = np.asarray(part.rotations_ENU, dtype=np.float64)
            attributes["ROTATION"] = addAccessor(
                np.column_stack([q[:, 0], q[:, 2], -q[:, 1], q[:, 3]]).astype(np.float32),
                pygltflib.FLOAT,
                pygltflib.VEC4,
                None,
            )
        gltf.nodes.append(
            pygltflib.Node(
                mesh=len(gltf.meshes) - 1,
                extensions={"EXT_mesh_gpu_instancing": {"attributes": attributes}},
            )
        )
        gltf.scenes[0].nodes.append(len(gltf.nodes) - 1)
    if instances:
        gltf.extensionsUsed.append("EXT_mesh_gpu_instancing")
        gltf.extensionsRequired.append("EXT_mesh_gpu_instancing")
    while len(blob) % 4:
        blob.append(0)
    gltf.buffers.append(pygltflib.Buffer(byteLength=len(blob)))
    gltf.set_binary_blob(bytes(blob))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    gltf.save_binary(str(out))
    return out
