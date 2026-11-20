"""Minimal GLB writer for map3d tiles.

A tile's GLB holds one mesh with up to two primitives:

- **textured:** terrain, roofs and deck tops, draped with the tile's
  texture (an embedded JPEG or PNG). Unlit (``KHR_materials_unlit``) for
  imagery, which already carries real light and shadow; lit, with vertex
  normals, for the conceptual style, so the shape reads through flat colours.
- **colored:** walls and deck sides, with per-vertex colours and normals. Lit
  and double-sided.

Positions are ENU metres relative to the scene centre, written y-up as glTF
requires: ``(x, y, z) = (E, U, -N)``. 3D Tiles rotates glTF content y-up ->
z-up, giving back ``(E, N, U)``. The tileset root transform then maps ENU to
ECEF.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pygltflib

__all__ = ["MeshPart", "enuToGltf", "vertexNormals", "writeGlb"]

FloatArray = npt.NDArray[np.float64]

_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963
_PNG_SIGNATURE = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])


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


def writeGlb(
    path: str | os.PathLike[str],
    textured: MeshPart | None,
    texture: bytes | None,
    colored: MeshPart | None,
    lit: bool = False,
) -> Path:
    """Write a GLB with a textured and/or a vertex-coloured primitive.

    Args:
        path: Output path.
        textured: Part with UVs (drawn with ``texture``), or ``None``.
        texture: JPEG or PNG bytes for the textured part.
        colored: Part with vertex colours, or ``None``.
        lit: Shade the textured part (vertex normals, no ``KHR_materials_unlit``).

    Returns:
        The written path.

    Raises:
        ValueError: If both parts are empty, or the textured part lacks UVs/texture.
    """
    hasTextured = textured is not None and not textured.isEmpty
    hasColored = colored is not None and not colored.isEmpty
    if not hasTextured and not hasColored:
        raise ValueError("cannot write an empty GLB")
    if hasTextured and (textured is None or textured.uv is None or texture is None):
        raise ValueError("the textured part needs uv coordinates and a texture")

    blob = bytearray()
    gltf = pygltflib.GLTF2(
        scene=0, scenes=[pygltflib.Scene(nodes=[0])], nodes=[pygltflib.Node(mesh=0)]
    )
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

    def addPrimitive(part: MeshPart, material: int, uv: bool, normals: bool) -> None:
        attributes = pygltflib.Attributes(
            POSITION=addAccessor(
                enuToGltf(part.positions_ENU_m).astype(np.float32),
                pygltflib.FLOAT,
                pygltflib.VEC3,
                _ARRAY_BUFFER,
                withBounds=True,
            )
        )
        if normals:
            n = enuToGltf(vertexNormals(part.positions_ENU_m, part.triangles)).astype(np.float32)
            n /= np.linalg.norm(n, axis=1, keepdims=True)  # unit length after float32 rounding
            attributes.NORMAL = addAccessor(n, pygltflib.FLOAT, pygltflib.VEC3, _ARRAY_BUFFER)
        if uv and part.uv is not None:
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
        primitives.append(
            pygltflib.Primitive(attributes=attributes, indices=indices, material=material)
        )

    if hasTextured and textured is not None and texture is not None:
        image = addView(texture, None)
        mime = "image/png" if texture.startswith(_PNG_SIGNATURE) else "image/jpeg"
        gltf.images.append(pygltflib.Image(bufferView=image, mimeType=mime))
        gltf.samplers.append(
            pygltflib.Sampler(magFilter=9729, minFilter=9987, wrapS=33071, wrapT=33071)
        )
        gltf.textures.append(pygltflib.Texture(source=0, sampler=0))
        gltf.materials.append(
            pygltflib.Material(
                pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                    baseColorTexture=pygltflib.TextureInfo(index=0),
                    metallicFactor=0.0,
                    roughnessFactor=1.0,
                ),
                extensions={} if lit else {"KHR_materials_unlit": {}},
            )
        )
        if not lit:
            gltf.extensionsUsed.append("KHR_materials_unlit")
        addPrimitive(textured, len(gltf.materials) - 1, uv=True, normals=lit)
    if hasColored and colored is not None:
        gltf.materials.append(
            pygltflib.Material(
                pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                    metallicFactor=0.0, roughnessFactor=1.0
                ),
                doubleSided=True,
            )
        )
        addPrimitive(colored, len(gltf.materials) - 1, uv=False, normals=True)

    gltf.meshes.append(pygltflib.Mesh(primitives=primitives))
    while len(blob) % 4:
        blob.append(0)
    gltf.buffers.append(pygltflib.Buffer(byteLength=len(blob)))
    gltf.set_binary_blob(bytes(blob))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    gltf.save_binary(str(out))
    return out
