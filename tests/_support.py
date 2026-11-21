"""Shared test helpers (importable from tests: ``from _support import ...``)."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

from map3d.pack.synthetic import SyntheticScene
from map3d.prepare.http import HttpSession
from map3d.request import SceneRequest
from map3d.scene import Scene, buildScene


def headingAxes(heading_deg: float = 0.0) -> np.ndarray:
    """Grid axes in ENU: grid y heads ``heading_deg`` clockwise of North, x to its right, z Up."""
    h = math.radians(heading_deg)
    return np.array(
        [[math.cos(h), -math.sin(h), 0.0], [math.sin(h), math.cos(h), 0.0], [0.0, 0.0, 1.0]]
    )


def rangeServer(files: dict[str, bytes], log: list[httpx.Request] | None = None):
    """MockTransport handler serving HEAD and byte-range GETs for ``files``."""

    def handler(request: httpx.Request) -> httpx.Response:
        if log is not None:
            log.append(request)
        body = files.get(str(request.url))
        if body is None:
            return httpx.Response(404)
        if request.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(body))})
        rng = request.headers.get("range")
        if rng:
            start, end = (int(v) for v in rng.removeprefix("bytes=").split("-"))
            part = body[start : end + 1]
            return httpx.Response(
                206,
                content=part,
                headers={"content-range": f"bytes {start}-{start + len(part) - 1}/{len(body)}"},
            )
        return httpx.Response(200, content=body)

    return handler


def validateTileset(tilesetPath) -> list[str]:
    """Structural 3D Tiles 1.1 checks used by the tests (the official validator needs Node).

    Checks: asset version; geometric errors non-negative and non-increasing down
    the tree; box volumes well-formed and containing their children (with slack)
    and every content vertex; content URIs resolve to GLB 2.0 files that parse.
    """
    import json
    from pathlib import Path

    import numpy as np
    import pygltflib

    problems: list[str] = []
    path = Path(tilesetPath)
    tileset = json.loads(path.read_text(encoding="utf-8"))
    if tileset.get("asset", {}).get("version") != "1.1":
        problems.append("asset.version must be '1.1'")
    root = tileset["root"]
    if tileset.get("geometricError", -1) < root.get("geometricError", 0):
        problems.append("tileset geometricError must be >= root geometricError")
    if root.get("refine") not in ("ADD", "REPLACE"):
        problems.append("root.refine must be ADD or REPLACE")
    if "transform" in root and len(root["transform"]) != 16:
        problems.append("root.transform must have 16 elements")

    def boxBounds(box):
        c = np.array(box[:3])
        half = np.abs(np.array(box[3:12]).reshape(3, 3)).sum(axis=0)
        return c - half, c + half

    def visit(tile, parentError, parentBox, where):
        err = tile.get("geometricError", -1)
        if err < 0 or err > parentError + 1e-9:
            problems.append(f"{where}: geometricError {err} invalid (parent {parentError})")
        box = tile.get("boundingVolume", {}).get("box")
        if box is None or len(box) != 12:
            problems.append(f"{where}: boundingVolume.box must have 12 numbers")
            return
        lo, hi = boxBounds(box)
        if parentBox is not None:
            plo, phi = parentBox
            if (lo < plo - 1.0).any() or (hi > phi + 1.0).any():
                problems.append(f"{where}: box not inside parent box")
        content = tile.get("content")
        if content:
            glbPath = path.parent / content["uri"]
            if not glbPath.is_file():
                problems.append(f"{where}: missing content {content['uri']}")
            else:
                raw = glbPath.read_bytes()
                if raw[:4] != b"glTF" or int.from_bytes(raw[4:8], "little") != 2:
                    problems.append(f"{where}: not a GLB 2.0 file")
                gltf = pygltflib.GLTF2().load_binary(str(glbPath))
                blob = gltf.binary_blob()

                def floats(index):
                    acc = gltf.accessors[index]
                    view = gltf.bufferViews[acc.bufferView]
                    data = blob[view.byteOffset : view.byteOffset + view.byteLength]
                    return np.frombuffer(data, np.float32).reshape(acc.count, -1)

                for node in gltf.nodes:
                    inst = (node.extensions or {}).get("EXT_mesh_gpu_instancing")
                    for prim in gltf.meshes[node.mesh].primitives:
                        acc = gltf.accessors[prim.attributes.POSITION]
                        amin, amax = np.array(acc.min), np.array(acc.max)
                        if inst is None:
                            gmin, gmax = amin, amax
                        else:  # every instance: translation + scale * local bounds
                            t = floats(inst["attributes"]["TRANSLATION"])
                            sc = floats(inst["attributes"]["SCALE"])
                            corners = np.stack([t + sc * amin, t + sc * amax])
                            gmin, gmax = corners.min(axis=(0, 1)), corners.max(axis=(0, 1))
                        # glTF y-up (E, U, -N) -> z-up ENU (E, N, U)
                        mn = np.array([gmin[0], -gmax[2], gmin[1]])
                        mx = np.array([gmax[0], -gmin[2], gmax[1]])
                        if (mn < lo - 1e-3).any() or (mx > hi + 1e-3).any():
                            problems.append(f"{where}: content outside its bounding box")
        for k, child in enumerate(tile.get("children", [])):
            visit(child, err, (lo, hi), f"{where}/{k}")

    visit(root, tileset.get("geometricError", 0), None, "root")
    return problems


def mockSession(handler, **kw: Any) -> HttpSession:
    """An ``HttpSession`` over an ``httpx.MockTransport`` handler, with no retry sleeps."""
    kw.setdefault("sleep", lambda s: None)
    return HttpSession(transport=httpx.MockTransport(handler), **kw)


def writeBands(path: Path, data: np.ndarray, crs: str, transform, nodata=None) -> Path:
    """Write a ``(bands, rows, cols)`` float32 GeoTIFF."""
    count, height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data.astype(np.float32))
    return path


def sceneOver(synthetic: SyntheticScene, angle_deg: float = 0.0, height_m: float = 1000.0) -> Scene:
    """A 1 km wide, 5 m scene centred on a synthetic pack, its grid rotated ``angle_deg``."""
    axes = headingAxes(angle_deg)
    centre = (synthetic.frame.lat0_deg, synthetic.frame.lon0_deg)
    return buildScene(SceneRequest(centre, 5.0, axes, 1000.0, height_m), synthetic.packDir)


def writeOrtho(
    scene: Scene, path: Path, colour: tuple[int, int, int], size: int = 100, bands: int = 3
) -> Path:
    """A uniform-colour orthophoto GeoTIFF (EPSG:4326) covering the scene with 50 m margin."""
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


def decodePng(data: bytes) -> np.ndarray:
    """PNG bytes -> ``(height, width, 3)`` pixels."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    with warnings.catch_warnings(), MemoryFile(data) as mem:
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with mem.open() as src:
            return np.moveaxis(src.read(), 0, -1)
