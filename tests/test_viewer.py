"""Tests for ``map3d.viewer`` (static install + localhost server)."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from map3d import viewer


def _get(url: str) -> tuple[bytes, str]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never proxy localhost
    with opener.open(url, timeout=10) as response:
        return response.read(), response.headers.get("Content-Type", "")


def testInstallViewer_copiesStaticFilesAndVendoredThree(tmp_path: Path):
    index = viewer.installViewer(tmp_path)
    assert index.is_file()
    assert (tmp_path / "viewer.js").is_file()
    assert (tmp_path / "vendor" / "three" / "build" / "three.module.min.js").is_file()
    assert (tmp_path / "vendor" / "three" / "LICENSE").is_file()
    html = index.read_text(encoding="utf-8")
    assert '"three": "./vendor/three/build/three.module.min.js"' in html


def testServe_servesViewerTilesetAndCorrectMimeTypes(tmp_path: Path):
    (tmp_path / "tileset.json").write_text(
        json.dumps({"asset": {"version": "1.1"}}), encoding="utf-8"
    )
    (tmp_path / "tiles").mkdir()
    (tmp_path / "tiles" / "t.glb").write_bytes(b"glTF")
    server = viewer.serve(tmp_path, port=0, openBrowser=False)
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}/"
        body, kind = _get(base)
        assert b"map3d viewer" in body and kind.startswith("text/html")
        _, kind = _get(base + "viewer.js")
        assert kind.startswith("text/javascript")
        body, kind = _get(base + "tileset.json")
        assert json.loads(body)["asset"]["version"] == "1.1" and kind.startswith("application/json")
        _, kind = _get(base + "tiles/t.glb")
        assert kind == "model/gltf-binary"
    finally:
        server.shutdown()
        server.server_close()


def testServe_opensBrowserWhenAsked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "tileset.json").write_text("{}", encoding="utf-8")
    opened: list[str] = []
    monkeypatch.setattr(viewer.webbrowser, "open", opened.append)
    server = viewer.serve(tmp_path, port=0, openBrowser=True)
    server.shutdown()
    server.server_close()
    assert opened == [f"http://127.0.0.1:{server.server_address[1]}/"]


def testServe_requiresTileset(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="tileset.json"):
        viewer.serve(tmp_path, port=0, openBrowser=False)
