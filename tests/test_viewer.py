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


def testVendoredViewerFiles_areNotGitIgnored():
    # A blanket `build/` ignore once excluded vendor/three/build/, silently dropping
    # three.module.min.js from git and from wheels (hatchling honours .gitignore).
    import shutil
    import subprocess

    git = shutil.which("git")
    if git is None:
        pytest.skip("git not available")
    repo = Path(__file__).resolve().parents[1]
    files = [str(p.relative_to(repo)) for p in viewer.STATIC_DIR.rglob("*") if p.is_file()]
    result = subprocess.run([git, "check-ignore", *files], cwd=repo, capture_output=True, text=True)
    assert result.stdout.strip() == "", f"git-ignored viewer files: {result.stdout}"


def testServe_refusesAPortAlreadyInUse(tmp_path: Path):
    # On Windows, SO_REUSEADDR would let a second server share the port and
    # silently split requests between them (stale viewers answering).
    (tmp_path / "tileset.json").write_text(json.dumps({"asset": {"version": "1.1"}}), "utf-8")
    first = viewer.serve(tmp_path, port=0, openBrowser=False)
    try:
        with pytest.raises(OSError):
            viewer.serve(tmp_path, port=first.server_address[1], openBrowser=False)
    finally:
        first.shutdown()
        first.server_close()
