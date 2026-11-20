"""Acceptance criterion 4 (SPEC section 12): scene builds never touch the network."""

from __future__ import annotations

import socket
from pathlib import Path

import numpy as np
import pytest
from _support import validateTileset

from map3d.pack.synthetic import makeSyntheticPack
from map3d.request import SceneRequest
from map3d.scene import buildScene


def testBuildAndAllWriters_workWithNetworkDisabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    syn = makeSyntheticPack(
        tmp_path / "p.pack", terrain="hill"
    )  # pack creation happens before lockdown
    attempts: list[object] = []

    def refuse(self: socket.socket, address: object) -> None:
        attempts.append(address)
        raise OSError("network access attempted during an offline build")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: refuse(None, a))  # type: ignore[arg-type]
    request = SceneRequest((syn.frame.lat0_deg, syn.frame.lon0_deg), 5.0, np.eye(3), 600.0, 600.0)
    with buildScene(request, syn.packDir, workers=4) as scene:
        scene.writeNpz(tmp_path / "out" / "scene")
        scene.writeGeoTiff(tmp_path / "out" / "geotiff")
        tileset = scene.write3dTiles(tmp_path / "out" / "tiles")
    assert attempts == []
    assert validateTileset(tileset) == []
