"""Tests for ``scripts/roadAudit.py`` (road-layer audit tool)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from map3d.pack.synthetic import SyntheticScene, makeSyntheticPack


@pytest.fixture(scope="module")
def plane(tmp_path_factory: pytest.TempPathFactory) -> SyntheticScene:
    return makeSyntheticPack(tmp_path_factory.mktemp("a") / "plane.pack")


def testRoadAudit_reportsContradictions(plane: SyntheticScene):
    """The audit runs on a synthetic pack and prints its report sections."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/roadAudit.py",
            "--pack",
            str(plane.packDir),
            "--lat",
            str(plane.frame.lat0_deg),
            "--lon",
            str(plane.frame.lon0_deg),
            "--spacing",
            "4.0",
            "--swath",
            "1000",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        timeout=600,
    )
    assert "road segments in bounds:" in result.stdout
    assert "layers covering each node:" in result.stdout
    assert "PROVABLY WRONG" in result.stdout
    assert "SAME-DIRECTION OVERLAPS" in result.stdout
    assert "DSM RESIDUAL EVIDENCE" in result.stdout  # no dsmHeight source: the fallback line
    assert result.returncode == 0


def testHeightAudit_requiresDsmSource(plane: SyntheticScene):
    """The height audit refuses to run without a dsmHeight source."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/heightAudit.py",
            "--pack",
            str(plane.packDir),
            "--lat",
            str(plane.frame.lat0_deg),
            "--lon",
            str(plane.frame.lon0_deg),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        timeout=600,
    )
    assert result.returncode != 0
    assert "no dsmHeight source" in (result.stdout + result.stderr)
