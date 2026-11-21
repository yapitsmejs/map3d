"""Tests for ``map3d.pack.manifest``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from map3d.pack.manifest import (
    MANIFEST_NAME,
    PACK_FORMAT_VERSION,
    Manifest,
    PackFile,
    SourceEntry,
    boundsIntersect,
    validateBounds,
)
from map3d.pack.registry import getSource

BOUNDS = (103.6, 1.2, 104.1, 1.5)


def _manifest() -> Manifest:
    manifest = Manifest.new(BOUNDS)
    entry = SourceEntry.fromSpec(getSource("fabdem"))
    entry.crs = "EPSG:4326"
    entry.files.append(PackFile("raster/fabdem/a.tif", "ab" * 32, 1234, (103.0, 1.0, 104.0, 2.0)))
    manifest.sources["fabdem"] = entry
    return manifest


def testSaveLoad_roundTrips(tmp_path: Path):
    manifest = _manifest()
    manifest.save(tmp_path)
    loaded = Manifest.load(tmp_path)
    assert loaded == manifest
    assert loaded.packFormatVersion == PACK_FORMAT_VERSION
    assert loaded.sources["fabdem"].files[0].bounds_deg == (103.0, 1.0, 104.0, 2.0)
    assert not list(tmp_path.glob("*.tmp"))


def testSave_writesAttributionFile(tmp_path: Path):
    _manifest().save(tmp_path)
    text = (tmp_path / "ATTRIBUTION.md").read_text(encoding="utf-8")
    assert "FABDEM" in text
    assert "CC BY-NC-SA 4.0" in text


def testSourceEntry_copiesRegistryFields():
    entry = SourceEntry.fromSpec(getSource("metaCanopy"))
    assert entry.role == "canopyHeight"
    assert entry.kind == "raster"
    assert entry.verticalReference == "aboveGround"
    assert entry.nativeResolution_m == 1.0
    assert entry.crs is None
    assert entry.files == []


def testLoad_missingManifestRaises(tmp_path: Path):
    with pytest.raises(ValueError, match="not a map3d pack"):
        Manifest.load(tmp_path)


def testLoad_rejectsUnknownFormatVersion(tmp_path: Path):
    _manifest().save(tmp_path)
    path = tmp_path / MANIFEST_NAME
    data = json.loads(path.read_text(encoding="utf-8"))
    data["packFormatVersion"] = 99
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="format version 99"):
        Manifest.load(tmp_path)


def testNew_rejectsBadBounds():
    with pytest.raises(ValueError, match="bounds"):
        Manifest.new((104.0, 1.2, 103.0, 1.5))


@pytest.mark.parametrize(
    "bounds",
    [
        (104.0, 1.2, 103.0, 1.5),
        (103.0, 1.5, 104.0, 1.2),
        (-181.0, 1.0, 104.0, 1.5),
        (103.0, -91.0, 104.0, 1.5),
        (103.0, 1.0, 181.0, 1.5),
        (103.0, 1.0, 104.0, 91.0),
        (103.0, 1.0, float("nan"), 1.5),
        (103.0, 1.0, 104.0),
    ],
)
def testValidateBounds_rejects(bounds: tuple[float, ...]):
    with pytest.raises(ValueError, match="bounds"):
        validateBounds(bounds)


def testValidateBounds_returnsFloatTuple():
    assert validateBounds([103, 1, 104, 2]) == (103.0, 1.0, 104.0, 2.0)


def testValidateBounds_degenerateOnlyWhenAllowed():
    point = (103.8, 1.3, 103.8, 1.3)
    with pytest.raises(ValueError, match="bounds"):
        validateBounds(point)
    assert validateBounds(point, allowDegenerate=True) == point


def testAttribution_fillsAccessedDate(tmp_path: Path):
    manifest = Manifest.new(BOUNDS)
    entry = SourceEntry.fromSpec(getSource("sgHdbBuildings"))
    entry.accessedDate = "2026-11-20"
    manifest.sources["sgHdbBuildings"] = entry
    text = manifest.attributionMarkdown()
    assert "accessed on 2026-11-20" in text
    assert "{accessed}" not in text


def testAttribution_unknownAccessedDate():
    manifest = Manifest.new(BOUNDS)
    manifest.sources["sgHdbBuildings"] = SourceEntry.fromSpec(getSource("sgHdbBuildings"))
    assert "accessed on (unknown date)" in manifest.attributionMarkdown()


def testBoundsIntersect_closedBoxes():
    singapore = (103.55, 1.13, 104.15, 1.48)
    assert boundsIntersect(BOUNDS, singapore)
    assert not boundsIntersect((100.0, 5.0, 101.0, 6.0), singapore)
    assert boundsIntersect((104.15, 1.0, 105.0, 1.13), singapore)  # touching corner
