"""Tests for ``map3d.prepare.pipeline`` (orchestration with fake fetchers)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from map3d.pack.manifest import Manifest
from map3d.prepare import pipeline

SG = (103.80, 1.28, 103.84, 1.32)
OUTSIDE_SG = (100.0, 5.0, 100.1, 5.1)


@pytest.fixture
def fakes(writeRaster, writeBuildings):
    def raster(name: str):
        def fetch(ctx: pipeline.PrepareContext) -> list[Path]:
            b = ctx.bounds
            pad = (b[0] - 0.01, b[1] - 0.01, b[2] + 0.01, b[3] + 0.01)
            return [
                writeRaster(
                    ctx.workDir(name) / f"{name}.tif", pad, values=np.full((8, 8), 5.0, np.float32)
                )
            ]

        return fetch

    def buildings(ctx: pipeline.PrepareContext) -> list[Path]:
        b = ctx.bounds
        return [
            writeBuildings(
                ctx.workDir("b") / "b.parquet",
                [(b[0] + 0.001, b[1] + 0.001, b[0] + 0.002, b[1] + 0.002)],
            )
        ]

    return {
        "fabdem": raster("fabdem"),
        "copernicusGlo30": raster("glo30"),
        "egm2008": raster("egm2008"),
        "overtureBuildings": buildings,
        "sgHdbBuildings": buildings,
        "metaCanopy": lambda ctx: [],
        "esaWorldCover": lambda ctx: (_ for _ in ()).throw(ConnectionError("network down")),
    }


def testPrepare_ingestsRecordsSkipsAndFailures(tmp_path: Path, fakes):
    report = pipeline.preparePack(
        SG,
        tmp_path / "p.pack",
        sources=["fabdem", "egm2008", "overtureBuildings", "metaCanopy", "esaWorldCover"],
        cacheDir=tmp_path / "cache",
        fetchers=fakes,
    )
    assert report.ingested == {"fabdem": 1, "egm2008": 1, "overtureBuildings": 1}
    assert report.skipped == {"metaCanopy": "no data in bounds"}
    assert "network down" in report.failed["esaWorldCover"]
    manifest = Manifest.load(tmp_path / "p.pack")
    assert manifest.bounds_deg == SG
    assert manifest.sources["fabdem"].accessedDate == datetime.now(UTC).date().isoformat()


def testPrepare_skipsSingaporeSourcesElsewhere(tmp_path: Path, fakes):
    report = pipeline.preparePack(
        OUTSIDE_SG,
        tmp_path / "p.pack",
        sources=["fabdem", "egm2008", "sgHdbBuildings"],
        cacheDir=tmp_path / "cache",
        fetchers=fakes,
    )
    assert report.skipped == {"sgHdbBuildings": "outside Singapore"}


def testPrepare_fallsBackToGlo30WhenFabdemYieldsNothing(tmp_path: Path, fakes):
    fakes["fabdem"] = lambda ctx: []
    report = pipeline.preparePack(
        SG,
        tmp_path / "p.pack",
        sources=["fabdem", "egm2008"],
        cacheDir=tmp_path / "c",
        fetchers=fakes,
    )
    assert report.ingested["copernicusGlo30"] == 1
    assert "fabdem" in report.skipped


def testPrepare_missingRequiredRoleRaises(tmp_path: Path, fakes):
    with pytest.raises(RuntimeError, match="geoid"):
        pipeline.preparePack(
            SG, tmp_path / "p.pack", sources=["fabdem"], cacheDir=tmp_path / "c", fetchers=fakes
        )


def testPrepare_nothingIngestedRaises(tmp_path: Path, fakes):
    fakes["copernicusGlo30"] = lambda ctx: []
    with pytest.raises(RuntimeError, match="ground"):
        pipeline.preparePack(
            SG, tmp_path / "p.pack", sources=["metaCanopy"], cacheDir=tmp_path / "c", fetchers=fakes
        )


def testPrepare_unknownSourceRejected(tmp_path: Path, fakes):
    with pytest.raises(ValueError, match="no fetcher"):
        pipeline.preparePack(SG, tmp_path / "p.pack", sources=["nope"], fetchers=fakes)


def testDefaultSources_haveFetchersAndRegistryEntries():
    from map3d.pack.registry import SOURCES

    for sourceId in pipeline.DEFAULT_SOURCES:
        assert sourceId in pipeline.FETCHERS
        assert sourceId in SOURCES
    assert "copernicusGlo30" in pipeline.FETCHERS


def testContext_lazyS3AndRelease(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from map3d.prepare.sources import overture

    calls = {"fs": 0}

    def fakeFs(ca):
        calls["fs"] += 1
        return "FS"

    monkeypatch.setattr(overture, "s3FileSystem", fakeFs)
    monkeypatch.setattr(overture, "latestRelease", lambda fs: "2026-11-18.0")
    ctx = pipeline.PrepareContext(SG, tmp_path, session=None, year=2023)  # type: ignore[arg-type]
    assert ctx.releaseBase() == "overturemaps-us-west-2/release/2026-11-18.0"
    ctx.s3()
    assert calls["fs"] == 1
    assert ctx.workDir("x") == tmp_path / "work" / "x"
