"""Tests for the ``map3d`` command-line entry point."""

from __future__ import annotations

import sys

import pytest
from typer.testing import CliRunner

import map3d
from map3d import cli

runner = CliRunner()


def testCliVersion_printsPackageVersion():
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"map3d {map3d.__version__}"


def testCliNoArgs_showsHelp():
    result = runner.invoke(cli.app, [])
    assert "Usage" in result.output


def testCliHelp_describesTool():
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    assert "3D" in result.output


def testCliMain_runsAppFromArgv(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["map3d", "--version"])
    with pytest.raises(SystemExit) as excInfo:
        cli.main()
    assert excInfo.value.code == 0
    assert map3d.__version__ in capsys.readouterr().out


BOUNDS_ARGS = ["--bbox", "103.0", "1.0", "104.0", "2.0"]


def testCliIngest_createsPackAndReports(tmp_path, writeRaster):
    src = writeRaster(tmp_path / "dem.tif", (103.0, 1.0, 104.0, 2.0))
    pack = tmp_path / "p.pack"
    result = runner.invoke(
        cli.app, ["ingest", "--pack", str(pack), "--source", "fabdem", *BOUNDS_ARGS, str(src)]
    )
    assert result.exit_code == 0, result.output
    assert "Ingested 1 file" in result.output
    assert (pack / "manifest.json").is_file()


def testCliIngest_reportsErrorsWithExitCode1(tmp_path, writeRaster):
    src = writeRaster(tmp_path / "dem.tif", (103.0, 1.0, 104.0, 2.0))
    result = runner.invoke(
        cli.app, ["ingest", "--pack", str(tmp_path / "p"), "--source", "fabdem", str(src)]
    )
    assert result.exit_code == 1
    assert "Error: bounds are required" in result.output


def testCliInfo_listsSourcesAndVerifies(tmp_path, writeRaster):
    pack = tmp_path / "p.pack"
    src = writeRaster(tmp_path / "dem.tif", (103.0, 1.0, 104.0, 2.0))
    runner.invoke(
        cli.app, ["ingest", "--pack", str(pack), "--source", "fabdem", *BOUNDS_ARGS, str(src)]
    )
    result = runner.invoke(cli.app, ["info", str(pack), "--verify"])
    assert result.exit_code == 0, result.output
    assert "fabdem" in result.output
    assert "ground" in result.output
    assert "CC BY-NC-SA 4.0" in result.output
    assert "all 1 files verified" in result.output


def testCliInfo_verifyFailureExitsNonZero(tmp_path, writeRaster):
    pack = tmp_path / "p.pack"
    src = writeRaster(tmp_path / "dem.tif", (103.0, 1.0, 104.0, 2.0))
    runner.invoke(
        cli.app, ["ingest", "--pack", str(pack), "--source", "fabdem", *BOUNDS_ARGS, str(src)]
    )
    (pack / "raster" / "fabdem" / "dem.tif").write_bytes(b"x")
    result = runner.invoke(cli.app, ["info", str(pack), "--verify"])
    assert result.exit_code == 1
    assert "checksum mismatch" in result.output


def testCliInfo_notAPack(tmp_path):
    result = runner.invoke(cli.app, ["info", str(tmp_path)])
    assert result.exit_code == 1
    assert "not a map3d pack" in result.output


def testCliIngest_acceptsAccessedDate(tmp_path, writeRaster):
    src = writeRaster(tmp_path / "dem.tif", (103.0, 1.0, 104.0, 2.0))
    pack = tmp_path / "p.pack"
    args = [
        "ingest",
        "--pack",
        str(pack),
        "--source",
        "fabdem",
        *BOUNDS_ARGS,
        "--accessed",
        "2026-03-04",
    ]
    result = runner.invoke(cli.app, [*args, str(src)])
    assert result.exit_code == 0, result.output
    from map3d.pack.manifest import Manifest

    assert Manifest.load(pack).sources["fabdem"].accessedDate == "2026-03-04"


@pytest.fixture(scope="module")
def synPack(tmp_path_factory):
    from map3d.pack.synthetic import makeSyntheticPack

    return makeSyntheticPack(tmp_path_factory.mktemp("cli") / "syn.pack")


def _buildArgs(synPack, out, *extra):
    lat, lon = synPack.frame.lat0_deg, synPack.frame.lon0_deg
    return [
        "build",
        "--lat",
        str(lat),
        "--lon",
        str(lon),
        "--spacing",
        "10",
        "--swath",
        "400",
        "300",
        "--pack",
        str(synPack.packDir),
        "--out",
        str(out),
        *extra,
    ]


def testCliBuild_writesNpzAndGeoTiff(tmp_path, synPack):
    result = runner.invoke(
        cli.app, _buildArgs(synPack, tmp_path / "out", "--format", "npz,geotiff")
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "scene.npz").is_file()
    assert (tmp_path / "out" / "scene.json").is_file()
    assert (tmp_path / "out" / "geotiff" / "surfaceHeight_ENU_m.tif").is_file()
    assert "30 x 40" in result.output
    assert "warning:" in result.output
    lines = result.output.splitlines()
    assert len([ln for ln in lines if "imagery" in ln]) == 1  # reported once, not twice


def testCliBuild_defaultFormatIsNpzOnly(tmp_path, synPack):
    result = runner.invoke(cli.app, _buildArgs(synPack, tmp_path / "out"))
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "scene.npz").is_file()
    assert not (tmp_path / "out" / "geotiff").exists()


def testCliBuild_axesInlineJsonAndFile(tmp_path, synPack):
    axes = "[[0, 1, 0], [-1, 0, 0], [0, 0, 1]]"
    result = runner.invoke(cli.app, _buildArgs(synPack, tmp_path / "a", "--axes", axes))
    assert result.exit_code == 0, result.output
    path = tmp_path / "axes.json"
    path.write_text(axes, encoding="utf-8")
    result = runner.invoke(cli.app, _buildArgs(synPack, tmp_path / "b", "--axes", str(path)))
    assert result.exit_code == 0, result.output
    import json

    meta = json.loads((tmp_path / "b" / "scene.json").read_text(encoding="utf-8"))
    assert meta["request"]["gridAxes_ENU"] == [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]


def testCliBuild_badAxesJson(tmp_path, synPack):
    result = runner.invoke(cli.app, _buildArgs(synPack, tmp_path / "a", "--axes", "[[1, 0"))
    assert result.exit_code == 1
    assert "Error:" in result.output and "axes" in result.output


def testCliBuild_invalidRequestReported(tmp_path, synPack):
    args = _buildArgs(synPack, tmp_path / "a")
    args[args.index("--spacing") + 1] = "7"
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 1
    assert "whole multiple" in result.output


def testCliBuild_writes3dTilesWithViewer(tmp_path, synPack):
    result = runner.invoke(cli.app, _buildArgs(synPack, tmp_path / "a", "--format", "3dtiles"))
    assert result.exit_code == 0, result.output
    assert (tmp_path / "a" / "tiles" / "tileset.json").is_file()
    assert (tmp_path / "a" / "tiles" / "index.html").is_file()
    assert "map3d view" in result.output


def testCliView_servesUntilInterrupted(tmp_path, synPack, monkeypatch):
    runner.invoke(cli.app, _buildArgs(synPack, tmp_path / "a", "--format", "3dtiles"))

    def interrupt():
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_waitForever", interrupt)
    result = runner.invoke(
        cli.app, ["view", str(tmp_path / "a" / "tiles"), "--port", "0", "--no-browser"]
    )
    assert result.exit_code == 0, result.output
    assert "Serving" in result.output and "http://127.0.0.1:" in result.output


def testCliView_missingTilesetFails(tmp_path):
    result = runner.invoke(cli.app, ["view", str(tmp_path), "--port", "0", "--no-browser"])
    assert result.exit_code == 1
    assert "tileset.json" in result.output


def testCliBuild_rejectsUnknownFormats(tmp_path, synPack):
    result = runner.invoke(cli.app, _buildArgs(synPack, tmp_path / "a", "--format", "obj"))
    assert result.exit_code == 1
    assert "unknown format" in result.output


def testCliPrepare_runsPipelineAndPrintsReport(tmp_path, monkeypatch):
    from map3d.prepare import pipeline

    seen = {}

    def fake(
        bounds, packDir, sources=None, cacheDir=None, caBundle=None, year=2023, log=None, **kw
    ):
        seen.update(bounds=bounds, sources=sources, caBundle=caBundle, year=year, cacheDir=cacheDir)
        log("fabdem: fetching")
        return pipeline.PrepareReport(
            ingested={"fabdem": 2},
            skipped={"metaCanopy": "no data in bounds"},
            failed={"esaWorldCover": "ConnectError: x"},
        )

    monkeypatch.setattr(pipeline, "preparePack", fake)
    result = runner.invoke(
        cli.app,
        [
            "prepare",
            "--bbox",
            "103.8",
            "1.28",
            "103.84",
            "1.32",
            "--out",
            str(tmp_path / "p"),
            "--sources",
            "fabdem,metaCanopy",
            "--year",
            "2022",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["bounds"] == (103.8, 1.28, 103.84, 1.32)
    assert seen["sources"] == ["fabdem", "metaCanopy"]
    assert seen["year"] == 2022 and seen["caBundle"] is None
    assert "ingested  fabdem" in result.output
    assert "skipped   metaCanopy" in result.output
    assert "FAILED    esaWorldCover" in result.output


def testCliPrepare_systemCertsExportsBundleIntoCache(tmp_path, monkeypatch):
    from map3d.prepare import certs, pipeline

    exported = {}
    monkeypatch.setattr(certs, "exportCaBundle", lambda path: exported.setdefault("path", path))
    monkeypatch.setattr(
        pipeline,
        "preparePack",
        lambda *a, **kw: exported.update(ca=kw["caBundle"]) or pipeline.PrepareReport(),
    )
    result = runner.invoke(
        cli.app,
        [
            "prepare",
            "--bbox",
            "103.8",
            "1.28",
            "103.84",
            "1.32",
            "--out",
            str(tmp_path / "p"),
            "--cache",
            str(tmp_path / "c"),
            "--system-certs",
        ],
    )
    assert result.exit_code == 0, result.output
    assert exported["path"] == tmp_path / "c" / "ca-bundle.pem"
    assert exported["ca"] == exported["path"]


def testCliPrepare_errorsExitNonZero(tmp_path, monkeypatch):
    from map3d.prepare import pipeline

    def boom(*a, **kw):
        raise RuntimeError("pack is missing required role 'geoid'")

    monkeypatch.setattr(pipeline, "preparePack", boom)
    result = runner.invoke(
        cli.app,
        ["prepare", "--bbox", "103.8", "1.28", "103.84", "1.32", "--out", str(tmp_path / "p")],
    )
    assert result.exit_code == 1
    assert "geoid" in result.output


def testCliPrepare_caBundleAndSystemCertsAreExclusive(tmp_path):
    result = runner.invoke(
        cli.app,
        [
            "prepare",
            "--bbox",
            "103.8",
            "1.28",
            "103.84",
            "1.32",
            "--out",
            str(tmp_path / "p"),
            "--ca-bundle",
            "x.pem",
            "--system-certs",
        ],
    )
    assert result.exit_code == 1
    assert "either" in result.output


def testCliPrepare_missingExtraIsExplained(tmp_path, monkeypatch):
    import builtins

    realImport = builtins.__import__

    def fakeImport(name, *args, **kwargs):
        if name.startswith("map3d.prepare"):
            raise ImportError("No module named 'httpx'")
        return realImport(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fakeImport)
    result = runner.invoke(
        cli.app,
        ["prepare", "--bbox", "103.8", "1.28", "103.84", "1.32", "--out", str(tmp_path / "p")],
    )
    assert result.exit_code == 1
    assert "prepare" in result.output and "extra" in result.output


def testCliBuild_groundSmoothingFlag(tmp_path, synPack):
    import json

    result = runner.invoke(cli.app, _buildArgs(synPack, tmp_path / "a", "--ground-smoothing", "0"))
    assert result.exit_code == 0, result.output
    assert json.loads((tmp_path / "a" / "scene.json").read_text())["groundSmoothing_m"] == 0.0
    result = runner.invoke(cli.app, _buildArgs(synPack, tmp_path / "b"))
    assert json.loads((tmp_path / "b" / "scene.json").read_text())["groundSmoothing_m"] == 90.0
    result = runner.invoke(cli.app, _buildArgs(synPack, tmp_path / "c", "--ground-smoothing", "-5"))
    assert result.exit_code == 1 and "smoothing" in result.output


def testCliBuild_styleFlag(tmp_path, synPack):
    import json

    result = runner.invoke(cli.app, _buildArgs(synPack, tmp_path / "a", "--format", "3dtiles"))
    assert result.exit_code == 0, result.output
    tileset = json.loads((tmp_path / "a" / "tiles" / "tileset.json").read_text())
    assert tileset["asset"]["extras"]["style"] == "conceptual"
    result = runner.invoke(
        cli.app, _buildArgs(synPack, tmp_path / "b", "--format", "3dtiles", "--style", "satellite")
    )
    assert result.exit_code == 1 and "imagery" in result.output
