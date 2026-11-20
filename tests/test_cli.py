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
