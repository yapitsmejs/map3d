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
