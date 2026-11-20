"""``map3d`` command-line entry point.

Subcommands (``build``, ``prepare``, ``ingest``, ``view``, ``info``) are added
in later milestones; see ``plans/developmentPlan.md``.
"""

from __future__ import annotations

from typing import Annotated

import typer

from map3d import __version__

__all__ = ["app", "main"]

app = typer.Typer(
    help="Build offline, georeferenced 3D scene grids and 3D Tiles meshes from open data.",
    no_args_is_help=True,
    add_completion=False,
)


def _printVersion(value: bool) -> None:
    if value:
        typer.echo(f"map3d {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_printVersion,
            is_eager=True,
            help="Show the map3d version and exit.",
        ),
    ] = False,
) -> None:
    """Build offline, georeferenced 3D scene grids and 3D Tiles meshes from open data."""


def main() -> None:
    """Run the ``map3d`` CLI using ``sys.argv``."""
    app()
