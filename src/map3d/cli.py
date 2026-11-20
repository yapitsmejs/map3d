"""``map3d`` command-line entry point.

Implemented: ``ingest``, ``info``. Still to come (see ``plans/developmentPlan.md``):
``build``, ``prepare``, ``view``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, NoReturn

import typer

from map3d import __version__
from map3d.pack.ingest import ingestFiles
from map3d.pack.reader import Pack

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


def _fail(exc: Exception) -> NoReturn:
    typer.echo(f"Error: {exc}", err=True)
    raise typer.Exit(code=1)


@app.command()
def ingest(
    files: Annotated[list[Path], typer.Argument(help="GeoTIFF or canonical GeoParquet files.")],
    pack: Annotated[Path, typer.Option("--pack", help="Pack directory (created if missing).")],
    source: Annotated[str, typer.Option("--source", help="Registry source id, e.g. fabdem.")],
    bbox: Annotated[
        tuple[float, float, float, float] | None,
        typer.Option(
            "--bbox",
            metavar="MINLON MINLAT MAXLON MAXLAT",
            help="Pack coverage; required when creating a pack.",
        ),
    ] = None,
    accessed: Annotated[
        str | None,
        typer.Option("--accessed", metavar="YYYY-MM-DD", help="Date the data was obtained."),
    ] = None,
) -> None:
    """Validate pack-ready files and add them to a pack."""
    try:
        ingestFiles(pack, source, files, bounds_deg=bbox, accessedDate=accessed)
    except ValueError as exc:
        _fail(exc)
    typer.echo(f"Ingested {len(files)} file(s) into {pack} as {source}")


@app.command()
def info(
    pack: Annotated[Path, typer.Argument(help="Pack directory.")],
    verify: Annotated[bool, typer.Option("--verify", help="Check file checksums.")] = False,
) -> None:
    """Show a pack's coverage, sources and licences."""
    try:
        opened = Pack.open(pack)
    except ValueError as exc:
        _fail(exc)
    manifest = opened.manifest
    typer.echo(f"Pack: {pack}")
    typer.echo(f"Bounds (minLon minLat maxLon maxLat): {manifest.bounds_deg}")
    typer.echo(f"Created: {manifest.createdUtc}  format v{manifest.packFormatVersion}")
    for entry in sorted(manifest.sources.values(), key=lambda e: (e.role, e.sourceId)):
        typer.echo(
            f"  {entry.role:<22} {entry.sourceId:<28} {len(entry.files):>3} file(s)  "
            f"[{entry.licence}]"
        )
    if verify:
        problems = opened.verify()
        if problems:
            for problem in problems:
                typer.echo(f"  {problem}", err=True)
            raise typer.Exit(code=1)
        nFiles = sum(len(e.files) for e in manifest.sources.values())
        typer.echo(f"OK: all {nFiles} files verified")


def main() -> None:
    """Run the ``map3d`` CLI using ``sys.argv``."""
    app()
