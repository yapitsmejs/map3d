"""``map3d`` command-line entry point.

Commands: ``prepare`` (online) -> ``build`` (npz, geotiff, 3dtiles) -> ``view``;
``ingest`` / ``info`` manage packs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, NoReturn

import numpy as np
import typer

from map3d import __version__
from map3d.layers.terrain import DEFAULT_GROUND_SMOOTHING_m
from map3d.pack.ingest import ingestFiles
from map3d.pack.reader import Pack
from map3d.request import SceneRequest
from map3d.scene import buildScene

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


_FORMATS = ("npz", "geotiff", "3dtiles")


def _parseAxes(text: str | None) -> list[list[float]]:
    if text is None:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    path = Path(text)
    raw = path.read_text(encoding="utf-8") if path.is_file() else text
    try:
        axes: list[list[float]] = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--axes is neither a JSON file nor inline JSON: {exc}") from None
    return axes


@app.command()
def build(
    lat: Annotated[float, typer.Option("--lat", help="Scene center latitude (deg).")],
    lon: Annotated[float, typer.Option("--lon", help="Scene center longitude (deg).")],
    spacing: Annotated[float, typer.Option("--spacing", help="Grid spacing (m).")],
    swath: Annotated[
        tuple[float, float],
        typer.Option(
            "--swath", metavar="HORIZONTAL VERTICAL", help="Swath along grid x and y (m)."
        ),
    ],
    pack: Annotated[Path, typer.Option("--pack", help="Offline data pack.")],
    out: Annotated[Path, typer.Option("--out", help="Output directory.")],
    axes: Annotated[
        str | None,
        typer.Option(
            "--axes",
            help="3x3 grid axes in ENU, rows = x, y, z (JSON file or inline JSON). Default: identity.",
        ),
    ] = None,
    datum: Annotated[str, typer.Option("--datum", help="msl or ellipsoid.")] = "msl",
    decks: Annotated[int, typer.Option("--decks", help="Max elevated-deck levels.")] = 5,
    formats: Annotated[
        str, typer.Option("--format", help="Comma list: npz, geotiff, 3dtiles.")
    ] = "npz",
    imagery: Annotated[Path | None, typer.Option("--imagery", help="BYO orthophoto.")] = None,
    groundSmoothing: Annotated[
        float,
        typer.Option(
            "--ground-smoothing",
            help="Ground low-pass width (m, FWHM): removes terrain bumps narrower than this. 0 = off.",
        ),
    ] = DEFAULT_GROUND_SMOOTHING_m,
) -> None:
    """Build a scene (grid layers and/or 3D Tiles) from an offline pack."""
    try:
        wanted = [f.strip() for f in formats.split(",") if f.strip()]
        for fmt in wanted:
            if fmt not in _FORMATS:
                raise ValueError(f"unknown format {fmt!r}; choose from {', '.join(_FORMATS)}")
        request = SceneRequest(
            sceneCenter_deg=(lat, lon),
            gridSpacing_m=spacing,
            gridAxes_ENU=np.asarray(_parseAxes(axes), dtype=np.float64),
            swathHorizontal_m=swath[0],
            swathVertical_m=swath[1],
            verticalDatum=datum,  # type: ignore[arg-type]
            maxDeckLevels=decks,
            imageryPath=imagery,
        )
        scene = buildScene(request, pack, groundSmoothing_m=groundSmoothing)
    except ValueError as exc:
        _fail(exc)
    for message in scene.warnings:
        typer.echo(f"warning: {message}", err=True)
    typer.echo(f"Built {scene.geometry.nRows} x {scene.geometry.nCols} nodes (rows x cols)")
    if "npz" in wanted:
        npzPath, jsonPath = scene.writeNpz(out / "scene")
        typer.echo(f"  wrote {npzPath} and {jsonPath.name}")
    if "geotiff" in wanted:
        paths = scene.writeGeoTiff(out / "geotiff")
        typer.echo(f"  wrote {len(paths)} GeoTIFFs to {out / 'geotiff'}")
    if "3dtiles" in wanted:
        tileset = scene.write3dTiles(out / "tiles")
        typer.echo(f"  wrote {tileset} (view with: map3d view {out / 'tiles'})")
    scene.close()


def _waitForever() -> None:  # pragma: no cover - interactive; replaced in tests
    import time

    while True:
        time.sleep(3600)


@app.command()
def view(
    tiles: Annotated[Path, typer.Argument(help="Directory holding tileset.json.")],
    port: Annotated[int, typer.Option("--port", help="Localhost port (0 = any free port).")] = 8000,
    noBrowser: Annotated[bool, typer.Option("--no-browser", help="Don't open a browser.")] = False,
) -> None:
    """Serve a tileset with the offline viewer on localhost (Ctrl+C to stop)."""
    from map3d.viewer import serve

    try:
        server = serve(tiles, port=port, openBrowser=not noBrowser)
    except (FileNotFoundError, OSError) as exc:
        _fail(exc)
    typer.echo(f"Serving {tiles} at http://127.0.0.1:{server.server_address[1]}/ (Ctrl+C to stop)")
    try:
        _waitForever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


@app.command()
def prepare(
    bbox: Annotated[
        tuple[float, float, float, float],
        typer.Option("--bbox", metavar="MINLON MINLAT MAXLON MAXLAT", help="Pack coverage."),
    ],
    out: Annotated[Path, typer.Option("--out", help="Pack directory to create or update.")],
    sources: Annotated[
        str | None, typer.Option("--sources", help="Comma list of source ids (default: all).")
    ] = None,
    cache: Annotated[Path | None, typer.Option("--cache", help="Download cache directory.")] = None,
    caBundle: Annotated[
        Path | None, typer.Option("--ca-bundle", help="PEM CA bundle (TLS-intercepting networks).")
    ] = None,
    systemCerts: Annotated[
        bool, typer.Option("--system-certs", help="Trust the operating system's CA store.")
    ] = False,
    year: Annotated[
        int, typer.Option("--year", help="Year for Google Open Buildings 2.5D.")
    ] = 2023,
) -> None:
    """Download open data for a box and build an offline pack (needs internet)."""
    try:
        from map3d.prepare import certs, pipeline
    except ImportError as exc:
        _fail(
            RuntimeError(
                f"`map3d prepare` needs the 'prepare' extra (pip install 'map3d[prepare]'): {exc}"
            )
        )
    if caBundle is not None and systemCerts:
        _fail(ValueError("use either --ca-bundle or --system-certs, not both"))
    cacheDir = cache if cache is not None else out.with_name(out.name + ".cache")
    bundle = certs.exportCaBundle(cacheDir / "ca-bundle.pem") if systemCerts else caBundle
    wanted = [s.strip() for s in sources.split(",") if s.strip()] if sources else None
    try:
        report = pipeline.preparePack(
            bbox,
            out,
            sources=wanted,
            cacheDir=cacheDir,
            caBundle=bundle,
            year=year,
            log=lambda message: typer.echo(message, err=True),
        )
    except (ValueError, RuntimeError) as exc:
        _fail(exc)
    for sourceId, count in report.ingested.items():
        typer.echo(f"ingested  {sourceId} ({count} file(s))")
    for sourceId, reason in report.skipped.items():
        typer.echo(f"skipped   {sourceId}: {reason}")
    for sourceId, reason in report.failed.items():
        typer.echo(f"FAILED    {sourceId}: {reason}")


def main() -> None:
    """Run the ``map3d`` CLI using ``sys.argv``."""
    app()
