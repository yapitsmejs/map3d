"""``map3d prepare``: fetch every source for a box and build an offline pack.

Each source's fetcher writes pack-ready files into a cache/work directory,
then they go through :func:`map3d.pack.ingest.ingestFiles`, the same
validation as a manual ingest, with today's access date.

- A failing source is recorded and the run continues.
- Singapore sources are skipped for boxes outside Singapore.
- If FABDEM yields nothing, Copernicus GLO-30 is fetched as the ground
  fallback.
- The run raises if the pack ends up without a required role (ground, geoid).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from map3d.pack.ingest import ingestFiles
from map3d.pack.manifest import MANIFEST_NAME, Bounds, Manifest, validateBounds
from map3d.pack.registry import ROLES, getSource
from map3d.prepare.http import HttpSession
from map3d.prepare.sources import dataGovSg, googleBuildings, imagery, overture, rasters

__all__ = ["DEFAULT_SOURCES", "FETCHERS", "PrepareContext", "PrepareReport", "preparePack"]


@dataclass
class PrepareContext:
    """What a fetcher gets: the box, a work area, the HTTP session and lazy S3 access."""

    bounds: Bounds
    cacheDir: Path
    session: HttpSession
    year: int = 2023
    _s3: Any = field(default=None, repr=False)
    _release: str | None = field(default=None, repr=False)

    def workDir(self, name: str) -> Path:
        """Per-source working directory inside the cache."""
        path = self.cacheDir / "work" / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def s3(self) -> Any:
        """Anonymous S3 filesystem (created on first use, sharing the CA bundle)."""
        if self._s3 is None:
            self._s3 = overture.s3FileSystem(self.session.caBundle)
        return self._s3

    def releaseBase(self) -> str:
        """``<bucket>/release/<latest>`` for Overture."""
        if self._release is None:
            self._release = overture.latestRelease(self.s3())
        return f"{overture.BUCKET}/release/{self._release}"


Fetcher = Callable[[PrepareContext], list[Path]]


def _sgDownload(ctx: PrepareContext, key: str) -> Path:
    datasetId, isCsv = dataGovSg.DATASETS[key]
    target = ctx.cacheDir / "datagovsg" / f"{key}.{'csv' if isCsv else 'geojson'}"
    return dataGovSg.downloadDataset(ctx.session, datasetId, target, isCsv)


def _sgHdb(ctx: PrepareContext) -> list[Path]:
    return [
        dataGovSg.adaptHdbBuildings(
            _sgDownload(ctx, "hdbBuildings"),
            _sgDownload(ctx, "hdbPropertyInfo"),
            ctx.bounds,
            ctx.workDir("sgHdbBuildings") / "sgHdbBuildings.parquet",
        )
    ]


FETCHERS: dict[str, Fetcher] = {
    "fabdem": lambda c: rasters.fetchFabdem(c.session, c.bounds, c.workDir("fabdem")),
    "copernicusGlo30": lambda c: rasters.fetchGlo30(
        c.session, c.bounds, c.workDir("copernicusGlo30")
    ),
    "egm2008": lambda c: rasters.fetchEgm2008(c.session, c.bounds, c.workDir("egm2008")),
    "esaWorldCover": lambda c: rasters.fetchWorldCover(
        c.session, c.bounds, c.workDir("esaWorldCover")
    ),
    "metaCanopy": lambda c: rasters.fetchMetaCanopy(c.session, c.bounds, c.workDir("metaCanopy")),
    "googleOpenBuildings25d": lambda c: googleBuildings.fetchGoogleBuildings(
        c.session, c.bounds, c.workDir("googleOpenBuildings25d"), c.year
    ),
    "overtureBuildings": lambda c: [
        overture.fetchOvertureBuildings(
            c.s3(),
            c.releaseBase(),
            c.bounds,
            c.workDir("overtureBuildings") / "overtureBuildings.parquet",
        )
    ],
    "overtureTransportation": lambda c: [
        overture.fetchOvertureTransport(
            c.s3(),
            c.releaseBase(),
            c.bounds,
            c.workDir("overtureTransportation") / "overtureTransportation.parquet",
        )
    ],
    "sentinel2Composite": lambda c: imagery.fetchSentinel2Cloudless(
        c.session, c.bounds, c.workDir("sentinel2Composite")
    ),
    "sgHdbBuildings": _sgHdb,
    "sgUraBuildings": lambda c: [
        dataGovSg.adaptUraBuildings(
            _sgDownload(c, "uraBuildings"),
            c.bounds,
            c.workDir("sgUraBuildings") / "sgUraBuildings.parquet",
        )
    ],
    "sgMasterPlanRail": lambda c: [
        dataGovSg.adaptMasterPlanRail(
            _sgDownload(c, "mpRail"),
            c.bounds,
            c.workDir("sgMasterPlanRail") / "sgMasterPlanRail.parquet",
        )
    ],
    "sgLtaElevatedStructures": lambda c: [
        dataGovSg.adaptLtaStructures(
            _sgDownload(c, "ltaStructures"),
            c.bounds,
            c.workDir("sgLtaElevatedStructures") / "sgLtaElevatedStructures.parquet",
        )
    ],
}

DEFAULT_SOURCES: tuple[str, ...] = (
    "fabdem",
    "egm2008",
    "overtureBuildings",
    "overtureTransportation",
    "googleOpenBuildings25d",
    "metaCanopy",
    "esaWorldCover",
    "sentinel2Composite",
    "sgHdbBuildings",
    "sgUraBuildings",
    "sgMasterPlanRail",
    "sgLtaElevatedStructures",
)

_SINGAPORE_ONLY = {s for s in FETCHERS if s.startswith("sg")}


@dataclass
class PrepareReport:
    """Outcome per source."""

    ingested: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)


def preparePack(
    bounds_deg: Sequence[float],
    packDir: str | os.PathLike[str],
    sources: Sequence[str] | None = None,
    cacheDir: str | os.PathLike[str] | None = None,
    caBundle: str | os.PathLike[str] | None = None,
    year: int = 2023,
    session: HttpSession | None = None,
    fetchers: dict[str, Fetcher] | None = None,
    log: Callable[[str], None] = lambda message: None,
) -> PrepareReport:
    """Download and ingest every requested source for a box.

    Args:
        bounds_deg: Pack coverage ``(minLon, minLat, maxLon, maxLat)``.
        packDir: Pack directory to create or update.
        sources: Source ids (default: :data:`DEFAULT_SOURCES`).
        cacheDir: Download/work cache (default: ``<packDir>.cache`` beside the pack).
        caBundle: PEM CA bundle for TLS-intercepting networks.
        year: Year for time-series sources (Google 2.5D).
        session: HTTP session to use (default: a new throttled session).
        fetchers: Fetcher overrides (tests).
        log: Progress callback.

    Returns:
        Per-source outcome.

    Raises:
        ValueError: For unknown sources or bounds.
        RuntimeError: If the pack lacks ground or geoid afterwards.
    """
    bounds = validateBounds(bounds_deg)
    table = fetchers if fetchers is not None else FETCHERS
    wanted = list(sources or DEFAULT_SOURCES)
    for sourceId in wanted:
        if sourceId not in table:
            raise ValueError(
                f"no fetcher for source {sourceId!r}; available: {', '.join(sorted(table))}"
            )
        getSource(sourceId)
    pack = Path(packDir)
    cache = Path(cacheDir) if cacheDir is not None else pack.with_name(pack.name + ".cache")
    ownSession = session is None
    http = (
        session
        if session is not None
        else HttpSession(caBundle, minInterval_s=dataGovSg.MIN_INTERVAL_s)
    )
    ctx = PrepareContext(bounds, cache, http, year)
    report = PrepareReport()
    today = datetime.now(UTC).date().isoformat()
    try:
        for sourceId in wanted:
            _runOne(sourceId, table, ctx, pack, bounds, today, report, log)
        if (
            not _hasRole(pack, "ground")
            and "copernicusGlo30" not in wanted
            and "copernicusGlo30" in table
        ):
            log("no ground source ingested; falling back to Copernicus GLO-30")
            _runOne("copernicusGlo30", table, ctx, pack, bounds, today, report, log)
    finally:
        if ownSession:
            http.close()
    for role, spec in ROLES.items():
        if spec.required and not _hasRole(pack, role):
            raise RuntimeError(f"pack is missing required role {role!r}; report: {report}")
    return report


def _runOne(
    sourceId: str,
    table: dict[str, Fetcher],
    ctx: PrepareContext,
    pack: Path,
    bounds: Bounds,
    today: str,
    report: PrepareReport,
    log: Callable[[str], None],
) -> None:
    if sourceId in _SINGAPORE_ONLY and not dataGovSg.intersectsSingapore(bounds):
        report.skipped[sourceId] = "outside Singapore"
        return
    log(f"{sourceId}: fetching")
    try:
        files = table[sourceId](ctx)
    except Exception as exc:  # network/provider failures must not abort the whole run
        report.failed[sourceId] = f"{type(exc).__name__}: {exc}"
        log(f"{sourceId}: FAILED ({report.failed[sourceId]})")
        return
    if not files:
        report.skipped[sourceId] = "no data in bounds"
        log(f"{sourceId}: no data in bounds")
        return
    ingestFiles(pack, sourceId, files, bounds_deg=bounds, accessedDate=today)
    report.ingested[sourceId] = len(files)
    log(f"{sourceId}: ingested {len(files)} file(s)")


def _hasRole(pack: Path, role: str) -> bool:
    if not (pack / MANIFEST_NAME).is_file():
        return False
    return any(entry.role == role for entry in Manifest.load(pack).sources.values())
