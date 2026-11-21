"""Pack manifest: coverage, sources, licences and file checksums (SPEC section 9)."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from map3d.pack.registry import DataKind, SourceSpec, VerticalReference

__all__ = [
    "ATTRIBUTION_NAME",
    "MANIFEST_NAME",
    "PACK_FORMAT_VERSION",
    "Bounds",
    "Manifest",
    "PackFile",
    "SourceEntry",
    "boundsIntersect",
    "fileSha256",
    "validateBounds",
]

MANIFEST_NAME = "manifest.json"
ATTRIBUTION_NAME = "ATTRIBUTION.md"
PACK_FORMAT_VERSION = 1

Bounds = tuple[float, float, float, float]


def fileSha256(path: str | os.PathLike[str]) -> str:
    """Hex SHA-256 of a file's contents.

    Args:
        path: File path.

    Returns:
        The hex digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validateBounds(bounds: Sequence[float], allowDegenerate: bool = False) -> Bounds:
    """Validate a ``(minLon, minLat, maxLon, maxLat)`` box in degrees.

    Args:
        bounds: The box.
        allowDegenerate: Accept zero width/height (e.g. the extent of a single point).

    Returns:
        The box as a float tuple.

    Raises:
        ValueError: If the box is malformed or out of range.
    """
    if len(bounds) != 4:
        raise ValueError(f"bounds must be (minLon, minLat, maxLon, maxLat), got {bounds!r}")
    minLon, minLat, maxLon, maxLat = (float(v) for v in bounds)
    ok = (
        all(math.isfinite(v) for v in (minLon, minLat, maxLon, maxLat))
        and -180.0 <= minLon <= maxLon <= 180.0
        and -90.0 <= minLat <= maxLat <= 90.0
        and (allowDegenerate or (minLon < maxLon and minLat < maxLat))
    )
    if not ok:
        raise ValueError(f"invalid bounds (minLon, minLat, maxLon, maxLat): {tuple(bounds)!r}")
    return (minLon, minLat, maxLon, maxLat)


def boundsIntersect(a: Bounds, b: Bounds) -> bool:
    """Whether two lon/lat boxes overlap (touching edges count).

    Args:
        a: ``(minLon, minLat, maxLon, maxLat)``.
        b: Another box.

    Returns:
        ``True`` if the closed boxes share at least one point.
    """
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


@dataclass
class PackFile:
    """One data file in a pack.

    Attributes:
        path: POSIX path relative to the pack root.
        sha256: Hex SHA-256 of the file contents.
        bytes: File size.
        bounds_deg: File extent in lon/lat, or ``None`` if empty.
    """

    path: str
    sha256: str
    bytes: int
    bounds_deg: Bounds | None


@dataclass
class SourceEntry:
    """A source present in a pack, with metadata frozen at ingest time.

    Attributes:
        sourceId: Source identifier.
        role: Role it fills.
        kind: ``"raster"`` or ``"vector"``.
        dataset: Dataset name and version.
        licence: Licence name.
        attribution: Attribution text.
        nativeResolution_m: Nominal resolution (0 for vectors).
        verticalReference: Vertical reference of heights, if any.
        crs: Raster CRS as WKT (``None`` for vectors, which are lon/lat).
        files: Files holding the data.
        accessedDate: ISO date the data was obtained (for attribution notices).
    """

    sourceId: str
    role: str
    kind: DataKind
    dataset: str
    licence: str
    attribution: str
    nativeResolution_m: float
    verticalReference: VerticalReference | None
    crs: str | None
    files: list[PackFile] = field(default_factory=list)
    accessedDate: str | None = None

    @classmethod
    def fromSpec(cls, spec: SourceSpec) -> SourceEntry:
        """Create an empty entry (no CRS, no files) from a registry spec.

        Args:
            spec: The registry source spec.

        Returns:
            The entry, with no files.
        """
        return cls(
            sourceId=spec.sourceId,
            role=spec.role,
            kind=spec.kind,
            dataset=spec.dataset,
            licence=spec.licence,
            attribution=spec.attribution,
            nativeResolution_m=spec.nativeResolution_m,
            verticalReference=spec.verticalReference,
            crs=None,
        )


@dataclass
class Manifest:
    """Pack manifest.

    Attributes:
        bounds_deg: Declared pack coverage ``(minLon, minLat, maxLon, maxLat)``.
        createdUtc: ISO-8601 creation time.
        sources: Sources keyed by identifier.
        packFormatVersion: Pack format version.
    """

    bounds_deg: Bounds
    createdUtc: str
    sources: dict[str, SourceEntry] = field(default_factory=dict)
    packFormatVersion: int = PACK_FORMAT_VERSION

    @classmethod
    def new(cls, bounds_deg: Sequence[float]) -> Manifest:
        """Create an empty manifest.

        Args:
            bounds_deg: Pack coverage in lon/lat.

        Returns:
            The manifest.
        """
        return cls(validateBounds(bounds_deg), datetime.now(UTC).isoformat(timespec="seconds"))

    @classmethod
    def load(cls, packDir: str | os.PathLike[str]) -> Manifest:
        """Load the manifest of a pack.

        Args:
            packDir: Pack root directory.

        Returns:
            The manifest.

        Raises:
            ValueError: If the directory is not a pack or the format is unsupported.
        """
        path = Path(packDir) / MANIFEST_NAME
        if not path.is_file():
            raise ValueError(f"{packDir} is not a map3d pack (no {MANIFEST_NAME})")
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        version = data.get("packFormatVersion")
        if version != PACK_FORMAT_VERSION:
            raise ValueError(
                f"unsupported pack format version {version} (expected {PACK_FORMAT_VERSION})"
            )
        sources = {
            sourceId: SourceEntry(
                **{k: v for k, v in entry.items() if k != "files"},
                files=[
                    PackFile(
                        f["path"],
                        f["sha256"],
                        f["bytes"],
                        None
                        if f["bounds_deg"] is None
                        else validateBounds(f["bounds_deg"], allowDegenerate=True),
                    )
                    for f in entry["files"]
                ],
            )
            for sourceId, entry in data["sources"].items()
        }
        return cls(validateBounds(data["bounds_deg"]), data["createdUtc"], sources, version)

    def save(self, packDir: str | os.PathLike[str]) -> None:
        """Write ``manifest.json`` (atomically) and ``ATTRIBUTION.md``.

        Args:
            packDir: Pack root directory (must exist).
        """
        root = Path(packDir)
        tmp = root / (MANIFEST_NAME + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        os.replace(tmp, root / MANIFEST_NAME)
        (root / ATTRIBUTION_NAME).write_text(self.attributionMarkdown(), encoding="utf-8")

    def attributionMarkdown(self) -> str:
        """Attribution and licence summary for every source in the pack.

        Returns:
            Markdown text.
        """
        lines = ["# Data attribution", ""]
        for entry in sorted(self.sources.values(), key=lambda e: (e.role, e.sourceId)):
            lines.append(f"- **{entry.dataset}** ({entry.role}; licence: {entry.licence}).")
            accessed = entry.accessedDate or "(unknown date)"
            lines.append(f"  {entry.attribution.replace('{accessed}', accessed)}")
        return "\n".join(lines) + "\n"
