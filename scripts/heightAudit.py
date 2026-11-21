"""Building-height audit: recorded heights vs a DSM, per source.

Compares every building's resolved height against the median
DSM-minus-ground residual over its footprint (Copernicus GLO-30 when the
pack has a ``dsmHeight`` source). Buildings are large relative to the 30 m
pixels, so the residual is a meaningful independent measurement wherever
the DSM itself is good - over Singapore, Copernicus GLO-30 is known to be
poor (on the Loyang scene it misses whole HDB precincts while reading the
trees around them), so treat large "errors" as evidence about the *DSM*
until the tile is cross-checked. Reports the median absolute error per
(source dataset, height provenance), plus the worst offenders.

Usage:
    uv run python scripts/roadAudit.py-style:
    uv run python scripts/heightAudit.py --pack sg.pack --lat 1.3626 --lon 103.9643 \\
        --spacing 4 --swath 2000 [--out heights.txt]
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import shapely

from map3d.gridGeometry import GridGeometry
from map3d.layers.buildings import loadBuildings
from map3d.layers.provenance import HeightSource
from map3d.layers.terrain import GroundModel
from map3d.pack.reader import Pack
from map3d.request import SceneRequest

_MIN_AREA_m2 = 400.0  # smaller footprints are not resolvable by a 30 m DSM


def _m2PerDeg2(lat_deg: float) -> float:
    """Square metres per square degree at a latitude (for lon/lat areas)."""
    return (111_320.0 * math.cos(math.radians(lat_deg))) ** 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--spacing", type=float, default=4.0)
    parser.add_argument("--swath", type=float, default=2000.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    request = SceneRequest(
        (args.lat, args.lon),
        args.spacing,
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        args.swath,
        args.swath,
    )
    geometry = GridGeometry.fromRequest(request)
    pack = Pack.open(args.pack)
    entry = pack.primarySource("dsmHeight")
    if entry is None:
        raise SystemExit("the pack has no dsmHeight source; run prepare with copernicusDsm")
    dsm = pack.readRaster(entry.sourceId, geometry.footprintBounds_deg(margin_m=100.0))
    ground = GroundModel.load(pack, geometry)
    buildings = loadBuildings(pack, geometry, ground, correctGround=False)

    errors: list[tuple[float, str, int, str, float]] = []  # |err|, source, hsrc, id, height
    for i in range(len(buildings)):
        shape = shapely.make_valid(buildings.geometries[i])
        area_m2 = shapely.area(shape) * _m2PerDeg2(float(shapely.get_y(shapely.centroid(shape))))
        if area_m2 < _MIN_AREA_m2:
            continue
        lat, lon = _samples(shape)
        surface = np.asarray(dsm.sample(lat, lon))
        base = np.asarray(ground.orthometricAt(lat, lon))
        residual = surface - base
        residual = residual[np.isfinite(residual)]
        if residual.size < 3:
            continue
        measured = float(np.percentile(residual, 90))
        if measured <= 1.0:
            continue  # not a building the DSM can see (verandas, low blocks)
        recorded = float(buildings.height_m[i])
        errors.append(
            (
                abs(measured - recorded),
                str(buildings.sourceIds[i]),
                int(buildings.heightSource[i]),
                str(buildings.sourceFeatureIds[i]),
                recorded - measured,
            )
        )

    if errors:
        byAuth = [e for e in errors if e[2] in (2, 3)]
        if byAuth:
            medAuth = float(np.median([e[0] for e in byAuth]))
            if medAuth > 5.0:
                print(
                    "WARNING: authoritative heights deviate from the DSM by a median of "
                    f"{medAuth:.1f} m - over Singapore, suspect the DSM first (see docstring)"
                )
    lines = _report(errors)
    text = "\n".join(lines)
    print(text)
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")


def _samples(shape: object) -> tuple[np.ndarray, np.ndarray]:
    """Interior lattice samples (~15 m, half a DSM pixel), in lat/lon.

    A boundary would miss roofs on compound footprints (depots: roofs plus
    open yards), and 30 m DSM pixels at the edge mix roof and terrain; the
    high interior percentile measures the roof.
    """
    loX, loY, hiX, hiY = shapely.bounds(shape)
    step = 1.35e-4  # ~15 m in degrees
    xs = np.arange(loX, hiX + step, step)
    ys = np.arange(loY, hiY + step, step)
    gx, gy = np.meshgrid(xs, ys)
    inside = shapely.contains(shape, shapely.points(gx.ravel(), gy.ravel()))
    gx, gy = gx.ravel()[inside], gy.ravel()[inside]
    return gy, gx


def _report(errors: list[tuple[float, str, int, str, float]]) -> list[str]:
    byKey: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for absErr, source, hsrc, _, signed in errors:
        byKey[(source, hsrc)].append((absErr, signed))
    names = {int(h): h.name for h in HeightSource}
    lines = [f"buildings measured against the DSM: {len(errors)}"]
    lines.append("median |error| by (source, height provenance):")
    rows = []
    for (source, hsrc), values in byKey.items():
        absErrs = np.array([v[0] for v in values])
        rows.append((float(np.median(absErrs)), source, names.get(hsrc, hsrc), len(values)))
    for medAbs, source, hsrc, count in sorted(rows, reverse=True):
        lines.append(f"  {medAbs:6.2f} m  {source:22} {hsrc:22} n={count}")
    lines.append("worst offenders (|error|, recorded-measured, source, provenance, feature):")
    for absErr, source, hsrc, fid, signed in sorted(errors, reverse=True)[:15]:
        lines.append(
            f"  {absErr:6.2f} m  {signed:+7.2f} m  {source:22} {names.get(hsrc, hsrc):22} {fid}"
        )
    return lines


if __name__ == "__main__":
    main()
