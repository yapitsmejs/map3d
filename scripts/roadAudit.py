"""Road-layer audit: where the rendered scene contradicts the layer truth.

Cross-checks the Overture/OSM road layers (`isBridge`, `level`) against what
the scene actually renders, and reports:

- the layer-count histogram over road nodes (how many stacked levels cover
  each node, from the transport vectors);
- **provably wrong** raised nodes: only level 0 present, yet the surface is
  more than 3 m above the ground (attributed to the winner: canopy,
  building, deck);
- elevated pieces swallowed by the surface or rendered correctly;
- at-grade roads hidden under decks (correct 2.5D semantics);
- missing decks: an elevated piece is present but nothing is rendered;
- same-direction overlaps: at-grade roads running directly beneath a
  parallel elevated piece;
- suspect bridges: tagged `isBridge` in the pack but carrying no deck.

Usage:
    uv run python scripts/roadAudit.py --pack sg.pack --lat 1.3626 --lon 103.9643 \\
        --spacing 0.5 --swath 4500 [--out audit.txt]
    uv run python scripts/roadAudit.py --pack sg.pack --scene out/scene.npz [--out audit.txt]

Without ``--scene`` the scene is built in memory from the request (no
scene.npz is written). With ``--scene`` the request is read from its
``scene.json`` sidecar and the existing layers are audited.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import shapely

from map3d.gridGeometry import GridGeometry
from map3d.layers.provenance import HeightSource
from map3d.pack.reader import Pack
from map3d.pack.registry import sourcesForRole
from map3d.request import SceneRequest
from map3d.scene import buildScene

_STEP_m = 5.0  # census sampling step along segments


@dataclass(frozen=True)
class Segment:
    """One road segment in ENU metres, with its layer-truth flags."""

    featureId: str
    enu: Any
    foot: Any
    length_m: float
    azimuth_deg: float
    elevated: bool
    level: int


_RAISED_m = 3.0  # surface this far above ground counts as raised
_PARALLEL_DEG = 20.0  # azimuth difference that counts as "same direction"

_SRC_NAMES = {
    int(HeightSource.NONE): "none",
    int(HeightSource.GROUND): "ground",
    int(HeightSource.CANOPY): "canopy",
    int(HeightSource.DECK_ESTIMATED): "deck",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=None, help="built scene.npz to audit")
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--spacing", type=float, default=None)
    parser.add_argument("--swath", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None, help="also write the report here")
    args = parser.parse_args()

    layers = None
    if args.scene is not None:
        request, layers = _loadScene(args.scene)
    else:
        missing = [
            name
            for name, value in (
                ("--lat", args.lat),
                ("--lon", args.lon),
                ("--spacing", args.spacing),
                ("--swath", args.swath),
            )
            if value is None
        ]
        if missing:
            parser.error(f"without --scene, all of {', '.join(missing)} are required")
        request = SceneRequest(
            (args.lat, args.lon),
            args.spacing,
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            args.swath,
            args.swath,
        )
    geometry = GridGeometry.fromRequest(request)

    pack = Pack.open(args.pack)
    if layers is None:
        scene = buildScene(request, pack)
        layers = {name: np.asarray(scene.layers[name]) for name in scene.layers}
        for name, values in scene.features.items():
            layers[f"features.{name}"] = np.asarray(values)
        scene.close()

    report = _audit(pack, geometry, layers)
    text = "\n".join(report)
    print(text)
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")


def _loadScene(scenePath: Path) -> tuple[SceneRequest, dict[str, np.ndarray] | None]:
    meta = json.loads((scenePath.parent / "scene.json").read_text(encoding="utf-8"))
    m = meta["request"]
    request = SceneRequest(
        (m["sceneCenter_deg"][0], m["sceneCenter_deg"][1]),
        m["gridSpacing_m"],
        m["gridAxes_ENU"],
        m["swathHorizontal_m"],
        m["swathVertical_m"],
        maxDeckLevels=0 if meta["layers"]["deckTopHeight_datum_m"]["shape"][0] == 0 else 5,
    )
    z = np.load(scenePath)
    return request, {name: np.asarray(z[name]) for name in z.files}


def _audit(pack: Pack, geometry: GridGeometry, layers: dict[str, np.ndarray]) -> list[str]:
    s = geometry.gridSpacing_m
    halfX, halfY = 0.5 * geometry.nCols * s, 0.5 * geometry.nRows * s

    surf = layers["surfaceHeight_datum_m"]
    groundH = layers["groundHeight_datum_m"]
    hsrc = layers["heightSource"]
    fSrc = layers["features.sourceDataset"]

    segments = _roadSegments(pack, geometry)
    out = [f"pack: {pack.root}", f"road segments in bounds: {len(segments)}"]

    tree = shapely.STRtree(np.array([seg.foot for seg in segments], dtype=object))
    layerHist: Counter = Counter()
    wrongRaised: Counter = Counter()
    bridgeElevated: Counter = Counter()
    bridgeAtGround: Counter = Counter()
    hiddenUnderDeck: Counter = Counter()
    missingDeck: Counter = Counter()
    sameDir: Counter = Counter()

    for seg in segments:
        n = max(2, int(seg.length_m / _STEP_m))
        for t in range(n + 1):
            p = shapely.line_interpolate_point(seg.enu, t / n, normalized=True)
            r = int(round((p.y + halfY) / s - 0.5))
            c = int(round((p.x + halfX) / s - 0.5))
            if not (0 <= r < surf.shape[0] and 0 <= c < surf.shape[1]):
                continue
            stack, parallel = _stackAt(tree, segments, p, seg.azimuth_deg)
            if not stack:
                continue
            layerHist[len(stack)] += 1
            dh = float(surf[r, c] - groundH[r, c])
            hs = int(hsrc[r, c])
            if parallel is not None and not seg.elevated:
                sameDir[(seg.featureId, parallel)] += seg.length_m / n
            if seg.elevated:
                (bridgeElevated if dh > _RAISED_m else bridgeAtGround)[max(stack)] += 1
                continue
            if stack == {0}:
                if dh > _RAISED_m:
                    fid = int(layers["featureId"][r, c])
                    winner = f"{_SRC_NAMES.get(hs, hs)}/{str(fSrc[fid - 1]) if fid > 0 else '-'}"
                    wrongRaised[winner] += 1
            elif dh > _RAISED_m:
                hiddenUnderDeck[max(stack)] += 1
            else:
                missingDeck[(max(stack), _SRC_NAMES.get(hs, hs))] += 1

    total = sum(layerHist.values())
    out.append(f"road nodes sampled: {total}")
    out.append("layers covering each node:")
    for nL in sorted(layerHist):
        out.append(f"  {nL}: {layerHist[nL]:7d}  ({layerHist[nL] / total:5.1%})")

    out.append("")
    out.append(f"PROVABLY WRONG: only level 0, surface > {_RAISED_m} m above ground:")
    for k, v in wrongRaised.most_common():
        out.append(f"  {k:34} {v}")
    out.append("ELEVATED PIECES rendered raised / at ground, by level:")
    for k, v in sorted(bridgeElevated.items()):
        out.append(f"  raised   level {k}: {v}")
    for k, v in sorted(bridgeAtGround.items()):
        out.append(f"  at-ground level {k}: {v}  (tapered ends meeting grade, or swallowed)")
    out.append("AT-GRADE ROAD HIDDEN UNDER A DECK, by top level:")
    for k, v in sorted(hiddenUnderDeck.items()):
        out.append(f"  level {k}: {v}")
    out.append("AT-GRADE VISIBLE, ELEVATED PIECE PRESENT BUT NOT RENDERED:")
    for k, v in sorted(missingDeck.items()):
        out.append(f"  {k}: {v}")
    out.append("SAME-DIRECTION OVERLAPS (at-grade under a parallel elevated piece), metres:")
    for (below, above), metres in sameDir.most_common(10):
        out.append(f"  {metres:6.0f} m: {below}  under  {above}")
    return out


def _roadSegments(pack: Pack, geometry: GridGeometry) -> list[Any]:
    bounds = geometry.footprintBounds_deg(margin_m=geometry.gridSpacing_m)
    frame = geometry.frame
    segments = []
    for sourceId in sourcesForRole("transport"):
        if sourceId not in pack.manifest.sources:
            continue
        sel = pack.readVectors(
            sourceId, bounds, columns=["sourceFeatureId", "kind", "isBridge", "level", "width_m"]
        )
        t = sel.table
        for i in range(sel.geometries.size):
            kind = str(t.column("kind")[i].as_py())
            if kind != "road":
                continue
            level = t.column("level")[i].as_py() or 0
            isBridge = bool(t.column("isBridge")[i].as_py())
            coords = shapely.get_coordinates(sel.geometries[i])
            e, n, _ = frame.geodeticToEnu(coords[:, 1], coords[:, 0], np.zeros(coords.shape[0]))
            enu = shapely.set_coordinates(
                shapely.from_wkb(shapely.to_wkb(sel.geometries[i])), np.column_stack([e, n])
            )
            width = t.column("width_m")[i].as_py()
            width = 8.0 if width is None else float(width)
            segments.append(
                Segment(
                    featureId=str(t.column("sourceFeatureId")[i].as_py()),
                    enu=enu,
                    foot=shapely.buffer(enu, width / 2, cap_style="flat"),
                    length_m=shapely.length(enu),
                    azimuth_deg=math.degrees(
                        math.atan2(coords[-1, 0] - coords[0, 0], coords[-1, 1] - coords[0, 1])
                    ),
                    elevated=isBridge or int(level) >= 1,
                    level=int(level),
                )
            )
    return segments


def _stackAt(
    tree: shapely.STRtree, segments: list[Any], point: Any, ownAzimuth_deg: float
) -> tuple[set[int], str | None]:
    stack: set[int] = set()
    parallel: str | None = None
    for h in np.asarray(tree.query(point, predicate="intersects")):
        seg = segments[h]
        stack.add(seg.level if seg.elevated else 0)
        if seg.elevated and abs(seg.azimuth_deg - ownAzimuth_deg) < _PARALLEL_DEG:
            parallel = seg.featureId
    return stack, parallel


if __name__ == "__main__":
    main()
