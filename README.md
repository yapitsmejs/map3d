# map3d

Build offline, georeferenced 3D scene grids and 3D Tiles meshes from open data.

## Purpose

Given a scene center (lat/lon), grid spacing, grid orientation (a 3×3 matrix of
grid axes expressed in ENU) and swath size, `map3d` produces:

- a **multi-layer grid** for simulation and analysis: ground and top-surface
  heights (both height-above-datum and curvature-aware ENU z), building and
  canopy heights, up to K elevated-deck levels, land class, feature IDs and
  per-node provenance. It is written as `.npz` + JSON and as GeoTIFF.
- a **3D Tiles mesh** for visualization, with imagery-draped terrain,
  buildings and true 3D bridges, plus a bundled offline viewer.

It runs fully offline from portable **data packs** built from open data
(Copernicus/FABDEM, Overture, Google Open Buildings 2.5D, Meta/WRI canopy,
ESA WorldCover, OSM, data.gov.sg). Packs are prepared on an
internet-connected machine and copied over.

- Specification: [`docs/SPEC.md`](docs/SPEC.md)
- Roadmap: [`plans/developmentPlan.md`](plans/developmentPlan.md)

> **Status:** M3 of 8. `map3d build` produces terrain + buildings grids
> (`.npz` + JSON, GeoTIFF) from offline packs; `map3d ingest` / `map3d info`
> manage packs. Canopy, land cover, decks (M5), `prepare` (M4) and 3D Tiles
> (M7) are still to come. Try it on a synthetic pack:
>
> ```bash
> uv run python -c "from map3d.pack import makeSyntheticPack; makeSyntheticPack('demo.pack', terrain='hill')"
> uv run map3d build --lat 1.3 --lon 103.8 --spacing 1 --swath 1000 1000 --pack demo.pack --out out --format npz,geotiff
> ```

## Install

This project is managed with [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev          # create .venv and install runtime + dev dependencies
uv sync --extra prepare      # (online machine only) extras for `map3d prepare`
```

## Usage

Target interface (see the spec, §10); `write3dTiles` and `prepare` are not
implemented yet:

```python
from map3d import SceneRequest, buildScene

request = SceneRequest(
    sceneCenter_deg=(1.2834, 103.8607),
    gridSpacing_m=0.5,
    gridAxes_ENU=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # rows: grid x, y, z in ENU
    swathHorizontal_m=5000.0,
    swathVertical_m=5000.0,
)
scene = buildScene(request, pack="sg.pack")
scene.writeNpz("out/scene")
scene.write3dTiles("out/tiles")
```

```bash
map3d prepare --bbox 103.5 1.1 104.2 1.5 --out sg.pack     # online machine
map3d build --lat 1.2834 --lon 103.8607 --spacing 0.5 --axes axes.json \
            --swath 5000 5000 --pack sg.pack --out out/ --format npz,geotiff,3dtiles
map3d view out/tiles
```

## Development

```bash
uv run pytest                 # run tests (dev loop, no coverage gate)
uv run ruff check .           # lint
uv run ruff format .          # format
uv run mypy src/map3d         # type check
uv run python scripts/check.py   # GATE: lint + format + types + tests + branch coverage ≥ 90%
```

Run `scripts/check.py` before pushing or opening a PR. It fails if ruff is
unhappy, formatting would change, **mypy** finds a type error, or branch coverage
on the package drops below 90%. See [`CONVENTIONS.md`](CONVENTIONS.md) for the
full naming, code-style, and test-driven conventions.

### Automation

These run the same gate as you do locally, so a red PR cannot merge:

- **CI** (`.github/workflows/ci.yml`) runs `scripts/check.py` on every push and PR.
- **Git hooks** (`.githooks/`) are opt-in per clone: fast `ruff` checks on
  commit, the full gate on push. One-time setup:
  ```bash
  git config core.hooksPath .githooks
  git update-index --chmod=+x .githooks/pre-commit .githooks/pre-push
  ```
  Use `git commit --no-verify` during active red-phase work; the pre-push hook
  and CI still enforce before anything lands.
- **Dependabot** (`.github/dependabot.yml`) opens weekly dependency PRs (uv +
  GitHub Actions) that must pass the gate like any other.
- **PR template** (`.github/PULL_REQUEST_TEMPLATE.md`) prompts contributors for
  the red-green workflow and the gate checklist.
