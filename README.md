# map3d

Build offline, georeferenced 3D scene grids and 3D Tiles meshes from open data.

## Purpose

Given a scene center (lat/lon), grid spacing, grid orientation (a 3×3 matrix
whose rows are the grid x, y, z axes in ENU) and swath size, `map3d` produces:

- **A multi-layer grid** for simulation and analysis, written as `.npz` + JSON
  and as GeoTIFF:
  - ground and top-surface heights, each above a datum (MSL/EGM2008 or WGS84)
    and as a curvature-aware ENU z;
  - building and canopy heights;
  - up to K stacked elevated-deck levels;
  - land class, feature IDs and per-node provenance.
- **A 3D Tiles 1.1 mesh** for visualization: the true ground with individual
  trees (from canopy data), buildings with mapped building parts, raised parts
  and roof shapes (domes, vaults, gables, ...), and true 3D bridge decks, in a
  clean map-like style. Buildings use imagery-derived palettes when reliable, with
  clearly procedural facade windows and neutral fallback. Includes an offline viewer.

It runs fully offline from portable **data packs** of open data:

- Ground: FABDEM / Copernicus GLO-30.
- Buildings: Overture footprints, Google Open Buildings 2.5D heights, OSM.
- Vegetation and land cover: Meta/WRI canopy height, ESA WorldCover, Overture
  water and land-use polygons (crisp shorelines, parks and plazas).
- Imagery: Sentinel-2 cloudless.
- Singapore: data.gov.sg.

Packs are prepared on an internet-connected machine and copied over.

| Doc | What |
|---|---|
| [`docs/SPEC.md`](docs/SPEC.md) | Specification: inputs, frames, layers, mesh, packs, performance |
| [`docs/acceptance.md`](docs/acceptance.md) | Acceptance record against the spec |
| [`docs/research.md`](docs/research.md) | Data-source research and access-path findings |
| [`plans/developmentPlan.md`](plans/developmentPlan.md) | Milestones M0–M8 and what each delivered |

**Performance:** 5 km × 5 km at 0.5 m (10⁸ nodes, real Singapore data) builds
every output in about 53 s, with 4.5 GB peak RAM.

## Install

This project is managed with [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev          # create .venv and install runtime + dev dependencies
uv sync --extra prepare      # (online machine only) extras for `map3d prepare`
```

## Usage

### Command line

```bash
# 1. Online machine: download open data for a box into an offline pack.
#    Add --system-certs behind TLS-intercepting proxies (trusts the OS CA store).
uv run map3d prepare --bbox 103.83 1.26 103.89 1.31 --out marina.pack --system-certs
uv run map3d info marina.pack --verify

# 2. Anywhere, offline: build a scene (axes default to north-aligned).
uv run map3d build --lat 1.2834 --lon 103.8607 --spacing 2 --swath 4000 4000 \
    --axes "[[0.866, -0.5, 0], [0.5, 0.866, 0], [0, 0, 1]]" \
    --pack marina.pack --out out --format npz,geotiff,3dtiles

# 3. Look at it.
uv run map3d view out/tiles
```

Other options:

- `map3d build --datum ellipsoid` switches heights to the WGS84 ellipsoid.
- `--decks K` sets the number of stacked deck levels.
- `--style` selects the mesh style; currently `conceptual` is supported and is
  the default.
- `--building-imagery ortho.tif` uses a high-resolution RGB orthophoto for
  building roof and facade palettes. Otherwise the pack's imagery is used where
  reliable; windows and floor bands remain procedural, not observed details.
- In the viewer, the URL hash (`#cam=...`) holds the camera, so a view can be
  bookmarked or shared.
- `--ground-smoothing M` sets the ground low-pass width in metres (default 90;
  0 keeps FABDEM's raw 30 m bumps).
- `map3d ingest` adds files you obtained yourself to a pack (see the spec,
  §9).

To try it without downloading anything, use a synthetic pack:

```bash
uv run python -c "from map3d.pack import makeSyntheticPack; makeSyntheticPack('demo.pack', terrain='hill')"
uv run map3d build --lat 1.3 --lon 103.8 --spacing 1 --swath 1000 1000 --pack demo.pack --out demo --format npz,3dtiles
```

### Python

```python
from map3d import SceneRequest, buildScene

request = SceneRequest(
    sceneCenter_deg=(1.2834, 103.8607),
    gridSpacing_m=0.5,
    gridAxes_ENU=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # rows: grid x, y, z in ENU
    swathHorizontal_m=5000.0,                         # along grid x
    swathVertical_m=5000.0,                           # along grid y
)
with buildScene(request, pack="marina.pack") as scene:   # large scenes are memory-mapped
    surface = scene.layers["surfaceHeight_ENU_m"]       # (rows, cols), row 0 at -y
    scene.writeNpz("out/scene")                          # scene.npz + scene.json
    scene.writeGeoTiff("out/geotiff")
    scene.write3dTiles("out/tiles")                      # + offline viewer
```

`scene.json` documents the frame, node convention, datum, every layer's units,
the provenance and land-class codes, the deck model and source attributions.
The spec covers the details, including the known limitations (§13): deck
heights are estimated, most buildings are LOD1 (roof shapes only where
mapped), and coarse sources are interpolated.

## Development

```bash
uv run pytest                 # run tests (dev loop, no coverage gate)
uv run ruff check .           # lint
uv run ruff format .          # format
uv run mypy src/map3d         # type check
uv run python scripts/check.py   # GATE: lint + format + types + tests + branch coverage ≥ 90%
uv run pytest -m slow         # design-point benchmark (set MAP3D_BENCH_PACK to a pack)
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

### Third-party code

The offline viewer vendors three.js r170 (MIT licence,
`src/map3d/viewer/static/vendor/three/LICENSE`).
