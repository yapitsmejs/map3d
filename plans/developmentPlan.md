# map3d — Development plan

Implements [`docs/SPEC.md`](../docs/SPEC.md). Naming, code style and the
test-first workflow follow [`CONVENTIONS.md`](../CONVENTIONS.md). Every
milestone lands behind the gate (`uv run python scripts/check.py`: ruff, mypy,
pytest, branch coverage ≥ 90%).

## Component breakdown

```
src/map3d/
  __init__.py            # public API: SceneRequest, buildScene, Scene, __all__
  request.py             # SceneRequest dataclass + strict validation (§4)
  frames.py              # LLA/ECEF/ENU/grid transforms, ortho CRS, geoid lookup
  gridGeometry.py        # node lattice, cell footprints, chunk tiling (§5)
  pack/
    manifest.py          # pack format, versioning, checksums, coverage queries
    reader.py            # windowed raster/vector reads from a pack
    ingest.py            # `ingest`: index user-supplied files
    prepare.py           # `prepare`: online downloaders (optional extra)
    sources/             # one module per source: fabdem.py, overture.py, ...
  layers/
    terrain.py           # ground model, resampling + cell-max
    buildings.py         # footprint/height fusion, precedence, rasterization
    canopy.py
    landCover.py
    decks.py             # elevated-structure levels, deck-height heuristic
    compose.py           # surface/top layer, class precedence, provenance
  scene.py               # Scene object; chunked build orchestration
  io/
    npzWriter.py         # .npz + scene.json
    geotiffWriter.py     # ortho CRS + rotated affine
  mesh/
    terrainMesh.py       # RTIN adaptive triangulation
    buildingMesh.py      # LOD1 extrusion (LOD2 hook)
    deckMesh.py          # swept decks
    imagery.py           # texture drape (Sentinel-2 / BYO)
    tiles3d.py           # 3D Tiles 1.1 tileset + GLB writer
  viewer/                # vendored CesiumJS static page + `view` server
  cli.py                 # `map3d` entry point
  _chunking.py, _numbaKernels.py
```

## Dependencies (added in M0)

| Package | Why |
|---|---|
| `numpy>=2.0` | Grid arrays everywhere. |
| `pyproj>=3.6` | LLA/ECEF/ENU, ortho CRS, EGM2008 geoid (via a local PROJ grid). |
| `rasterio>=1.4` | COG windowed reads, conservative rasterization, GeoTIFF writing. Windows/Linux wheels bundle GDAL. |
| `shapely>=2.0` | Footprint/centerline geometry, deck sweeps. |
| `pyarrow>=16` | GeoParquet in packs (Overture is GeoParquet). |
| `numba>=0.60` | Cell-max pooling and interpolation kernels for the 10-minute target. |
| `typer>=0.12` | CLI. |
| `pygltflib>=1.16` | GLB writing for 3D Tiles content (spike S4 may pick an alternative). |
| `duckdb>=1.1`, `httpx>=0.27` | **`[prepare]` extra only**: Overture bbox queries on S3, and downloads. Never needed on the air-gapped box. |
| dev: `hypothesis>=6` | Property tests for frame math and validation. |

## Spikes (time-boxed, before or alongside M1)

| ID | Question | Feeds |
|---|---|---|
| S1 | How do we bulk-download **Google Open Buildings 2.5D** for a bbox without Earth Engine, and does it cover Singapore/Johor/Riau? | M4, building heights |
| S2 | **data.gov.sg inventory**: which datasets give footprints, heights/storeys, trees, and viaduct/deck geometry or elevation, and under what licences? | M4, M5 |
| S3 | **Ground source**: sanity-check FABDEM (the confirmed default) over Singapore against GLO-30, which is a DSM with buildings and trees baked in. | M3 |
| S4 | **3D Tiles writer**: py3dtiles vs a custom writer on pygltflib. Check implicit tiling, texture handling, validator compliance and speed. | M7 |
| S5 | **Deck-height heuristic**: is OSM `layer` plus nominal clearance plausible against a few known SG flyovers? | M5 |

## Milestones
### M0 — Bootstrap the template ✅ (2026-11-20)
Template set up: package renamed to `map3d`, CLI entry
`map3d = "map3d.cli:main"` (only `--version` so far), dependencies added, README rewritten,
example files and setup guide deleted.
**Done:** the gate is green on the skeleton (5 tests, 94.7% branch coverage).
**Done when** the gate is green on the empty skeleton.

### M1 — Request, frames and grid geometry (pure math, no data)
`request.py`, `frames.py`, `gridGeometry.py`.
**Done when:**
- every §4.1 rejection is tested;
- property tests (hypothesis) show the transforms round-trip;
- node positions and both height representations match an independent pyproj
  computation to ≤ 1 mm over a 5 km swath (acceptance criterion 2);
- the ortho-CRS-plus-affine description reproduces node ENU coordinates
  exactly.

### M2 — Pack format and ingest
`pack/manifest.py`, `pack/reader.py`, `pack/ingest.py`, plus a **synthetic
pack generator** for tests (analytic terrain: plane, slope, Gaussian hill;
box buildings; a single bridge).
**Done when:**
- `map3d ingest` and `map3d info` work;
- windowed reads are verified;
- coverage checks reject scenes outside the pack.

### M3 — Grid v0: terrain + buildings + surface, all writers
`layers/terrain.py`, `layers/buildings.py` (with height precedence),
`layers/compose.py`, `scene.py`, `io/*`, and the `map3d build` command.
**Done when:**
- synthetic packs reproduce the expected cell-max values exactly (acceptance
  criterion 3);
- `.npz` + JSON and GeoTIFF round-trip;
- the GeoTIFF overlays correctly in QGIS on a manual check;
- the upsampling warning fires.

### M4 — `prepare`: real data acquisition
`pack/prepare.py` and `pack/sources/*` for FABDEM/GLO-30, Overture
(buildings, transportation), Google Open Buildings 2.5D, Meta canopy,
WorldCover, Impact Observatory, Sentinel-2 composite, the EGM2008 grid and the
data.gov.sg datasets from S2.
**Done when:**
- a Singapore + Johor + Riau pack builds end to end;
- the manifest records version, licence and checksum per source;
- downloader tests use recorded fixtures (no live network in CI).

### M5 — Full layer set
`layers/canopy.py`, `layers/landCover.py`, `layers/decks.py` (K levels, the S5
heuristic), `featureId` + `features` table, `heightSource` /
`nativeResolution_m` provenance, and class precedence.
**Done when:**
- every §6.1 layer is populated and documented in `scene.json`;
- the synthetic bridge yields the correct deck top, bottom and clearance;
- a small SG golden scene (e.g. 200 m around a known flyover) is stored as a
  regression fixture.

### M6 — Scale and performance
Chunked out-of-core build (`_chunking.py`), numba kernels, parallel chunks,
incremental writers.
**Done when** the design-point benchmark (5 km × 5 km @ 0.5 m, K = 5) runs in
**under 10 minutes** within the RAM budget. The benchmark is a `slow`-marked
test, run on demand and before releases rather than in the default gate.

### M7 — 3D Tiles mesh
`mesh/*`: RTIN terrain, LOD1 buildings (LOD2 hook), swept decks, imagery
drape (Sentinel-2 / BYO), and the tileset writer chosen in S4.
**Done when:**
- the SG reference tileset passes the 3D Tiles validator (in CI);
- mesh and grid agree (sampled mesh surface vs `surfaceHeight` within
  tolerance on synthetic packs);
- the tiles fit within the 10-minute total budget.

### M8 — Viewer, CLI polish, docs, offline acceptance
`viewer/` (vendored CesiumJS), `map3d view`, CLI help, README usage, and an
end-to-end run with the network disabled.
**Done when** acceptance criteria 1–6 in SPEC §12 all pass.

## Dependency order

```
M0 → M1 → M2 → M3 → M5 → M6 → M8
            ↘ M4 ↗      ↘ M7 ↗
S3 → M3   S1, S2 → M4   S5 → M5   S4 → M7
```

M4 (real data) can run in parallel with M3 once M2 fixes the pack format. M7
can start once M5 exists, independent of M6.

## Verification strategy

- **Unit and property tests:** frames and validation (hypothesis).
- **Analytic synthetic packs:** exact expected values for cell-max, decks and
  class precedence.
- **Golden regression:** a small real SG scene, with outputs hashed or
  compared within tolerance.
- **External validators:** 3D Tiles validator; GeoTIFF reopened via rasterio
  with the CRS and transform checked.
- **Benchmark:** a `slow` marker for the design point.
- **Offline guarantee:** a test that builds a scene with sockets disabled.

## Risks

| Risk | Mitigation |
|---|---|
| Google Open Buildings 2.5D is hard to bulk-download or doesn't cover SG | S1; the precedence chain falls back to OSM and data.gov.sg. |
| Deck elevations are unavailable in open data | S5 heuristic, flagged as estimated in provenance. |
| The 10-minute target at 10⁸ nodes, plus tiles | Chunking and numba from M6; mesh density decoupled from grid spacing. |
| Memory (≈ 12 GB if fully materialized) | Chunked build and incremental writers; sparse/compressed deck layers. |
| GLO-30 double-counts buildings and trees as "ground" | FABDEM is the default; S3 sanity-checks it. |
| Licence mix (NC, ODbL) | Internal-use assumption; per-source licence in every output. |
