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
| S1 ✅ | How do we bulk-download **Google Open Buildings 2.5D** for a bbox without Earth Engine, and does it cover Singapore/Johor/Riau? → Public GCS bucket, COG-like tiles, covers all three; window-read and resample to ~4 m. See `docs/research.md`. | M4, building heights |
| S2 ✅ | **data.gov.sg inventory**: which datasets give footprints, heights/storeys, trees, and viaduct/deck geometry or elevation, and under what licences? → HDB footprints + storeys (join gap), URA outlines, LTA flyover polygons, MP rail viaducts; no deck heights, LiDAR or tree inventory. See `docs/research.md`. | M4, M5 |
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

### M1 — Request, frames and grid geometry (pure math, no data) ✅ (2026-11-20)
`request.py`, `frames.py`, `gridGeometry.py`, `geoid.py`.
**Done when:**
- every §4.1 rejection is tested;
- property tests (hypothesis) show the transforms round-trip;
- node positions and both height representations match an independent pyproj
  computation to ≤ 1 mm over a 5 km swath (acceptance criterion 2);
- the ortho-CRS-plus-affine description reproduces node ENU coordinates
  exactly.

**Done:** 73 tests, 99.7% branch coverage. The transforms match pyproj
`cart`/`topocentric` to 1 µm over ±4.5 km. PROJ's ellipsoidal `ortho` equals
ENU E/N to 4e-10 m for points on the ellipsoid. The geoid is a bilinear
`GeoidGrid` loaded from GeoTIFF (the real EGM2008 grid arrives with packs in
M2/M4).

**Notes for M3/M6:**
- `columnPointAtHeight` runs at about 74 s per 10⁸ nodes in pure NumPy. Solve
  the column once (footprint), then convert every height layer with the
  per-column linear relation `u = u0 + k·h`; it is exact to < 0.1 mm over
  500 m, so the solver isn't needed per layer. Port the hot paths to numba in M6.
- A raised point's geodetic position drifts from its column footprint by about
  h·d/R (e.g. 5.5 cm at h = 100 m, d = 3.5 km). Sample sources at the column
  point for the relevant height, not only at the footprint.

### M2 — Pack format and ingest ✅ (2026-11-20)
`pack/manifest.py`, `pack/reader.py`, `pack/ingest.py`, plus a **synthetic
pack generator** for tests (analytic terrain: plane, slope, Gaussian hill;
box buildings; a single bridge).
**Done when:**
- `map3d ingest` and `map3d info` work;
- windowed reads are verified;
- coverage checks reject scenes outside the pack.

**Done:** 172 tests, 99.7% branch coverage.
- **Modules:**
  - `pack/registry.py`: roles, canonical vector schemas, and sources in
    priority order, including the Singapore sources from S2.
  - `pack/geoparquet.py`
  - `pack/manifest.py`: per-file sha256 and extent, plus `accessedDate` for
    Singapore Open Data Licence notices.
  - `pack/ingest.py`: validates every file before copying any.
  - `pack/reader.py`: grid-snapped raster mosaics, bilinear/nearest sampling
    in any CRS, vector selection by box, coverage checks, `verify`.
  - `pack/synthetic.py`: plane/hill terrain, 3 buildings, a bridge with a deck
    footprint and a ground road, canopy patch, land cover, trees, constant
    geoid.
- **Also added:** `GridGeometry.footprintBounds_deg`, and the
  `map3d ingest` / `map3d info [--verify]` commands.
- **Design note:** vector coverage is checked by presence only. Feature
  extents are not clip extents.

### M3 — Grid v0: terrain + buildings + surface, all writers ✅ (2026-11-20)
`layers/terrain.py`, `layers/buildings.py` (with height precedence),
`layers/compose.py`, `scene.py`, `io/*`, and the `map3d build` command.
**Done when:**
- synthetic packs reproduce the expected cell-max values exactly (acceptance
  criterion 3);
- `.npz` + JSON and GeoTIFF round-trip;
- the GeoTIFF overlays correctly in QGIS on a manual check;
- the upsampling warning fires.

**Done:** 234 tests, 99.7% branch coverage.
- **Modules:**
  - `layers/columns.py`: one exact footprint solve per node, then
    `u = u0 + k*h` with `k = 1/(Up0 . n_local)`; matches the exact solver to
    1 µm at 5 km.
  - `layers/terrain.py`: cell-max over a sub-lattice that includes the cell
    corners, subdivided up to 8x when the source is finer than the grid.
  - `layers/buildings.py`: priority fusion with >50% overlap dedupe, the full
    height precedence, raster-height median inside footprints, base at the
    lowest ground on the outline, roof-height pixel footprints, `all_touched`
    with the tallest roof winning.
  - `layers/provenance.py`, `scene.py`, `io/npzWriter.py`,
    `io/geotiffWriter.py`.
- **CLI:** `map3d build` (npz, geotiff; `--axes` takes inline JSON or a file,
  identity by default).
- **Checked:** a 1000×1000 render of a rotated synthetic scene puts every
  building exactly where predicted.
- **Timing (unoptimised):** 1M nodes in ~4 s of compute, so the 10⁸-node
  design point is ~400+ s for the grid alone. M6 must bring that down
  (coarse-grid footprint interpolation, numba, parallel chunks).
- **Defaults:** default building height 9.0 m; duplicate threshold 50%
  overlap.

### M4 — `prepare`: real data acquisition ✅ (2026-11-20)
`pack/prepare.py` and `pack/sources/*` for FABDEM/GLO-30, Overture
(buildings, transportation), Google Open Buildings 2.5D, Meta canopy,
WorldCover, Impact Observatory, Sentinel-2 composite, the EGM2008 grid and the
data.gov.sg datasets from S2.
**Done when:**
- a Singapore + Johor + Riau pack builds end to end;
- the manifest records version, licence and checksum per source;
- downloader tests use recorded fixtures (no live network in CI).

**Inputs from S1/S2** (`docs/research.md`):
- **Open Buildings 2.5D:** read from `storage.googleapis.com/open-buildings-temporal-data/v1/`
  using the manifest `31_EPSG_32648_{year}.json`, with windowed `/vsicurl/`
  reads resampled to ~4 m. Never download whole tiles (~6.5 GB/year for
  Singapore alone). Use the latest year (2023).
- **data.gov.sg:** use `poll-download` (CSV needs `initiate-download`) and
  throttle to 2 requests per 10 s without a key; optionally accept an API key.
- **HDB adapter:** join footprints (BLK_NO, ST_COD, postal code) to Property
  Information (`blk_no`, street name, `max_floor_lvl`). There's no shared key,
  so match block number plus nearest geocoded address (OneMap search). Check
  the result against HDB3D heights.
- **Adapters write the canonical schemas**, so `ingest` and `prepare` share
  them:
  - URA rail: `GRND_LEVEL == ABOVEGROUND` maps to `isBridge`.
  - LTA flyovers map to `elevatedStructures`.

**Done:** 307 tests, 99.1% branch coverage. Live run verified (see below).
- **Layout:** the code lives in `map3d/prepare/` (network-only; needs the
  `prepare` extra):
  - `certs.py`: OS trust-store export.
  - `http.py`: retries, per-host throttle, cached downloads, and
    `HttpRangeFile` as rasterio's `opener`.
  - `rasterClip.py`: windowed clip with decimation aligned to whole blocks.
  - `sources/rasters.py`: FABDEM (zip members only), GLO-30, EGM2008,
    WorldCover, Meta canopy.
  - `sources/googleBuildings.py`: manifests, UTM tile selection, band 2 at
    ~4 m.
  - `sources/overture.py`: Arrow S3 bbox pushdown; linear-referenced segment
    splitting.
  - `sources/dataGovSg.py`: poll-download plus four adapters, including the
    98.6% HDB street-code join.
  - `pipeline.py`: orchestration, GLO-30 fallback, report.
  - `map3d prepare` CLI: `--system-certs` / `--ca-bundle`.
- **Changes from the original plan:** `duckdb` was dropped (Arrow S3 is
  enough), `httpx` was added to dev, and imagery moved to M7.
- **Environment finding:** TLS interception on this network. GDAL's curl
  can't be pointed at the organisation's CA, so GDAL never makes network
  requests itself (`docs/research.md`).
- **Live run** (Marina Bay, 0.06°×0.05° ≈ 6.7×5.6 km):
  - All 11 sources were ingested in 5 min 10 s; the pack is 6.8 MB.
  - A 4 km × 4 km scene at 2 m (4M nodes) built in 12.6 s.
  - The render is recognisable: CBD towers, Marina Bay Sands at 207 m
    (explicit OSM height), the ArtScience Museum, Gardens by the Bay, and
    MBFC T3 at 245 m.
- **Observations for later milestones:**
  - Google 2.5D heights are capped at 100 m, so tall towers rely on OSM
    explicit heights (precedence already does this).
  - 577 of 4,842 scene buildings fell back to the 9 m default.
  - FABDEM's 30 m coastline shows as stair-steps in the ground layer (M5 land
    cover could mask water).
  - There's no Singapore trees source, so `trees` is always missing there.

### M5 — Full layer set ✅ (2026-11-20)
`layers/canopy.py`, `layers/landCover.py`, `layers/decks.py` (K levels, the S5
heuristic), `featureId` + `features` table, `heightSource` /
`nativeResolution_m` provenance, and class precedence.
**Done when:**
- every §6.1 layer is populated and documented in `scene.json`;
- the synthetic bridge yields the correct deck top, bottom and clearance;
- a small SG golden scene (e.g. 200 m around a known flyover) is stored as a
  regression fixture.

**Done:** 337 tests, 99.1% branch coverage.
- **Modules:**
  - `layers/canopy.py`: raster cell-max plus tree crown discs.
  - `layers/landCover.py`: WorldCover / Impact Observatory remap, plus ground
    roads/rails with their width; footways excluded.
  - `layers/decks.py`: bridge buffers plus uncovered outlines; tops
    interpolated between bridge ends + level height; per-node level stacking
    with a 3 m merge tolerance; K storage plus overflow count.
  - `layers/pixels.py`: shared lon/lat → grid-pixel conversion at height.
- **Scene changes:**
  - Surface = max(roof, highest deck, ground + canopy, ground). On ties,
    provenance prefers building > deck > canopy > ground.
  - New `heightSource` codes `CANOPY` and `DECK_ESTIMATED`.
  - `featureId` covers decks too: their ids follow the buildings in the
    features table.
  - `scene.json` gains `landClassCodes` and `deckModel`.
- **New transport kind `footway`** (from Overture `class`), found on real data:
  park paths had been drawn as 7 m roads, and footbridges as 7 m-high decks.
- **Marina Bay check** (4 km × 4 km at 2 m):
  - Built in ~25 s (about 2× M3; M6 target).
  - 186 decks; 1,427 nodes have two levels.
  - Classes: water 30%, building 18%, built-up 14%, tree 14%, grass 12%,
    road 7%, deck 2%.
  - Decks render on the East Coast Parkway viaduct, Helix and Esplanade
    bridges, and the expressway viaducts.
- **Known limitation:** deck heights are nominal (7 m per level). The
  Benjamin Sheares Bridge (~25–30 m clearance) is underestimated, and there's
  no open data to do better.

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
