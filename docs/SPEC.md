# map3d — Specification (v1)

Status: **implemented** (M0–M8), 2026-11-20; see `docs/acceptance.md`. Originally agreed 2026-11-20. All defaults proposed during review
(EGM2008 geoid, FABDEM ground, class precedence, 3.0 m storey, 8 GB RAM) are confirmed.

## 1. Purpose

`map3d` builds a georeferenced 3D model of a rectangular scene anywhere on
Earth, from open data, entirely offline. Given a scene center, grid spacing,
grid orientation and swath size it produces:

1. a **multi-layer grid** (the canonical, simulation-grade output), and
2. a **3D Tiles mesh** (for visualization), built from the same source data.

Fidelity is boosted where better open data exists, starting with Singapore and
the surrounding region (Johor, Riau). The data stack mirrors geosg.app
(Copernicus DEM, Overture, Google Open Buildings 2.5D, Meta/WRI canopy, ESA
WorldCover / Impact Observatory, OSM, data.gov.sg).

**Primary consumers:** simulation/analysis code (RF, radar/sensor, line of
sight) and human visualization.

## 2. Non-goals (v1)

- Tilted grids (grid z must be ≈ Up). The API accepts a full 3×3 axis matrix
  so tilt can be added later without a breaking change.
- Any network access at scene-build time. Only `map3d prepare` touches the
  internet.
- Bring-your-own elevation (DEM/DSM/LiDAR). BYO is supported for imagery only.
- Redistributable outputs. Licensing assumes **internal use only**; non-commercial
  and share-alike sources are allowed (attribution is still embedded).
- REST/HTTP service. The interface is a Python library + CLI.
- Historical/temporal selection. The latest version of each source is used.
- Bathymetry, building interiors, street furniture.

## 3. Platform

- Python **≥ 3.12**, Windows and Linux, pip-installable from binary wheels
  only (no system GDAL).
- Single workstation; no GPU required.

## 4. Inputs

| Parameter | Type | Meaning |
|---|---|---|
| `sceneCenter_deg` | `(lat, lon)` | WGS84 geodetic latitude/longitude of the grid center. |
| `gridSpacing_m` | `float > 0` | Node spacing `s`, identical on both axes. |
| `gridAxes_ENU` | 3×3 array-like | **Rows are axes**: row 0 = grid x, row 1 = grid y, row 2 = grid z, each expressed as a unit vector in the scene-center ENU frame. |
| `swathHorizontal_m` | `float > 0` | Swath extent `Lx` along grid **x**. |
| `swathVertical_m` | `float > 0` | Swath extent `Ly` along grid **y** (an in-plane extent, *not* altitude). |
| `verticalDatum` | `"msl"` (default) \| `"ellipsoid"` | Datum for the "height above datum" layers. MSL = EGM2008 geoid (not Singapore Height Datum). |
| `maxDeckLevels` | `int ≥ 0`, default `5` | `K`, the number of elevated-structure levels stored in the grid. |
| `imageryPath` | path \| `None` | Optional BYO orthophoto (GeoTIFF). It replaces Sentinel-2 in the mesh drape. |
| `pack` | path | Offline data pack to read from (§9). |

### 4.1 Validation (strict; failures raise `ValueError` with a specific message)

Tolerance `axesTol = 1e-6` (overridable).

- `gridAxes_ENU` is 3×3 and finite.
- Each row has unit length: `| ‖R[i]‖ − 1 | ≤ axesTol`.
- Rows are mutually orthogonal: `|R[i]·R[j]| ≤ axesTol` for `i ≠ j`.
- Right-handed: `det(R) = +1` within `axesTol` (equivalently `R[0] × R[1] = R[2]`).
- v1 only: grid z is Up: `‖R[2] − (0, 0, 1)‖ ≤ axesTol`. Therefore grid x and y
  are horizontal, and the matrix is effectively a heading.
- `Lx / s` and `Ly / s` must each be an integer within `1e-9` relative
  tolerance; otherwise reject. No silent rounding.
- `|lat| < 90`, `lon` in `[-180, 180]`.
- The scene footprint must lie fully inside the pack's coverage.

Example: grid y heads 30° clockwise of North, x to its right, z Up:

```python
gridAxes_ENU = [[0.8660254, -0.5,       0.0],   # grid x
                [0.5,        0.8660254, 0.0],   # grid y
                [0.0,        0.0,       1.0]]   # grid z
```

## 5. Geometry conventions

### 5.1 Frames

- **ENU**: local East-North-Up tangent frame at the scene center, with origin
  on the WGS84 ellipsoid at `(lat, lon, h=0)`.
- **Grid frame**: origin at the ENU origin, axes = rows of `gridAxes_ENU`.
  A grid-frame point `(gx, gy, gz)` maps to ENU as
  `p_ENU = gx·R[0] + gy·R[1] + gz·R[2]`, i.e. `p_ENU = Rᵀ · g`.
- **ECEF**: WGS84 Earth-centered, Earth-fixed. Used by 3D Tiles.

### 5.2 Nodes (cell-centered)

`Nx = Lx / s` columns and `Ny = Ly / s` rows. Arrays are indexed
`arr[i, j]` (row, column) with **row 0 at −y** and column 0 at −x:

```
gx_j = (j + 0.5)·s − Lx/2        j = 0 … Nx−1
gy_i = (i + 0.5)·s − Ly/2        i = 0 … Ny−1
```

Each node represents the `s × s` cell centered on it. Plotting needs
`origin="lower"`.

### 5.3 Node column and the two height representations

Each node owns a **vertical column**: the line through `(gx, gy)` on the
tangent plane, parallel to the scene-center Up axis. Every height layer is
reported twice for the surface point(s) found in that column:

- `*_datum_m`: height of the surface point above the chosen datum
  (EGM2008 geoid or WGS84 ellipsoid). Intuitive and GIS-friendly.
- `*_ENU_m`: the Up coordinate of the same point in the scene-center ENU
  frame. This includes Earth curvature, so the ground "falls away" with
  distance (about 0.8 m at 3.5 km). Exact for Cartesian simulation.

Horizontal node positions correspond exactly to an ellipsoidal orthographic
projection centered on the scene center (`+proj=ortho +lat_0=… +lon_0=…
+ellps=WGS84`). The GeoTIFFs use this CRS (§6.3), so they are exact, not
approximate.

### 5.4 Cell aggregation

Every surface-height layer stores the **maximum** surface height within the
node's cell footprint (cell-max):

- Vector features (building footprints, decks) are rasterized conservatively:
  any feature touching the cell contributes.
- Raster sources are resampled (bilinear for heights, nearest for classes)
  onto a supersampled lattice, then max-pooled per cell.
- If the requested spacing is finer than a source's native resolution, the
  source is interpolated and a **warning** is logged and recorded in metadata.
  No error is raised.

## 6. Grid output

### 6.1 Layers

`K = maxDeckLevels`. Heights are float32 meters; "no data" is `NaN`.

| Layer | Shape | dtype | Meaning |
|---|---|---|---|
| `groundHeight_datum_m` / `groundHeight_ENU_m` | (Ny, Nx) | f4 | Bare-earth ground. |
| `surfaceHeight_datum_m` / `surfaceHeight_ENU_m` | (Ny, Nx) | f4 | Top surface, i.e. the cell-max DSM over terrain, buildings, canopy and decks. |
| `buildingHeight_m` | (Ny, Nx) | f4 | Building height above ground (NaN where there is no building). |
| `canopyHeight_m` | (Ny, Nx) | f4 | Vegetation canopy height above ground (NaN/0 where there is none). |
| `deckTopHeight_datum_m` / `deckTopHeight_ENU_m` | (K, Ny, Nx) | f4 | Top of elevated structure level k (k = 0 is the lowest). |
| `deckBottomHeight_datum_m` / `deckBottomHeight_ENU_m` | (K, Ny, Nx) | f4 | Underside of level k. Clearance below level k = bottom[k] − top[k−1] (or − ground when k = 0). |
| `landClass` | (Ny, Nx) | u1 | Class of the top-most visible surface (§6.4). |
| `featureId` | (Ny, Nx) | i8 | ID of the top-most feature (building/deck); 0 = none. Indexes the `features` table. |
| `heightSource` | (Ny, Nx) | u1 | Enum of the source that set `surfaceHeight` (§8). |
| `nativeResolution_m` | (Ny, Nx) | f4 | Native resolution of that source (0 for vector-derived). |

Also included: `features`, a table with one row per feature (`featureId`,
`kind`, `sourceDataset`, `sourceFeatureId`, `height_m`, `heightSource`,
`deckLevel`).

### 6.2 `.npz` + JSON sidecar

- `scene.npz` holds every layer above (deck layers in compressed form; they
  are mostly NaN).
- `scene.json` holds:
  - the echoed request;
  - the frame: center LLA and ECEF, `gridAxes_ENU`, the ENU→ECEF rotation,
    node convention, array order, `Nx`, `Ny`;
  - the datum, with the geoid undulation at the center;
  - layer descriptions, units and dtypes;
  - the class table;
  - the source table (dataset, version, licence, attribution, native
    resolution);
  - warnings;
  - the `map3d` version and pack hash.

### 6.3 GeoTIFF

- One file per layer; deck layers are one K-band file each.
- CRS: ellipsoidal orthographic centered on the scene center (§5.3), with an
  affine geotransform that encodes the rotation and the row-0-at-−y
  orientation. Pixel values and array order are identical to the `.npz`.
- Tags: units, datum, layer description and attribution.

### 6.4 Land classes

ESA WorldCover codes (10 tree, 20 shrub, 30 grass, 40 cropland, 50 built-up,
60 bare, 70 snow/ice, 80 water, 90 wetland, 95 mangrove, 100 moss/lichen),
extended with vector-derived classes: 200 building, 210 road, 220
bridge/elevated deck, 230 rail. Precedence for the top surface is
building > deck > road/rail > land cover.

- **Land cover:** sampled nearest-pixel at the node footprint. Impact
  Observatory codes are remapped to WorldCover codes.
- **Ground roads/rails:** drawn with their width (defaults: road 7 m, rail
  4 m), cell centres only.
- **Canopy:** a raster (metres above ground) is cell-max pooled; tree points
  become crown discs of radius max(2 m, 0.25 × height). `surfaceHeight` takes
  ground + canopy where that is the highest candidate (provenance `CANOPY`).

## 7. Mesh output — 3D Tiles

- OGC **3D Tiles 1.1** with glTF 2.0 (GLB) content and a quadtree LOD
  hierarchy. The root transform is ENU→ECEF at the scene center; vertex heights
  are ellipsoidal (as 3D Tiles requires).
- Built **from source geometry, not from the grid**:
  - **Terrain:** adaptive triangulation (e.g. RTIN) of the ground model, with
    an error tolerance, so mesh density is independent of `gridSpacing_m`.
  - **Buildings:** best available geometry. LOD1 (footprint extruded to a flat
    roof) by default, and LOD2 roof shapes where open data provides them.
  - **Elevated structures:** true 3D decks (road/rail centerline swept to
    width, with thickness) with open space underneath. Every level is
    included, not just `K`.
- **Appearance:** two styles (`write3dTiles(style=...)`, `map3d build --style`):
  - **`conceptual` (default):** a clean, map-like look in the spirit of
    geosg.app. Terrain, roofs and deck tops get flat pastel colours:
    - **Land cover:** WorldCover, sampled bilinearly so its 10 m cells blend
      softly. Grass is pale green, built-up land warm light grey, water blue.
    - **Crisp overlays from the grid:** grey roads, rail and decks, grey roofs,
      and a soft green on the ground under canopy ≥ 2 m (not on roads), so gaps
      between crowns and distant views read as wooded.
    - **Shading:** materials are lit, with vertex normals (smooth on terrain
      and tree crowns, flat on roofs and walls). The renderer shades the
      shapes; walls are white.
    - **Not drawn:** road labels, lane markings, windows and roof colours (no
      open data for the last two).
  - **`satellite`:** the Sentinel-2 cloud-free composite (~10 m), or the BYO
    orthophoto, draped unlit. Chosen automatically when `--imagery` is given;
    an error if no imagery is available.
- `map3d view <tilesetDir>` serves the tileset on localhost with a bundled
  offline viewer.
  - **Why three.js:** the viewer is vendored three.js r170 (MIT, ~0.9 MB) plus
    a small 3D Tiles traversal. CesiumJS was the original plan, but it is
    ~30 MB and needs configuration (Ion token, base layers) to run offline.
  - **How it draws:** in the local ENU frame, refining by screen-space error.
    A parent stays visible until all its children have loaded.
  - **Interoperability:** the tiles are standards-compliant 3D Tiles 1.1 (the
    official `3d-tiles-validator` 0.6.1 reports 0 errors, 0 warnings), so
    Cesium and other viewers can load them too.
- **As built (M7):**
  - **Layout:** a quadtree in grid coordinates with leaves ≤ 512 m and
    `REPLACE` refinement.
  - **Terrain:** RTIN (Martini) on a 129² lattice per tile. The tolerance is
    0.25 m at the leaves and doubles per level.
  - **Terrain is the true ground.** Vegetation is never draped as a surface
    (that made trees look like green mounds); it is drawn as trees.
  - **Trees** (`withTrees=True`, the default; `map3d.mesh.trees`): one rounded
    crown (icosphere, 80 triangles, three shades of green) on a trunk per tree,
    standing on the ground model:
    - **Mapped tree points** (`trees` role) first, with crown radius
      0.3 h + 1 m (2–8 m).
    - **Canopy raster:** lightly smoothed (2 m FWHM); treetops are pixels
      ≥ 2 m tall that are the highest within 1.5 m. Treetops within half a
      crown of a taller one are merged. Every canopy pixel then joins the
      nearest treetop it is connected to through canopy (≤ 6 m away), and a
      crown's radius is that of a disc with its region's area (1.5–6 m, at most
      0.6 h + 1.5 m). Canopy no treetop reaches (smooth woods) is seeded again
      at its tallest points, a crown apart (up to 4 rounds, until < 1% is
      left). Crowns thus add up to the canopy without sprouting mini-trees on
      crown edges.
    - **Not on roofs or decks**, and only inside the swath.
    - **Tiles:** written with `EXT_mesh_gpu_instancing` (one shape plus a
      position and size per tree). All trees in the leaves; trees ≥ 8 m one
      level up; none higher (the ground green carries distant views).
    - **Loyang 4.5 km:** 69,000 trees (median 6.7 m tall, 3.3 m crown radius;
      2.5 km² of crowns over 3.9 km² of canopy ≥ 2 m), found in ~4 s.
    - The grid layers are unchanged: they keep bare ground and canopy height.
  - **Buildings:** parent tiles keep buildings ≥ 10 m × 2^(levels above the
    leaves − 1), so distant views keep the skyline.
  - **Decks:** in leaves and their parents.
  - **Geometric error:** max(terrain tolerance, omitted building height).
  - **Texture:** one per tile, projected top-down onto terrain, roofs and deck
    tops. Conceptual: 512² PNG, overlays sampled 2× per axis and averaged
    (anti-aliased edges, ~0.55 m pixels at the leaves). Satellite: 256² JPEG.
    Walls and deck sides use RGBA vertex colours.
  - **Loyang 4.5 km, conceptual with trees:** 95 MB of tiles, written in
    ~80 s.
  - **Viewer camera in the URL:** `#cam=E,N,U,targetE,targetN,targetU` (scene
    ENU metres). The hash follows the camera, so views can be bookmarked or
    shared.
  - **Attribution:** stored in `asset.extras.attribution`.

## 8. Data sources and precedence

| Role | Global source | SG / regional boost |
|---|---|---|
| Ground (bare earth) | FABDEM 30 m (Copernicus GLO-30 with buildings and forest removed; CC BY-NC-SA). Fallback: Copernicus GLO-30. | — |
| Building footprints | Overture Maps buildings (ODbL) | HDB Existing Building footprints; URA Master Plan building outlines (lowest priority, indicative) |
| Building heights | Google Open Buildings 2.5D (CC BY 4.0, ~4 m effective); OSM `height` / `building:levels` via Overture | HDB Property Information `max_floor_lvl` (storeys) |
| Canopy height | Meta/WRI global 1 m canopy height (CC BY 4.0) | — (NParks TreesSG is not downloadable) |
| Land cover | ESA WorldCover 2021 10 m; Impact Observatory 10 m annual LULC | — |
| Roads, rail, bridges | Overture transportation (OSM-derived: `is_bridge`, `level`, width) | URA Master Plan rail lines (`GRND_LEVEL` marks viaducts) |
| Elevated-structure footprints | — | LTA Vehicular Bridge/Underpass/Flyover polygons (location only, no deck heights) |
| Imagery | Sentinel-2 cloudless 2024 by EOX (s2maps.eu WMTS, zoom 14 ≈ 10 m; CC BY-NC-SA 4.0) | BYO orthophoto (`--imagery`) |
| Geoid | EGM2008 grid (PROJ-data `us_nga_egm08_25.tif`) | — |

**Building height precedence (authoritative first):** local government data →
OSM explicit `height` → Google Open Buildings 2.5D → OSM `building:levels` ×
storey height (default 3.0 m) → default height. Raster heights under 2 m
count as missing (the raster saw ground, not roof) and fall through to the next
source. The default height is 3 m (one storey) for footprints under 50 m²
(sheds, kiosks, shelters) and 9 m otherwise. The winning
source is recorded per feature and per node.

**Ground smoothing:** FABDEM is a 30 m model with metre-level noise and
leftover building bumps, so sampling it bilinearly at fine spacing gives a
faceted, bumpy surface. map3d low-passes the ground source with a Gaussian of
90 m FWHM (three FABDEM pixels) before anything samples it:
`buildScene(groundSmoothing_m=90)`, `map3d build --ground-smoothing 90`. 0 turns
it off. Details:
- **Edges and no-data:** the window is read beyond the scene by the filter
  radius, and no-data is excluded from the average, so planes are kept exactly
  and coastlines are not dragged toward zero.
- **Effect:** bumps narrower than ~90 m are removed. Everything that uses the
  ground (grid layers, building bases, ground correction, deck ends, mesh)
  sees the same smoothed surface. The `nativeResolution_m` layer reports 90 m
  for ground nodes, and `scene.json` records `groundSmoothing_m`.
- **On Loyang:** 10 m-scale curvature dropped 5× (p95 0.56 → 0.11 m) and the
  ground moved by 0.77 m RMS. The largest change fills a FABDEM pit (−12.6 →
  +7.2 m). Narrow genuine features (embankments, small hills under ~90 m) are
  flattened too; the largest drop was 8.6 m.

**Ground under buildings (FABDEM residue):** FABDEM's machine-learned
building removal sometimes fails for large buildings (depots, warehouses),
leaving a building-shaped mound in the "ground". It peaks mid-footprint and can
be taller than the roof, which buried e.g. Loyang Bus Depot. map3d corrects
this (`buildScene(correctGround=True)`, the default):

1. Fit a robust plane to the ground on rings 20–45 m outside each footprint.
2. If the ground inside stands clearly above that plane (90th-percentile
   excess > 1 m), correct this building.
3. The correction zone starts at the footprint + 10 m and grows across
   connected ground more than 0.5 m above the plane, up to 60 m from the
   walls. Residue often spills past the walls, or past the mapped outline.
4. Inside the zone, ground is capped at the plane, fading back to the raw
   ground over the last 20 m of the reach, so no cliffs form.

Ground is only ever lowered. The correction applies to the grid ground layer,
building bases, deck ends and the 3D Tiles terrain. `scene.json`
`groundCorrection` reports how many buildings were corrected, the largest
lowering and the zone area.

On the Loyang 4.5 km scene, with the default 90 m ground smoothing applied
first (smoothing already absorbs most small residue):
- **Corrected:** 108 buildings (833 without smoothing), lowered by up to
  17.5 m; correction zones cover 2.1 km².
- **The depot:** stands 12.6 m (median) of its 13.3 m height; 10.8 m without
  smoothing.
- **Cost:** about 16 s of build time.
- **Caveat:** without open LiDAR, genuine raised ground near buildings
  (embankments, landscaped mounds) can't be told apart from residue, so it
  may also be lowered.

**Deck heights:** no open source has deck elevations or clearances,
Singapore included (see `docs/research.md`). LTA flyover polygons and URA
aboveground rail locate decks but give no height, so v1 estimates them
(provenance `DECK_ESTIMATED`; parameters are recorded in `scene.json`
`deckModel`):

- **Footprints:**
  - Bridge pieces (transport `isBridge`) are buffered by their width, or a
    default (road 8 m, rail 8 m, footway 3 m).
  - Elevated-structure outlines are added unless more than half their area is
    already covered.
- **Top height:** ground interpolated linearly between the bridge's two ends
  (along the centerline) + level × per-level height (road 7 m, rail viaduct
  10 m, footbridge 6 m). Levels ≤ 0 or missing count as 1. Outline-only decks
  use the median ground on their outline.
- **Underside:** top − 1.5 m.
- **Per node:** tops within 3 m merge into one level (parallel carriageways,
  piece joints). The lowest K levels are stored, and the highest deck always
  sets `surfaceHeight`. Overflow is reported as a warning.
- **Known limitation:** high-clearance bridges (e.g. Benjamin Sheares, ~25–30 m)
  are underestimated.

**Transport kinds:** `road`, `rail` and `footway`. Overture road classes
footway/pedestrian/path/cycleway/steps/bridleway/sidewalk/crosswalk map to
`footway`. Footways never set the ground land class.

## 9. Offline data packs

A pack is a self-contained directory, portable to air-gapped machines:

```
sg.pack/
  manifest.json      # bbox, sources (role, licence, attribution, access date, CRS),
                     # per-file sha256 / size / extent, pack format version
  raster/<sourceId>/ # GeoTIFFs (COGs from `prepare`): ground, geoid, canopy, imagery, ...
  vector/<sourceId>/ # GeoParquet (WKB, lon/lat) in the role's canonical schema
  ATTRIBUTION.md     # generated from the manifest
```

- `map3d prepare --bbox <minLon> <minLat> <maxLon> <maxLat> --out sg.pack
  [--sources …] [--cache DIR] [--system-certs | --ca-bundle PEM] [--year 2023]`
  runs on an internet-connected machine with the `prepare` extra installed.
  - **Clipping:** rasters are clipped with windowed HTTP range reads in their
    native CRS. Only the needed blocks are fetched: FABDEM members come out of
    the 10° zips through the zip directory, and Google 2.5D is decimated to
    ~4 m. Vectors are bbox-filtered (Overture via Arrow S3 pushdown;
    data.gov.sg via its poll-download API, throttled to 2 requests per 10 s).
  - **Validation:** everything then goes through the same validation as
    `ingest`.
  - **Defaults:** all sources except `copernicusGlo30` (used automatically if
    FABDEM yields nothing) and `impactObservatoryLulc`; Singapore sources only
    when the box overlaps Singapore.
  - **Imagery:** `sentinel2Composite` (EOX s2cloudless WMTS tiles mosaicked to an RGB GeoTIFF) is a default source.
  - **Failures:** a failing source is reported and skipped; the run fails only
    if ground or geoid is missing.
  - **TLS interception:** on networks with TLS interception, `--system-certs`
    exports the OS trust store (plus certifi) to a PEM file used by `httpx`
    and Arrow S3. GDAL never makes network requests itself: raster reads go
    through a Python range-read `opener`, so they share the same verified TLS
    settings. Verification is never disabled.
- `map3d ingest --pack sg.pack --source <sourceId> [--bbox …] [--accessed DATE] <files…>`
  validates **pack-ready** files and copies them into a pack:
  - GeoTIFF with a CRS for raster sources (one CRS per source);
  - GeoParquet in the role's canonical schema for vector sources (columns
    listed in `map3d.pack.registry.ROLES`).

  Converting raw downloads into that form is done by per-source adapters,
  shared with `prepare`. All files are validated before any is copied, and
  `map3d info <pack> --verify` re-checks the checksums.
- Coverage: raster sources are checked against file extents. Vector sources
  are only checked for presence, because `prepare` clips them to the pack.
- Scene builds never touch the network. If a source is missing from a pack,
  the build degrades with a warning; a missing ground source is an error.
- First target packs: Singapore plus the surrounding region (Johor, Riau).

## 10. Interfaces

### 10.1 Python

```python
from map3d import SceneRequest, buildScene

request = SceneRequest(
    sceneCenter_deg=(1.2834, 103.8607),
    gridSpacing_m=0.5,
    gridAxes_ENU=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    swathHorizontal_m=5000.0,
    swathVertical_m=5000.0,
    verticalDatum="msl",
    maxDeckLevels=5,
)
scene = buildScene(request, pack="sg.pack")   # validates, then builds the grid

scene.layers["surfaceHeight_ENU_m"]            # numpy array (Ny, Nx)
scene.metadata                                 # dict, same content as scene.json
scene.writeNpz("out/scene")                    # scene.npz + scene.json
scene.writeGeoTiff("out/geotiff")
scene.write3dTiles("out/tiles", imageryPath=None)
```

### 10.2 CLI

```
map3d build   --lat 1.2834 --lon 103.8607 --spacing 0.5 --axes axes.json \
              --swath 5000 5000 --datum msl --decks 5 --pack sg.pack \
              --out out/ --format npz,geotiff,3dtiles [--imagery ortho.tif] \
              [--ground-smoothing 90] [--style conceptual|satellite]
map3d prepare --bbox 103.5 1.1 104.2 1.5 --out sg.pack
map3d ingest  --pack sg.pack --source overtureBuildings ./buildings.parquet
map3d view    out/tiles
map3d info    sg.pack
```

`--axes` takes a JSON file or an inline JSON 3×3 list (rows = axes).

## 11. Performance and scale

- **Design point:** up to 5 km × 5 km at 0.5 m spacing, i.e. 10 000 × 10 000
  = 10⁸ nodes.
- **Target:** under about 10 minutes on one workstation from a local pack, for
  all grid layers plus 3D Tiles.
- **Memory:** about 26 float32 layers at K = 5 is ≈ 12 GB if fully
  materialized, so processing is **chunked** (grid tiles) and written
  incrementally. Peak RAM budget is ≤ 8 GB at the design point.
- **How (M6):**
  - Node footprints, raster pixel coordinates and the geoid are solved exactly
    on a global coarse lattice (≤ 50 m) and interpolated. Interpolation error
    is < 1 µm in position and ≤ 0.07 mm in heights (float32).
  - Hot loops are numba `nogil` kernels.
  - Chunks run on a thread pool (`workers`, default min(8, CPUs)). Results are
    bitwise identical for any chunking or worker count.
  - Layers are memory-mapped `.npy` files when they would exceed
    `memoryBudget_GB` (default 2) or when `workDir` is given.
  - GeoTIFFs are written in row bands with multi-threaded deflate.
- **Measured** (24-core workstation, real Marina Bay pack, design point,
  K = 5):

  | Stage | Time |
  |---|---|
  | Build | 16.4 s |
  | Write `.npz` (11 GB) | 10.6 s |
  | Write GeoTIFFs (1 GB) | 11.2 s |
  | Write 3D Tiles (341 tiles, 33 MB) | 15.0 s |
  | **Total** | **53.2 s** |

  Peak private memory: **4.5 GB**. Reproduce with `scripts/benchmark.py`, or
  `pytest -m slow` with `MAP3D_BENCH_PACK` set.

## 12. Acceptance criteria

1. Validation rejects every malformed input in §4.1 with a specific message.
2. Node positions and both height representations match an independent pyproj
   computation to ≤ 1 mm horizontal and ≤ 1 mm vertical across the design
   swath.
3. Synthetic packs (analytic terrain and box buildings) reproduce the expected
   cell-max values exactly.
4. A Singapore reference scene builds offline (network disabled) and the
   outputs open correctly: the GeoTIFF in QGIS, the tiles in the bundled
   viewer, and the tileset passes the 3D Tiles validator.
5. The design-point benchmark finishes in under 10 minutes within the RAM
   budget.
6. Every output carries complete source attribution and licence metadata.

## 13. Known limitations

- At 0.5 m, terrain (30 m), land cover (10 m) and building heights (~4 m
  raster) are interpolated. Only footprint edges are truly sharp.
- Open LOD2 roof data is scarce, so expect mostly LOD1 buildings even in
  Singapore.
- Deck elevations are mostly heuristic (§8).
- Sentinel-2 imagery is ~10 m, so it gives context only. Use BYO imagery for
  detail.
