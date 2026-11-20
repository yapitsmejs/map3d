# Acceptance record (SPEC section 12)

Checked 2026-11-20 on a 24-core Windows workstation, with a real Singapore
(Marina Bay) pack built by `map3d prepare` and synthetic packs.

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Validation rejects every malformed §4.1 input with a specific message | ✅ | `tests/test_request.py` (32 cases incl. hypothesis) |
| 2 | Node positions and both height representations match pyproj to ≤ 1 mm | ✅ | `tests/test_frames.py` (1 µm, ±4.5 km); `tests/test_columns.py` (interpolated footprints 1e-10°, ENU heights 1 µm over a 5 km swath); `tests/test_gridGeometry.py` (ortho CRS + affine exact) |
| 3 | Synthetic packs reproduce expected cell-max values exactly | ✅ | `tests/test_terrain.py`, `test_buildings.py`, `test_canopy.py`, `test_decks.py`, `test_scene.py` |
| 4 | Singapore scene builds offline; outputs open; tileset passes the 3D Tiles validator | ✅ / ⚠ | See below |
| 5 | Design point < 10 min within the RAM budget | ✅ | 53.2 s total, 4.5 GB peak private (`scripts/benchmark.py`, `pytest -m slow`) |
| 6 | Every output carries complete attribution and licence metadata | ✅ | `scene.json` `sources[]`; GeoTIFF `attribution` tag; `tileset.json` `asset.extras.attribution`; pack `ATTRIBUTION.md` |

## Criterion 4 in detail

**Verified:**
- **Offline build:** `tests/test_offline.py` builds a scene and writes npz,
  GeoTIFF and 3D Tiles with socket connections forbidden. It makes zero
  attempts.
- **Real Singapore data:** the Marina Bay pack (11 sources plus imagery) was
  built live. Scenes at 4 km / 2 m and 5 km / 0.5 m build from it; renders
  are recognisable, with Marina Bay Sands at 207 m and MBFC T3 at 245 m.
- **GeoTIFF:** read back through GDAL/rasterio with exact georeferencing. A
  pixel centre maps to its node to 1e-6 m (`tests/test_geotiffWriter.py`).
- **3D Tiles:** the official `3d-tiles-validator` 0.6.1 reports **0 errors,
  0 warnings, 0 infos** for the Marina 4 km tileset (85 tiles) and the 5 km
  design-point tileset (341 tiles). The structural validator in
  `tests/_support.py` runs in CI.
- **Viewer serving:** `map3d view` serves the viewer, ES modules
  (`text/javascript`), `tileset.json` and GLBs (`model/gltf-binary`). Checked
  over HTTP, and `viewer.js` parses as an ES module.

**Not yet verified:**
- ⚠ **Visual viewer check:** no browser automation was available in this
  session. To check by hand: `map3d view <out>/tiles` and look at the page.
- ⚠ **QGIS:** QGIS is not installed here, so the GeoTIFFs have not been
  opened in it. GDAL reads them correctly. QGIS renders rotated
  geotransforms by warping on the fly.

## Reproduce

```bash
uv sync --extra prepare
uv run map3d prepare --bbox 103.83 1.26 103.89 1.31 --out marina.pack --system-certs
uv run python scripts/benchmark.py --pack marina.pack --lat 1.285 --lon 103.86 --out bench
# official validator (needs Node; e.g. `uvx --from nodejs-wheel npm install 3d-tiles-validator`)
node node_modules/3d-tiles-validator/build/main --tilesetFile bench/tiles/tileset.json
```
