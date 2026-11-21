# Research spikes

Findings that shape `map3d prepare` (M4) and the layer builders (M3/M5).
Checked 2026-11-20. Items marked *unverified* still need confirming.

## S1 — Google Open Buildings 2.5D Temporal

**Verdict:** usable, and no Earth Engine account is needed.

- **What it is:**
  - Rasters derived from Sentinel-2, one per year from 2016 to 2023, each
    dated 06-30.
  - Three float32 bands: `building_fractional_count`; `building_height` (m
    above terrain, clipped to 0–100); and `building_presence` (an
    uncalibrated confidence value). Nodata is −99.
  - Pixels are 0.5 m, but the effective resolution is about 4 m.
  - Google states a height MAE of 1.5 m. That was evaluated only in North
    America, Europe and Japan; accuracy in the Global South, especially for
    tall buildings, is less certain. Marina Bay samples look quantised to
    whole metres (*inferred*).
- **Coverage:** Singapore, Malaysia and Indonesia are listed. Tiles were
  confirmed to contain data for Marina Bay, Johor Bahru, Mersing, Batam and
  Tanjung Pinang. Riau south of the equator (UTM 48S manifests) is
  *unverified*.
- **Access:**
  - The public bucket `gs://open-buildings-temporal-data/v1/` works
    anonymously over HTTPS at `https://storage.googleapis.com/...`.
  - Manifests at `v1/manifests/{S2token}_EPSG_{utm}_{YYYY}_06_30.json` list
    each tile's URI and transform. All of Singapore, Johor and Riau north is in
    `31_EPSG_32648_{year}.json`.
  - Tiles at `v1/geotiffs/{token}_{YYYY}_06_30/tile_*.tif` are 25000² px
    (12.5 km) in UTM 48N. They are deflate-compressed with 512 px blocks and
    internal overviews, so they are COG-like and support windowed reads over
    HTTP (`/vsicurl/`).
- **Volume (one year):** Singapore is about 6.5 GB (20 tiles), Johor about
  19.5 GB, Batam/Bintan about 4.6 GB. **`prepare` must window-read and
  resample to about 4 m** (or read overviews) rather than download whole
  tiles.
- **Licence:** dual CC BY 4.0 / ODbL 1.0. The Sentinel-2 notice applies.
- **Open Buildings v3 polygons:** no height. Singapore, Johor and Batam are
  all in S2 cell `31d` (`31d_buildings.csv.gz`, 981 MB). They are not needed,
  since Overture already merges Google footprints.
- **Sources:**
  - <https://sites.research.google/gr/open-buildings/temporal/>
  - <https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_Research_open-buildings-temporal_v1>

## S2 — Singapore official open data

**Verdict:** useful for footprints and storeys. There are **no open deck
heights, no open LiDAR/DEM, and no downloadable tree inventory**.

| Need | Dataset (data.gov.sg ID) | Notes |
|---|---|---|
| Footprints | HDB Existing Building (`d_16b157c52ed637edd6ba1232e026258d`) | GeoJSON polygons with BLK_NO, ST_COD, POSTAL_COD. Refreshed every 2 months. |
| Storeys | HDB Property Information (`d_17f5382f26140b1fdae0ba2ef6239d2f`) | CSV with `blk_no`, `street`, `max_floor_lvl`. **Join gap:** footprints carry a street *code* and postal code, while this file carries the street *name* and no postal code. The join needs a geocode or a block-number + proximity match. |
| All-building outlines | URA Master Plan 2025 Building Layer (`d_255551acbbce32df735e14d62e067e4f`) | "Indicative" outlines with no height. Lowest building priority. |
| Viaduct rail | URA Master Plan 2025 Rail Line (`d_1c6365f0cad13a77bd79bdbb499131bf`) | LineStrings with `GRND_LEVEL` (ABOVEGROUND/UNDERGROUND). |
| Flyover footprints | LTA Vehicular Bridge/Underpass/Flyover (`d_97268dd70d5634bbed12c7845852ec4b`) | Polygons with STRUCTURE_NAME and TYP_CD. No deck elevation or clearance. |
| Trees | Heritage Trees (about 250 points, no height); TreesSG | TreesSG cannot be downloaded, so **no Singapore trees source**. Canopy comes from Meta/WRI. |
| Terrain | — | National LiDAR/DEM is **not open**. Use FABDEM. |
| 3D models | OneMap3D (view-only; its API needs an NDA); HDB3D (github.com/ualsg/hdb3d-data, LOD1.2, 2019, MIT) | HDB3D is useful as a **validation reference** for HDB heights, not as a source. |

- **data.gov.sg download API:**
  - Call `GET https://api-open.data.gov.sg/v1/public/api/datasets/{id}/poll-download`
    to get a temporary `data.url`. CSVs need `initiate-download` first.
  - Rate limits per 10 s: 2 with no key, 4 with a dev key, 10 with a prod key.
    `prepare` must throttle.
- **Licence (Singapore Open Data Licence v1.0):**
  - Allows commercial use and redistribution.
  - Requires the notice "Contains information from {dataset} accessed on
    {date} from {source} … Singapore Open Data Licence version 1.0". Packs
    record `accessedDate` per source for this.
  - Prohibits implying government endorsement.
- **Sources:**
  - <https://guide.data.gov.sg/developer-guide/dataset-apis/download-dataset>
  - <https://data.gov.sg/open-data-licence>

## Consequences already applied (M2)

- Registry gains `sgHdbBuildings`, `sgUraBuildings`, `sgMasterPlanRail`, and a
  new `elevatedStructures` role with `sgLtaElevatedStructures`.
- Buildings use authoritative-first precedence: HDB, then Overture, then URA.
- `SourceEntry.accessedDate` is filled into `{accessed}` in attribution
  notices.

## M4 access-path findings (verified 2026-11-20)

**Networking**
- **TLS interception:** this network intercepts TLS for most hosts (Google
  Cloud Storage was the exception). Python/certifi, GDAL's bundled curl
  (Schannel on Windows wheels) and Arrow S3 all fail verification by default.
- **Fix:** export the Windows ROOT + CA stores plus certifi to a PEM file
  (`map3d.prepare.certs.exportCaBundle`). It works for `httpx` (via an SSL
  context) and for Arrow (`S3FileSystem(tls_ca_file_path=…)`).
- **GDAL:** its curl/Schannel ignored the bundle in every combination tried
  (`CURL_CA_BUNDLE`, `GDAL_HTTP_SSL_OPTIONS=NATIVE_CA`). So GDAL never makes
  network requests itself: `rasterio.open(url, opener=…)` drives GDAL through
  a Python HTTP range reader (`HttpRangeFile`) instead. rasterio validates an
  opener by sizing a dummy path `test/test`; anything other than
  `AttributeError`/`TypeError`/`ValueError` (e.g. `FileNotFoundError`) passes.

**Rasters**
- **Windowed reads are cheap:** EGM2008, GLO-30 and WorldCover windows cost
  2–4 range requests (~2–4 MB) each.
- **FABDEM:** the Bristol repository serves 10°×10° zips (`N00E100-N10E110` is
  556 MB, 61 members). Reading the central directory takes one request, and
  the `N01E103` member is 14.8 MB.
- **Meta canopy:** zoom-9 quadkey tiles are 1.1 GB, EPSG:3857, uint8,
  65536² px, in row strips with no overviews. Windowed reads still work, but
  cost roughly one strip per output row.
- **Google 2.5D manifests:**
  - `uriPrefix` (`gs://…/v1/geotiffs/31`) is a literal string prefix:
    prefix + uri gives `…/geotiffs/3123c_2023_06_30/tile_*.tif`.
  - The 2023 manifests list 151 files; Singapore/Johor/Riau-north use
    `31_EPSG_32648`, and Riau-south uses `2f_EPSG_32748`.
  - Tile discovery: list the bucket, keep the area's UTM zone(s), then select
    by each tile's affine transform.

**Vectors**
- **Overture `2026-11-18.0`:** buildings have `height`, `num_floors`,
  `is_underground`, `has_parts` and a `bbox` struct (512 files, 128 row groups
  each). Segments have `subtype` (road/rail/water), plus
  `road_flags`/`rail_flags`/`level_rules`/`width_rules` as linear-referenced
  `between` lists.
- **data.gov.sg:**
  - **Formats:** GeoJSONs have plain WGS84 properties (no HTML
    `Description`). The metadata endpoint is not under `/v1/public/api`;
    poll-download needs no key.
  - **LTA `TYP_CD`:** Vehicular Bridge 368, Flyover 189, Vehicular Tunnel 25,
    Vehicular Underpass 21. Tunnels and underpasses are excluded.
  - **Master Plan rail:** 466 ABOVEGROUND vs 931 UNDERGROUND segments.
- **HDB storey join:**
  - Block-number-set matching between `ST_COD` and street names joins 91.9%
    of footprints.
  - Adding the code's two-letter prefix as a tie-breaker (`YIA06H` → YIshun
    Ave) raises that to **98.6%** (634/660 codes).
  - A hard prefix *filter* is worse (89.6%) because Property Information
    abbreviates names (BT = Bukit, etc.).


**Deck heights from the DSM (negative result, 2026-11-21):** the plan to
calibrate deck heights from a Copernicus GLO-30 DSM minus the ground model
fails on real data, for two compounding reasons. (1) FABDEM removes
buildings and forest but **keeps road and bridge decks baked in**, so the
"DTM" along a bridge centerline already sits at deck height and the on-line
difference is near zero. Measuring the ground laterally (25/40 m offsets,
per-point low percentile) restores a signal but shows (2) a 30 m DSM pixel
mixes an 8–20 m deck with the road beneath it: real flyovers read 2–5 m,
not their true 7–14 m clearance, which is indistinguishable from a
mis-tagged bridge over a road lined with trees. Neither height calibration
nor automatic mis-tag dropping is reliable; the `dsmHeight` role is kept as
prepared infrastructure (buildings are large enough for the 30 m pixels to
measure - see `scripts/heightAudit.py`-style validation) and the residual
is reported per bridge by `scripts/roadAudit.py` as human-review evidence.
