"""Tests for ``map3d.pack.ingest``."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import shapely

from map3d.pack.geoparquet import writeGeoParquet
from map3d.pack.ingest import ingestFiles
from map3d.pack.manifest import Manifest

BOUNDS = (103.6, 1.2, 104.1, 1.5)


def testIngestRaster_createsPackAndCopiesFile(tmp_path: Path, writeRaster):
    src = writeRaster(tmp_path / "in" / "dem.tif", (103.5, 1.1, 104.0, 1.4))
    pack = tmp_path / "sg.pack"
    manifest = ingestFiles(pack, "fabdem", [src], bounds_deg=BOUNDS)

    copied = pack / "raster" / "fabdem" / "dem.tif"
    assert copied.is_file()
    entry = manifest.sources["fabdem"]
    assert entry.role == "ground"
    assert entry.crs is not None and "WGS 84" in entry.crs
    (packFile,) = entry.files
    assert packFile.path == "raster/fabdem/dem.tif"
    assert packFile.sha256 == hashlib.sha256(src.read_bytes()).hexdigest()
    assert packFile.bytes == src.stat().st_size
    assert packFile.bounds_deg == pytest.approx((103.5, 1.1, 104.0, 1.4))
    assert Manifest.load(pack) == manifest


def testIngestVector_copiesIntoVectorDir(tmp_path: Path, writeBuildings):
    src = writeBuildings(tmp_path / "in" / "b.parquet", [(103.8, 1.3, 103.81, 1.31)])
    pack = tmp_path / "sg.pack"
    manifest = ingestFiles(pack, "overtureBuildings", [src], bounds_deg=BOUNDS)
    assert (pack / "vector" / "overtureBuildings" / "b.parquet").is_file()
    entry = manifest.sources["overtureBuildings"]
    assert entry.crs is None
    assert entry.files[0].bounds_deg == pytest.approx((103.8, 1.3, 103.81, 1.31))


def testIngest_intoExistingPackWithoutBounds(tmp_path: Path, writeRaster):
    pack = tmp_path / "sg.pack"
    ingestFiles(pack, "fabdem", [writeRaster(tmp_path / "a.tif", BOUNDS)], bounds_deg=BOUNDS)
    manifest = ingestFiles(pack, "metaCanopy", [writeRaster(tmp_path / "c.tif", BOUNDS)])
    assert set(manifest.sources) == {"fabdem", "metaCanopy"}


def testIngest_sameBoundsOnExistingPackAccepted(tmp_path: Path, writeRaster):
    pack = tmp_path / "sg.pack"
    ingestFiles(pack, "fabdem", [writeRaster(tmp_path / "a.tif", BOUNDS)], bounds_deg=BOUNDS)
    ingestFiles(pack, "metaCanopy", [writeRaster(tmp_path / "c.tif", BOUNDS)], bounds_deg=BOUNDS)


def testIngest_reingestReplacesFileEntry(tmp_path: Path, writeRaster):
    pack = tmp_path / "sg.pack"
    src = tmp_path / "in" / "dem.tif"
    writeRaster(src, BOUNDS)
    ingestFiles(pack, "fabdem", [src], bounds_deg=BOUNDS)
    writeRaster(src, BOUNDS, values=np.ones((8, 8), dtype=np.float32))
    manifest = ingestFiles(pack, "fabdem", [src])
    (packFile,) = manifest.sources["fabdem"].files
    assert packFile.sha256 == hashlib.sha256(src.read_bytes()).hexdigest()


def testIngest_newPackRequiresBounds(tmp_path: Path, writeRaster):
    with pytest.raises(ValueError, match="bounds are required"):
        ingestFiles(tmp_path / "p", "fabdem", [writeRaster(tmp_path / "a.tif", BOUNDS)])


def testIngest_conflictingBoundsRejected(tmp_path: Path, writeRaster):
    pack = tmp_path / "sg.pack"
    ingestFiles(pack, "fabdem", [writeRaster(tmp_path / "a.tif", BOUNDS)], bounds_deg=BOUNDS)
    with pytest.raises(ValueError, match="already covers"):
        ingestFiles(pack, "fabdem", [tmp_path / "a.tif"], bounds_deg=(103.0, 1.0, 104.0, 2.0))


def testIngest_unknownSourceRejected(tmp_path: Path, writeRaster):
    with pytest.raises(ValueError, match="unknown source"):
        ingestFiles(tmp_path / "p", "nope", [writeRaster(tmp_path / "a.tif", BOUNDS)], BOUNDS)


def testIngest_noFilesRejected(tmp_path: Path):
    with pytest.raises(ValueError, match="no files"):
        ingestFiles(tmp_path / "p", "fabdem", [], BOUNDS)


def testIngest_missingFileRejected(tmp_path: Path):
    with pytest.raises(ValueError, match="not found"):
        ingestFiles(tmp_path / "p", "fabdem", [tmp_path / "missing.tif"], BOUNDS)


@pytest.mark.parametrize(
    ("sourceId", "name"), [("fabdem", "a.png"), ("overtureBuildings", "a.csv")]
)
def testIngest_wrongSuffixRejected(tmp_path: Path, sourceId: str, name: str):
    path = tmp_path / name
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="must be"):
        ingestFiles(tmp_path / "p", sourceId, [path], BOUNDS)


def testIngestRaster_withoutCrsRejected(tmp_path: Path, writeRaster):
    src = writeRaster(tmp_path / "a.tif", BOUNDS, crs=None)
    with pytest.raises(ValueError, match="no CRS"):
        ingestFiles(tmp_path / "p", "fabdem", [src], BOUNDS)


def testIngestRaster_mixedCrsRejected(tmp_path: Path, writeRaster):
    pack = tmp_path / "p"
    ingestFiles(pack, "fabdem", [writeRaster(tmp_path / "a.tif", BOUNDS)], BOUNDS)
    utm = writeRaster(
        tmp_path / "b.tif", (360000.0, 130000.0, 370000.0, 140000.0), crs="EPSG:32648"
    )
    with pytest.raises(ValueError, match="CRS"):
        ingestFiles(pack, "fabdem", [utm])


def testIngestRaster_outsidePackRejected(tmp_path: Path, writeRaster):
    src = writeRaster(tmp_path / "a.tif", (10.0, 10.0, 11.0, 11.0))
    with pytest.raises(ValueError, match="does not intersect"):
        ingestFiles(tmp_path / "p", "fabdem", [src], BOUNDS)


def testIngestGeoid_mustBeGeographic(tmp_path: Path, writeRaster):
    utm = writeRaster(
        tmp_path / "g.tif", (360000.0, 130000.0, 370000.0, 140000.0), crs="EPSG:32648"
    )
    with pytest.raises(ValueError, match="geographic"):
        ingestFiles(tmp_path / "p", "egm2008", [utm], BOUNDS)


def testIngestVector_missingColumnsRejected(tmp_path: Path):
    path = tmp_path / "b.parquet"
    writeGeoParquet(
        path, np.array([shapely.box(103.8, 1.3, 103.81, 1.31)]), {"sourceFeatureId": ["a"]}
    )
    with pytest.raises(ValueError, match="missing canonical columns.*height_m"):
        ingestFiles(tmp_path / "p", "overtureBuildings", [path], BOUNDS)


def testIngestVector_emptyFileAccepted(tmp_path: Path):
    path = tmp_path / "b.parquet"
    writeGeoParquet(
        path, np.array([], dtype=object), {"sourceFeatureId": [], "height_m": [], "numFloors": []}
    )
    manifest = ingestFiles(tmp_path / "p", "overtureBuildings", [path], BOUNDS)
    assert manifest.sources["overtureBuildings"].files[0].bounds_deg is None


def testIngest_failedValidationCopiesNothing(tmp_path: Path, writeRaster):
    good = writeRaster(tmp_path / "a.tif", BOUNDS)
    bad = writeRaster(tmp_path / "b.tif", BOUNDS, crs=None)
    pack = tmp_path / "p"
    with pytest.raises(ValueError):
        ingestFiles(pack, "fabdem", [good, bad], BOUNDS)
    assert not (pack / "raster" / "fabdem" / "a.tif").exists()


def testIngest_recordsAccessedDate(tmp_path: Path, writeRaster):
    src = writeRaster(tmp_path / "a.tif", BOUNDS)
    manifest = ingestFiles(tmp_path / "p", "fabdem", [src], BOUNDS, accessedDate="2026-01-02")
    assert manifest.sources["fabdem"].accessedDate == "2026-01-02"


def testIngest_defaultsAccessedDateToToday(tmp_path: Path, writeRaster):
    from datetime import UTC, datetime

    src = writeRaster(tmp_path / "a.tif", BOUNDS)
    manifest = ingestFiles(tmp_path / "p", "fabdem", [src], BOUNDS)
    assert manifest.sources["fabdem"].accessedDate == datetime.now(UTC).date().isoformat()


def testIngest_rejectsMalformedAccessedDate(tmp_path: Path, writeRaster):
    src = writeRaster(tmp_path / "a.tif", BOUNDS)
    with pytest.raises(ValueError, match="accessedDate"):
        ingestFiles(tmp_path / "p", "fabdem", [src], BOUNDS, accessedDate="20/11/2026")
