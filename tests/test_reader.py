"""Tests for ``map3d.pack.reader``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from map3d.pack.ingest import ingestFiles
from map3d.pack.manifest import MANIFEST_NAME
from map3d.pack.reader import Pack

BOUNDS = (103.0, 1.0, 104.0, 2.0)


def _linear(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    return 100.0 * lon - 50.0 * lat


def _linearRaster(writeRaster, path: Path, bounds, shape=(40, 40), nodata=None) -> Path:
    minLon, minLat, maxLon, maxLat = bounds
    rows, cols = shape
    lon = minLon + (np.arange(cols) + 0.5) * (maxLon - minLon) / cols
    lat = maxLat - (np.arange(rows) + 0.5) * (maxLat - minLat) / rows
    lonMesh, latMesh = np.meshgrid(lon, lat)
    return writeRaster(
        path, bounds, values=_linear(lonMesh, latMesh).astype(np.float64), nodata=nodata
    )


@pytest.fixture
def pack(tmp_path: Path, writeRaster, writeBuildings) -> Pack:
    root = tmp_path / "p.pack"
    # Ground split over two tiles side by side.
    ingestFiles(
        root,
        "fabdem",
        [
            _linearRaster(writeRaster, tmp_path / "w.tif", (103.0, 1.0, 103.5, 2.0), (40, 20)),
            _linearRaster(writeRaster, tmp_path / "e.tif", (103.5, 1.0, 104.0, 2.0), (40, 20)),
        ],
        bounds_deg=BOUNDS,
    )
    ingestFiles(root, "egm2008", [writeRaster(tmp_path / "g.tif", BOUNDS, shape=(4, 4))])
    ingestFiles(
        root,
        "overtureBuildings",
        [
            writeBuildings(
                tmp_path / "b.parquet",
                [(103.10, 1.10, 103.11, 1.11), (103.60, 1.60, 103.61, 1.61)],
                heights=[10.0, 20.0],
            )
        ],
    )
    return Pack.open(root)


def testOpen_exposesManifest(pack: Pack):
    assert pack.bounds_deg == BOUNDS
    assert set(pack.manifest.sources) == {"fabdem", "egm2008", "overtureBuildings"}


def testOpen_toleratesRetiredSources(pack: Pack, tmp_path: Path, writeRaster):
    """A pack made before a source left the registry (copernicusDsm) still works."""
    data = json.loads((pack.root / MANIFEST_NAME).read_text(encoding="utf-8"))
    retired = dict(data["sources"]["fabdem"], sourceId="copernicusDsm", role="dsmHeight")
    data["sources"]["copernicusDsm"] = retired
    (pack.root / MANIFEST_NAME).write_text(json.dumps(data), encoding="utf-8")
    old = Pack.open(pack.root)
    assert "copernicusDsm" in old.manifest.sources
    assert old.verify() == []
    assert not any("dsmHeight" in w for w in old.checkCoverage((103.2, 1.2, 103.8, 1.8)))
    ingestFiles(pack.root, "copernicusGlo30", [writeRaster(tmp_path / "c.tif", BOUNDS)])
    assert "copernicusDsm" in Pack.open(pack.root).manifest.sources


def testPrimarySource_followsRolePriority(pack: Pack, tmp_path: Path, writeRaster):
    assert pack.primarySource("ground").sourceId == "fabdem"
    ingestFiles(pack.root, "copernicusGlo30", [writeRaster(tmp_path / "c.tif", BOUNDS)])
    assert Pack.open(pack.root).primarySource("ground").sourceId == "fabdem"
    assert pack.primarySource("canopyHeight") is None


def testReadRaster_mosaicsTilesAndSamplesBilinear(pack: Pack):
    window = pack.readRaster("fabdem", (103.3, 1.3, 103.7, 1.7))
    rng = np.random.default_rng(1)
    lon = rng.uniform(103.3, 103.7, 300)
    lat = rng.uniform(1.3, 1.7, 300)
    np.testing.assert_allclose(window.sample(lat, lon), _linear(lon, lat), atol=1e-9)


def testRasterWindow_nearestSampling(pack: Pack):
    window = pack.readRaster("fabdem", (103.3, 1.3, 103.7, 1.7))
    # Pixel size 0.025 deg; center of a pixel -> exact value, small offsets snap to it.
    lon, lat = np.array([103.4125, 103.4125 + 0.01]), np.array([1.4875, 1.4875 - 0.01])
    np.testing.assert_allclose(
        window.sample(lat, lon, method="nearest"), _linear(lon[:1], lat[:1]).repeat(2)
    )


def testRasterWindow_outsideWindowIsNan(pack: Pack):
    window = pack.readRaster("fabdem", (103.3, 1.3, 103.4, 1.4))
    assert np.isnan(window.sample(np.array([1.9]), np.array([103.9]))).all()
    assert np.isnan(window.sample(np.array([1.9]), np.array([103.9]), method="nearest")).all()


def testRasterWindow_rejectsUnknownMethod(pack: Pack):
    window = pack.readRaster("fabdem", (103.3, 1.3, 103.4, 1.4))
    with pytest.raises(ValueError, match="method"):
        window.sample(np.array([1.35]), np.array([103.35]), method="cubic")


def testReadRaster_nodataBecomesNan(tmp_path: Path, writeRaster):
    values = np.full((8, 8), 5.0, dtype=np.float32)
    values[:4, :] = -99.0
    root = tmp_path / "n.pack"
    ingestFiles(
        root,
        "metaCanopy",
        [writeRaster(tmp_path / "c.tif", BOUNDS, values=values, nodata=-99.0)],
        BOUNDS,
    )
    window = Pack.open(root).readRaster("metaCanopy", BOUNDS, marginPixels=0)
    assert window.values.shape == (8, 8)
    assert np.isnan(window.values).sum() == 32
    assert np.nanmax(window.values) == 5.0


def testReadRaster_projectedSource(tmp_path: Path, writeRaster):
    # A UTM 48N raster around Singapore, value = easting / 1000.
    rows, cols = 20, 20
    left, bottom, right, top = 360000.0, 140000.0, 380000.0, 160000.0
    east = left + (np.arange(cols) + 0.5) * (right - left) / cols
    values = np.tile(east / 1000.0, (rows, 1))
    root = tmp_path / "u.pack"
    src = writeRaster(
        tmp_path / "u.tif", (left, bottom, right, top), values=values, crs="EPSG:32648"
    )
    ingestFiles(root, "metaCanopy", [src], bounds_deg=(103.5, 1.0, 104.5, 2.0))
    window = Pack.open(root).readRaster("metaCanopy", (103.8, 1.3, 103.9, 1.4))
    import pyproj

    toUtm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
    lon, lat = np.array([103.82, 103.88]), np.array([1.33, 1.37])
    e, _ = toUtm.transform(lon, lat)
    np.testing.assert_allclose(window.sample(lat, lon), e / 1000.0, atol=1e-6)


def testReadRaster_unknownOrDisjoint(pack: Pack):
    with pytest.raises(ValueError, match="not in pack"):
        pack.readRaster("metaCanopy", BOUNDS)
    with pytest.raises(ValueError, match="no files"):
        pack.readRaster("fabdem", (110.0, 1.0, 111.0, 2.0))


def testReadVectors_selectsIntersectingFeatures(pack: Pack):
    selection = pack.readVectors("overtureBuildings", (103.05, 1.05, 103.2, 1.2))
    assert len(selection) == 1
    assert selection.table.column("height_m").to_pylist() == [10.0]
    assert selection.geometries.shape == (1,)


def testReadVectors_emptySelectionKeepsSchema(pack: Pack):
    selection = pack.readVectors(
        "overtureBuildings", (103.8, 1.8, 103.9, 1.9), columns=["height_m"]
    )
    assert len(selection) == 0
    assert selection.table.column_names == ["height_m"]


def testReadVectors_rejectsRasterSource(pack: Pack):
    with pytest.raises(ValueError, match="not a vector"):
        pack.readVectors("fabdem", BOUNDS)


def testReadRaster_rejectsVectorSource(pack: Pack):
    with pytest.raises(ValueError, match="not a raster"):
        pack.readRaster("overtureBuildings", BOUNDS)


def testGeoid_loadsGrid(pack: Pack):
    assert float(pack.geoid().undulation_m(1.5, 103.5)) == pytest.approx(0.0)


def testCheckCoverage_warnsForMissingOptionalRoles(pack: Pack):
    warnings = pack.checkCoverage((103.2, 1.2, 103.8, 1.8))
    assert any("canopyHeight" in w for w in warnings)
    assert not any("buildings" in w and "no source" in w for w in warnings)


def testCheckCoverage_warnsForPartialOptionalCoverage(tmp_path: Path, pack: Pack, writeRaster):
    ingestFiles(
        pack.root, "metaCanopy", [writeRaster(tmp_path / "c.tif", (103.0, 1.0, 103.4, 1.4))]
    )
    warnings = Pack.open(pack.root).checkCoverage((103.2, 1.2, 103.8, 1.8))
    assert any("metaCanopy" in w and "partially" in w for w in warnings)


def testCheckCoverage_outsidePackRaises(pack: Pack):
    with pytest.raises(ValueError, match="outside the pack"):
        pack.checkCoverage((103.5, 1.5, 104.5, 2.5))


def testCheckCoverage_missingRequiredRoleRaises(tmp_path: Path, writeRaster):
    root = tmp_path / "x.pack"
    ingestFiles(root, "fabdem", [writeRaster(tmp_path / "a.tif", BOUNDS)], BOUNDS)
    with pytest.raises(ValueError, match="required role 'geoid'"):
        Pack.open(root).checkCoverage((103.2, 1.2, 103.8, 1.8))


def testCheckCoverage_requiredRoleNotCoveringRaises(tmp_path: Path, writeRaster):
    root = tmp_path / "x.pack"
    ingestFiles(root, "fabdem", [writeRaster(tmp_path / "a.tif", (103.0, 1.0, 103.5, 2.0))], BOUNDS)
    ingestFiles(root, "egm2008", [writeRaster(tmp_path / "g.tif", BOUNDS)])
    with pytest.raises(ValueError, match="does not cover"):
        Pack.open(root).checkCoverage((103.2, 1.2, 103.8, 1.8))


def testVerify_detectsTamperingAndMissingFiles(pack: Pack):
    assert pack.verify() == []
    (pack.root / "raster" / "fabdem" / "w.tif").write_bytes(b"tampered")
    (pack.root / "raster" / "fabdem" / "e.tif").unlink()
    problems = pack.verify()
    assert any("checksum" in p and "w.tif" in p for p in problems)
    assert any("missing" in p and "e.tif" in p for p in problems)
