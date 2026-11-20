"""Tests for ``map3d.layers.groundFix`` (removing building residue from the ground model)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import rasterio

from map3d.gridGeometry import GridGeometry
from map3d.layers.buildings import loadBuildings
from map3d.layers.groundFix import GroundCorrection
from map3d.layers.terrain import GroundModel
from map3d.mesh import scenery
from map3d.pack.ingest import ingestFiles
from map3d.pack.reader import Pack
from map3d.pack.synthetic import SyntheticScene, makeSyntheticPack
from map3d.request import SceneRequest
from map3d.scene import buildScene

MOUND_m = 20.0  # peak of the residual mound, taller than B1 (12 m) like the depot
B1 = (50.0, 50.0, 150.0, 90.0)  # synthetic building B1 footprint (ENU), 12 m tall


def _plane(e, n):
    return 10.0 + 0.02 * np.asarray(e) - 0.01 * np.asarray(n)


@pytest.fixture(scope="module")
def mound(tmp_path_factory: pytest.TempPathFactory) -> SyntheticScene:
    """Plane terrain with a FABDEM-style residual mound over B1.

    Smoothed residue is low at the walls and peaks mid-footprint: a pyramid,
    20 m at B1's centre, fading to zero 5 m outside its walls.
    """
    root = tmp_path_factory.mktemp("g")
    scene = makeSyntheticPack(root / "mound.pack", terrain="plane")
    src = scene.packDir / "raster" / "syntheticGround" / "ground.tif"
    with rasterio.open(src) as ds:
        profile, data, transform = ds.profile, ds.read(1), ds.transform
    rows, cols = np.mgrid[0 : data.shape[0], 0 : data.shape[1]]
    lon, lat = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
    e, n, _ = scene.frame.geodeticToEnu(np.array(lat), np.array(lon), 0.0)
    halfE, halfN = (B1[2] - B1[0]) / 2 + 5, (B1[3] - B1[1]) / 2 + 5
    pyramid = np.minimum(1 - np.abs(e - 100.0) / halfE, 1 - np.abs(n - 70.0) / halfN)
    data = data + (MOUND_m * np.clip(pyramid, 0, None)).reshape(data.shape).astype(data.dtype)
    out = root / "staging" / "ground.tif"
    out.parent.mkdir()
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(data, 1)
    ingestFiles(scene.packDir, "syntheticGround", [out])
    return scene


def _request(scene: SyntheticScene, heading_deg: float = 0.0, s: float = 2.0) -> SceneRequest:
    h = math.radians(heading_deg)
    axes = [[math.cos(h), -math.sin(h), 0.0], [math.sin(h), math.cos(h), 0.0], [0.0, 0.0, 1.0]]
    return SceneRequest((scene.frame.lat0_deg, scene.frame.lon0_deg), s, axes, 1000.0, 1000.0)


def _correction(scene: SyntheticScene) -> GroundCorrection:
    geom = GridGeometry.fromRequest(_request(scene))
    pack = Pack.open(scene.packDir)
    ground = GroundModel.load(pack, geom)
    buildings = loadBuildings(pack, geom, ground, correctGround=False)
    return GroundCorrection.build(geom, ground, buildings.geometries)


def testBuild_triggersOnlyOnTheMoundedBuildingAndFitsTheTruePlane(mound: SyntheticScene):
    corr = _correction(mound)
    assert corr.buildingIndex.tolist() == [1]  # B1 only
    a, b, c = corr.coeffs[0]
    assert a == pytest.approx(10.0, abs=0.05)
    assert b == pytest.approx(0.02, abs=2e-4) and c == pytest.approx(-0.01, abs=2e-4)
    assert corr.maxLowering_m[0] == pytest.approx(
        MOUND_m, abs=2.5
    )  # 5 m interior sampling straddles the peak


def testCorrectPoints_lowersOnlyInsideTriggeredZones(mound: SyntheticScene):
    corr = _correction(mound)
    e = np.array([100.0, -160.0, 0.0, 100.0])  # B1 centre, B0 centre, open ground, B1 again
    n = np.array([70.0, -100.0, 0.0, 70.0])
    lat, lon, _ = mound.frame.columnPointAtHeight(e, n, 0.0)
    raw = _plane(e, n) + np.array([MOUND_m, 0.0, 0.0, MOUND_m])
    raw[3] = np.nan
    out = corr.correctPoints(lat, lon, raw)
    assert out[0] == pytest.approx(_plane(100.0, 70.0), abs=0.05)
    assert out[1] == raw[1] and out[2] == raw[2]
    assert np.isnan(out[3])  # missing data stays missing


def testScene_groundCorrectedBuildingExposedAndMetadata(mound: SyntheticScene):
    fixed = buildScene(_request(mound, heading_deg=25.0), mound.packDir, chunkSize=97, workers=4)
    raw = buildScene(_request(mound, heading_deg=25.0), mound.packDir, correctGround=False)
    e, n = fixed.geometry.nodesEnu()
    centre = np.unravel_index(np.argmin((e - 100) ** 2 + (n - 70) ** 2), e.shape)
    assert fixed.layers["groundHeight_datum_m"][centre] == pytest.approx(
        _plane(e[centre], n[centre]), abs=0.1
    )
    # The mound is present without the fix (5 m source raster rounds the sharp peak).
    assert raw.layers["groundHeight_datum_m"][centre] > _plane(e[centre], n[centre]) + 0.8 * MOUND_m
    # B1 is 12 m tall above its lowest outline ground (the plane at (50, 90)); on the
    # slope that leaves 12 - (plane(centre) - plane(50, 90)) exposed at its centre.
    exposed = 12.0 - (_plane(e[centre], n[centre]) - _plane(50.0, 90.0))
    assert fixed.layers["buildingHeight_m"][centre] == pytest.approx(exposed, abs=0.3)
    assert raw.layers["buildingHeight_m"][centre] < 1.0
    meta = fixed.metadata["groundCorrection"]
    assert meta["buildingsCorrected"] == 1 and meta["maxLowering_m"] == pytest.approx(
        MOUND_m, abs=2.5
    )
    assert raw.metadata["groundCorrection"]["buildingsCorrected"] == 0
    far = (e - 100) ** 2 + (n - 70) ** 2 > 120.0**2  # well away from B1's zone
    np.testing.assert_array_equal(
        fixed.layers["groundHeight_datum_m"][far], raw.layers["groundHeight_datum_m"][far]
    )


def testMesh_terrainUsesCorrectedGround(mound: SyntheticScene):
    scene = buildScene(_request(mound), mound.packDir)
    part = scenery.terrainPart(
        scene, scenery.TileExtent(50.0, 150.0, 30.0, 110.0), gridSize=33, maxError_m=0.0
    )
    e, n, u = part.positions_ENU_m.T
    inside = (e > 60) & (e < 140) & (n > 55) & (n < 85)
    _, _, uPlane = scene.geometry.frame.columnPointAtHeight(
        e[inside], n[inside], _plane(e[inside], n[inside]) + 8.0
    )
    np.testing.assert_allclose(u[inside], uPlane, atol=0.1)


def testNoBuildings_givesEmptyCorrection(mound: SyntheticScene):
    geom = GridGeometry.fromRequest(_request(mound))
    ground = GroundModel.load(Pack.open(mound.packDir), geom)
    corr = GroundCorrection.build(geom, ground, np.array([], dtype=object))
    assert corr.coeffs.shape == (0, 3)
    vals = np.array([5.0])
    assert corr.correctPoints(np.array([1.3]), np.array([103.8]), vals)[0] == 5.0


def testPixelZoneRoundTrip_gridToEnu(mound: SyntheticScene, tmp_path: Path):
    corr = _correction(mound)
    import shapely

    geom = GridGeometry.fromRequest(_request(mound))
    cx, cy = shapely.centroid(corr.pixelZones[0]).x, shapely.centroid(corr.pixelZones[0]).y
    s = geom.gridSpacing_m
    gx, gy = cx * s - 500.0, cy * s - 500.0
    assert (gx, gy) == pytest.approx((100.0, 70.0), abs=0.5)


def testBuild_degenerateFootprintsAreSkippedSafely(mound: SyntheticScene):
    import shapely

    geom = GridGeometry.fromRequest(_request(mound))
    ground = GroundModel.load(Pack.open(mound.packDir), geom)
    lat, lon, _ = mound.frame.columnPointAtHeight(np.array([0.0, 0.0]), np.array([0.0, 2.0]), 0.0)
    tiny = shapely.box(
        lon[0], lat[0], lon[0] + 1e-5, lat[0] + 1e-5
    )  # ~1 m, below the 5 m sampling step
    sliver = shapely.box(
        lon[0], lat[0], lon[0] + 2e-4, lat[0] + 1e-7
    )  # area but no interior sample hit
    offPack = shapely.box(120.0, 10.0, 120.001, 10.001)  # no ground data at all (ring is NaN)
    corr = GroundCorrection.build(geom, ground, np.array([tiny, sliver, offPack], dtype=object))
    assert corr.coeffs.shape == (0, 3)  # flat open ground: nothing to correct, nothing crashes
