"""Catalogue of data roles and known sources (SPEC section 8).

A *role* is what the scene builder needs (ground heights, building footprints,
...). A *source* is a concrete dataset that fills a role. Within a role, sources
are listed in priority order: the first one present in a pack wins. For the
``buildings`` role the layer code instead *fuses* every source present, and the
order is the height precedence (authoritative data first, SPEC section 8).

Attribution strings may contain ``{accessed}``, filled with the source's access
date (required by the Singapore Open Data Licence notice).

Vector sources are stored in packs in a **canonical schema** per role, so layer
code never sees source-specific attribute names. Raster sources are stored as
GeoTIFFs; their vertical reference is recorded here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "BUILDING_3D_COLUMNS",
    "ROLES",
    "SOURCES",
    "DataKind",
    "RoleSpec",
    "SourceSpec",
    "VerticalReference",
    "getSource",
    "sourcesForRole",
]

DataKind = Literal["raster", "vector"]
# Optional building columns: bottom of a raised part (height above ground, or
# floors skipped), and roof shape / height / facing bearing / ridge orientation.
BUILDING_3D_COLUMNS = (
    "minHeight_m",
    "minFloor",
    "roofShape",
    "roofHeight_m",
    "roofDirection_deg",
    "roofOrientation",
)
VerticalReference = Literal["EGM2008", "WGS84", "aboveGround"]


@dataclass(frozen=True)
class RoleSpec:
    """What a role holds.

    Attributes:
        role: Role identifier.
        kind: ``"raster"`` or ``"vector"``.
        required: Whether a scene cannot be built without it.
        requiredColumns: Canonical attribute columns (vector roles only).
        optionalColumns: Canonical columns a source may add; layer code uses
            them when present.
    """

    role: str
    kind: DataKind
    required: bool
    requiredColumns: tuple[str, ...] = ()
    optionalColumns: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceSpec:
    """A concrete dataset that fills a role.

    Attributes:
        sourceId: Source identifier (camelCase).
        role: Role it fills.
        kind: ``"raster"`` or ``"vector"``.
        dataset: Dataset name and version.
        licence: Licence name.
        attribution: Required attribution text.
        nativeResolution_m: Nominal resolution (0 for vector data).
        verticalReference: Vertical reference of height values, if any.
        authoritative: Official (government) data; wins the height precedence.
    """

    sourceId: str
    role: str
    kind: DataKind
    dataset: str
    licence: str
    attribution: str
    nativeResolution_m: float
    verticalReference: VerticalReference | None = None
    authoritative: bool = False


ROLES: dict[str, RoleSpec] = {
    spec.role: spec
    for spec in (
        RoleSpec("ground", "raster", True),  # bare-earth terrain height
        RoleSpec("geoid", "raster", True),  # geoid undulation grid (EGM2008)
        # footprints (or building parts) with optional height (m) / floor count,
        # plus optional OSM Simple 3D Buildings attributes (SPEC section 8)
        RoleSpec(
            "buildings",
            "vector",
            False,
            ("sourceFeatureId", "height_m", "numFloors"),
            BUILDING_3D_COLUMNS,
        ),
        RoleSpec("buildingHeightRaster", "raster", False),  # building height above ground
        RoleSpec("canopyHeight", "raster", False),  # vegetation canopy height above ground
        RoleSpec("landCover", "raster", False),  # land-cover class (ESA WorldCover codes)
        # road/rail/footway centerlines with bridge flag, level and width
        RoleSpec(
            "transport",
            "vector",
            False,
            ("sourceFeatureId", "kind", "isBridge", "level", "width_m"),
        ),
        RoleSpec("trees", "vector", False, ("sourceFeatureId", "height_m")),  # points + height
        # flyover/bridge/viaduct footprints (location only; no deck heights)
        RoleSpec("elevatedStructures", "vector", False, ("sourceFeatureId", "structureType")),
        RoleSpec("imagery", "raster", False),  # RGB imagery for the mesh drape
    )
}


def _synthetic(
    sourceId: str,
    role: str,
    kind: DataKind,
    what: str,
    nativeResolution_m: float = 0.0,
    verticalReference: VerticalReference | None = None,
) -> SourceSpec:
    """A ``makeSyntheticPack`` source: test data, lowest priority in its role."""
    return SourceSpec(
        sourceId,
        role,
        kind,
        f"map3d synthetic {what}",
        "synthetic test data",
        "map3d synthetic test data",
        nativeResolution_m,
        verticalReference,
    )


def _sgOdl(datasetName: str) -> tuple[str, str]:
    """Licence and attribution notice for a data.gov.sg dataset (``{accessed}`` filled later)."""
    return (
        "Singapore Open Data Licence v1.0",
        f"Contains information from {datasetName} accessed on {{accessed}} from data.gov.sg, "
        "made available under the terms of the Singapore Open Data Licence version 1.0 "
        "(https://data.gov.sg/open-data-licence)",
    )


_SOURCE_LIST: tuple[SourceSpec, ...] = (
    # --- ground (priority order) ---
    SourceSpec(
        "fabdem",
        "ground",
        "raster",
        "FABDEM V1-2",
        "CC BY-NC-SA 4.0",
        "FABDEM © University of Bristol / Fathom, CC BY-NC-SA 4.0; "
        "derived from Copernicus DEM GLO-30 © DLR e.V. / Airbus Defence and Space",
        30.0,
        "EGM2008",
    ),
    SourceSpec(
        "copernicusGlo30",
        "ground",
        "raster",
        "Copernicus DEM GLO-30",
        "Copernicus DEM licence",
        "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence "
        "and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; "
        "all rights reserved",
        30.0,
        "EGM2008",
    ),
    _synthetic("syntheticGround", "ground", "raster", "terrain", 5.0, "EGM2008"),
    # --- geoid ---
    SourceSpec(
        "egm2008",
        "geoid",
        "raster",
        "EGM2008 2.5' grid (PROJ-data us_nga_egm08_25)",
        "Public domain (NGA)",
        "EGM2008 geoid © U.S. National Geospatial-Intelligence Agency",
        4600.0,
    ),
    _synthetic("syntheticGeoid", "geoid", "raster", "geoid", 100.0),
    # --- buildings (fused; height precedence order) ---
    SourceSpec(
        "sgHdbBuildings",
        "buildings",
        "vector",
        "HDB Existing Building + HDB Property Information (max_floor_lvl)",
        *_sgOdl("HDB Existing Building and HDB Property Information"),
        0.0,
        authoritative=True,
    ),
    SourceSpec(
        "overtureBuildings",
        "buildings",
        "vector",
        "Overture Maps buildings",
        "ODbL 1.0",
        "© Overture Maps Foundation, ODbL (OpenStreetMap contributors, Google, Microsoft)",
        0.0,
    ),
    SourceSpec(
        "sgUraBuildings",
        "buildings",
        "vector",
        "URA Master Plan 2025 Building Layer (indicative outlines, no height)",
        *_sgOdl("Master Plan 2025 Building Layer"),
        0.0,
        authoritative=True,
    ),
    _synthetic("syntheticBuildings", "buildings", "vector", "buildings"),
    # --- building height raster ---
    SourceSpec(
        "googleOpenBuildings25d",
        "buildingHeightRaster",
        "raster",
        "Google Open Buildings 2.5D Temporal",
        "CC BY 4.0 (dual-licensed CC BY 4.0 / ODbL 1.0)",
        "Building heights © Google Open Buildings 2.5D, CC BY 4.0; "
        "contains modified Copernicus Sentinel data",
        4.0,
        "aboveGround",
    ),
    # --- canopy ---
    SourceSpec(
        "metaCanopy",
        "canopyHeight",
        "raster",
        "Meta / WRI global 1 m canopy height",
        "CC BY 4.0",
        "Canopy height © Meta and WRI, CC BY 4.0",
        1.0,
        "aboveGround",
    ),
    _synthetic("syntheticCanopy", "canopyHeight", "raster", "canopy", 5.0, "aboveGround"),
    # --- land cover ---
    SourceSpec(
        "esaWorldCover",
        "landCover",
        "raster",
        "ESA WorldCover 2021 v200",
        "CC BY 4.0",
        "Land cover © ESA WorldCover 2021, CC BY 4.0; contains modified Copernicus Sentinel data",
        10.0,
    ),
    SourceSpec(
        "impactObservatoryLulc",
        "landCover",
        "raster",
        "Impact Observatory / Esri 10 m annual LULC",
        "CC BY 4.0",
        "Land cover © Impact Observatory, Microsoft, Esri, CC BY 4.0; "
        "contains modified Copernicus Sentinel data",
        10.0,
    ),
    _synthetic("syntheticLandCover", "landCover", "raster", "land cover", 5.0),
    # --- transport ---
    SourceSpec(
        "overtureTransportation",
        "transport",
        "vector",
        "Overture Maps transportation segments",
        "ODbL 1.0",
        "© Overture Maps Foundation, ODbL (OpenStreetMap contributors)",
        0.0,
    ),
    SourceSpec(
        "sgMasterPlanRail",
        "transport",
        "vector",
        "URA Master Plan 2025 Rail Line Layer (GRND_LEVEL marks viaducts)",
        *_sgOdl("Master Plan 2025 Rail Line Layer"),
        0.0,
        authoritative=True,
    ),
    _synthetic("syntheticTransport", "transport", "vector", "transport"),
    # --- trees ---
    _synthetic("syntheticTrees", "trees", "vector", "trees"),
    # --- elevated structures ---
    SourceSpec(
        "sgLtaElevatedStructures",
        "elevatedStructures",
        "vector",
        "LTA Vehicular Bridge / Underpass / Flyover",
        *_sgOdl("LTA Vehicular Bridge/Underpass/Flyover"),
        0.0,
        authoritative=True,
    ),
    _synthetic(
        "syntheticElevatedStructures", "elevatedStructures", "vector", "elevated structures"
    ),
    # --- imagery ---
    SourceSpec(
        "sentinel2Composite",
        "imagery",
        "raster",
        "Sentinel-2 cloudless 2024 by EOX (s2maps.eu)",
        "CC BY-NC-SA 4.0",
        "Sentinel-2 cloudless - https://s2maps.eu by EOX IT Services GmbH "
        "(Contains modified Copernicus Sentinel data 2024)",
        10.0,
    ),
)

SOURCES: dict[str, SourceSpec] = {spec.sourceId: spec for spec in _SOURCE_LIST}


def getSource(sourceId: str) -> SourceSpec:
    """Look up a source by identifier.

    Args:
        sourceId: Source identifier.

    Returns:
        The source specification.

    Raises:
        ValueError: If the source is unknown.
    """
    try:
        return SOURCES[sourceId]
    except KeyError:
        known = ", ".join(sorted(SOURCES))
        raise ValueError(f"unknown source {sourceId!r}; known sources: {known}") from None


def sourcesForRole(role: str) -> list[str]:
    """Source identifiers for a role, highest priority first.

    Args:
        role: Role identifier.

    Returns:
        Source identifiers in priority order (catalogue order).
    """
    return [spec.sourceId for spec in _SOURCE_LIST if spec.role == role]
