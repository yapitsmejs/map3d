"""Tests for ``map3d.pack.registry`` (the source catalogue)."""

from __future__ import annotations

import pytest

from map3d.pack import registry


def testSources_haveConsistentRoleAndKind():
    for sourceId, spec in registry.SOURCES.items():
        assert spec.sourceId == sourceId
        assert spec.role in registry.ROLES
        assert spec.kind == registry.ROLES[spec.role].kind
        assert spec.licence
        assert spec.attribution
        assert spec.nativeResolution_m >= 0


def testRolePriority_listsEveryRoleSourceExactlyOnce():
    for role in registry.ROLES:
        ids = registry.sourcesForRole(role)
        assert len(ids) == len(set(ids))
        assert set(ids) == {s for s, spec in registry.SOURCES.items() if spec.role == role}


def testRolePriority_fabdemBeatsGlo30():
    ground = registry.sourcesForRole("ground")
    assert ground.index("fabdem") < ground.index("copernicusGlo30")


def testGetSource_returnsSpec():
    assert registry.getSource("overtureBuildings").role == "buildings"


def testGetSource_unknownRaises():
    with pytest.raises(ValueError, match="unknown source 'nope'"):
        registry.getSource("nope")


def testRequiredRoles_onlyGroundAndGeoid():
    required = {r for r, spec in registry.ROLES.items() if spec.required}
    assert required == {"ground", "geoid"}


def testVectorRoles_declareRequiredColumns():
    for spec in registry.ROLES.values():
        if spec.kind == "vector":
            assert "sourceFeatureId" in spec.requiredColumns
        else:
            assert spec.requiredColumns == ()


def testHeightRasters_declareVerticalReference():
    assert registry.getSource("fabdem").verticalReference == "EGM2008"
    assert registry.getSource("metaCanopy").verticalReference == "aboveGround"
    assert registry.getSource("esaWorldCover").verticalReference is None


def testSingaporeSources_registeredWithOpenDataLicence():
    for sourceId in (
        "sgHdbBuildings",
        "sgUraBuildings",
        "sgMasterPlanRail",
        "sgLtaElevatedStructures",
    ):
        spec = registry.getSource(sourceId)
        assert spec.licence == "Singapore Open Data Licence v1.0"
        assert "{accessed}" in spec.attribution


def testBuildingSources_authoritativeFirst():
    buildings = registry.sourcesForRole("buildings")
    assert buildings.index("sgHdbBuildings") < buildings.index("overtureBuildings")
    assert buildings.index("overtureBuildings") < buildings.index("sgUraBuildings")


def testElevatedStructuresRole_isOptionalVector():
    spec = registry.ROLES["elevatedStructures"]
    assert spec.kind == "vector" and not spec.required
    assert spec.requiredColumns == ("sourceFeatureId", "structureType")
