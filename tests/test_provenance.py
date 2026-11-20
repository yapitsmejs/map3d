"""Tests for ``map3d.layers.provenance``."""

from __future__ import annotations

from map3d.layers.provenance import HeightSource, heightSourceTable


def testHeightSource_codesFitUint8AndAreUnique():
    codes = [member.value for member in HeightSource]
    assert len(codes) == len(set(codes))
    assert all(0 <= c <= 255 for c in codes)
    assert HeightSource.NONE == 0


def testHeightSourceTable_mapsNamesToCodes():
    table = heightSourceTable()
    assert table["GROUND"] == HeightSource.GROUND
    assert len(table) == len(HeightSource)
