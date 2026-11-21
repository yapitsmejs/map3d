"""Tests for ``map3d.request.SceneRequest`` validation (SPEC section 4.1)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from _support import headingAxes
from hypothesis import given
from hypothesis import strategies as st

from map3d.request import SceneRequest

IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _make(**overrides: object) -> SceneRequest:
    kwargs: dict[str, object] = dict(
        sceneCenter_deg=(1.2834, 103.8607),
        gridSpacing_m=0.5,
        gridAxes_ENU=IDENTITY,
        swathHorizontal_m=2000.0,
        swathVertical_m=1000.0,
    )
    kwargs.update(overrides)
    return SceneRequest(**kwargs)  # type: ignore[arg-type]


def testSceneRequest_validDefaults():
    req = _make()
    assert req.verticalDatum == "msl"
    assert req.maxDeckLevels == 5
    assert req.nCols == 4000
    assert req.nRows == 2000
    assert req.gridAxes_ENU.shape == (3, 3)
    assert req.gridAxes_ENU.dtype == np.float64


def testSceneRequest_axesAreImmutable():
    req = _make()
    with pytest.raises(ValueError):
        req.gridAxes_ENU[0, 0] = 2.0


def testSceneRequest_acceptsSpecExampleHeading():
    axes = [[0.8660254, -0.5, 0.0], [0.5, 0.8660254, 0.0], [0.0, 0.0, 1.0]]
    req = _make(gridAxes_ENU=axes, axesTol=1e-7)
    assert req.nCols == 4000


def testSceneRequest_ellipsoidDatumAndZeroDecksAllowed():
    req = _make(verticalDatum="ellipsoid", maxDeckLevels=0)
    assert req.verticalDatum == "ellipsoid"
    assert req.maxDeckLevels == 0


def testSceneRequest_nonIntegerRatioWithinRoundoffAccepted():
    # 0.1 is not exactly representable; 300 / 0.1 = 2999.9999999999995
    assert _make(gridSpacing_m=0.1, swathHorizontal_m=300.0, swathVertical_m=0.3).nCols == 3000


@given(heading_deg=st.floats(-360.0, 360.0))
def testSceneRequest_anyHeadingAccepted(heading_deg: float):
    assert _make(gridAxes_ENU=headingAxes(heading_deg)).nRows == 2000


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"sceneCenter_deg": (90.0, 0.0)}, "latitude"),
        ({"sceneCenter_deg": (-91.0, 0.0)}, "latitude"),
        ({"sceneCenter_deg": (0.0, 180.5)}, "longitude"),
        ({"sceneCenter_deg": (math.nan, 0.0)}, "latitude"),
        ({"sceneCenter_deg": (1.0, 2.0, 3.0)}, "sceneCenter_deg"),
        ({"gridSpacing_m": 0.0}, "gridSpacing_m"),
        ({"gridSpacing_m": -1.0}, "gridSpacing_m"),
        ({"gridSpacing_m": math.inf}, "gridSpacing_m"),
        ({"swathHorizontal_m": 0.0}, "swathHorizontal_m"),
        ({"swathVertical_m": math.nan}, "swathVertical_m"),
        ({"swathHorizontal_m": 1000.25}, "whole multiple"),
        ({"swathVertical_m": 0.25}, "whole multiple"),
        ({"gridSpacing_m": 0.7, "swathHorizontal_m": 5000.0}, "whole multiple"),
        ({"verticalDatum": "shd"}, "verticalDatum"),
        ({"maxDeckLevels": -1}, "maxDeckLevels"),
        ({"maxDeckLevels": 2.5}, "maxDeckLevels"),
        ({"maxDeckLevels": True}, "maxDeckLevels"),
        ({"axesTol": 0.0}, "axesTol"),
        ({"gridAxes_ENU": [[1.0, 0.0], [0.0, 1.0]]}, "3x3"),
        ({"gridAxes_ENU": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, math.nan]]}, "finite"),
        ({"gridAxes_ENU": [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}, "unit length"),
        (
            {"gridAxes_ENU": [[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]]},
            "orthogonal",
        ),
        (
            {"gridAxes_ENU": [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]},
            "right-handed",
        ),
        (
            {"gridAxes_ENU": [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]},
            "Up",
        ),
    ],
)
def testSceneRequest_rejectsInvalid(overrides: dict[str, object], message: str):
    with pytest.raises(ValueError, match=message):
        _make(**overrides)


def testSceneRequest_rejectsSlightTiltBeyondTolerance():
    tilt_rad = 1e-4
    axes = [
        [1.0, 0.0, 0.0],
        [0.0, math.cos(tilt_rad), math.sin(tilt_rad)],
        [0.0, -math.sin(tilt_rad), math.cos(tilt_rad)],
    ]
    with pytest.raises(ValueError, match="Up"):
        _make(gridAxes_ENU=axes)
