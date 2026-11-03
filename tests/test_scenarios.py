"""Checks for every scenario in aegis/scenarios.py."""

import math

import pytest

from aegis.scenarios import SCENARIOS, goal_position


@pytest.mark.parametrize("key", sorted(SCENARIOS))
def test_goal_inside_tray(key):
    sc = SCENARIOS[key]
    gx, gy, _gz = goal_position(sc)
    tx, ty = sc.tray.xy
    half = sc.tray.half_extent
    assert abs(gx - tx) <= half, f"{key}: goal x={gx} outside tray x={tx} half={half}"
    assert abs(gy - ty) <= half, f"{key}: goal y={gy} outside tray y={ty} half={half}"


@pytest.mark.parametrize("key", sorted(SCENARIOS))
def test_obstacles_outside_tray(key):
    sc = SCENARIOS[key]
    tx, ty = sc.tray.xy
    half = sc.tray.half_extent
    for obs in sc.obstacles:
        ox, oy = obs.xy
        chebyshev = max(abs(ox - tx), abs(oy - ty))
        assert chebyshev > half, (
            f"{key}: obstacle {obs.name} at {(ox, oy)} inside tray at {(tx, ty)} half={half}"
        )


@pytest.mark.parametrize("key", sorted(SCENARIOS))
def test_payload_far_from_tray_center(key):
    sc = SCENARIOS[key]
    dist = math.dist(sc.payload.xy, sc.tray.xy)
    assert dist >= 0.1, f"{key}: payload {dist=} < 0.1 m from tray center"


@pytest.mark.parametrize("key", sorted(SCENARIOS))
def test_three_randomization_seeds(key):
    sc = SCENARIOS[key]
    assert len(sc.randomization) == 3, f"{key}: {len(sc.randomization)=} != 3"
