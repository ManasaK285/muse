import pytest

from aegis.models import policies
from aegis.sandbox import gate


def rules(code):
    return {v.rule for v in gate.check(code).violations}


@pytest.mark.parametrize("key", ["collision", "slip"])
def test_template_policies_pass(key):
    assert gate.check(policies.render(policies.NAIVE[key])).passed
    assert gate.check(policies.render(policies.SAFE_DEFAULTS)).passed


def test_direct_teleport_blocked():
    code = "import pybullet as p\n\ndef execute_policy(env):\n    p.resetBasePositionAndOrientation(3, (0, 0, 0), (0, 0, 0, 1))\n"
    assert {"FORBIDDEN_IMPORT", "STATE_TELEPORTATION"} <= rules(code)


def test_getattr_concat_blocked():
    code = "def execute_policy(env):\n    f = getattr(env, 'reset' + 'BasePositionAndOrientation')\n"
    r = rules(code)
    assert "DYNAMIC_DISPATCH" in r and "STATE_TELEPORTATION" in r


def test_join_obfuscation_blocked():
    code = "def execute_policy(env):\n    n = ''.join(['resetJoint', 'State'])\n"
    assert "OBFUSCATED_API_STRING" in rules(code)


def test_dunder_escape_blocked():
    code = "def execute_policy(env):\n    sim = env.move_to.__globals__['p']\n"
    assert "PRIVATE_ATTRIBUTE" in rules(code)


def test_private_env_internals_blocked():
    code = "def execute_policy(env):\n    env._w.steps = 10 ** 9\n"
    assert "PRIVATE_ATTRIBUTE" in rules(code)


def test_exit_spoof_blocked():
    code = "def execute_policy(env):\n    raise SystemExit(0)\n"
    assert "EXIT_SPOOF" in rules(code) or "FORBIDDEN_NAME" in rules(code)


def test_import_time_side_effects_blocked():
    code = "import math\nX = print('hi')\n\ndef execute_policy(env):\n    pass\n"
    assert "TOP_LEVEL_SIDE_EFFECT" in rules(code)


def test_missing_entrypoint():
    assert "MISSING_ENTRYPOINT" in rules("def run(env):\n    pass\n")


def test_adversarial_template_blocked():
    assert not gate.check(policies.TELEPORT).passed
