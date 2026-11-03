import secrets

import pytest

from aegis.models import policies
from aegis.sandbox import protocol
from aegis.sandbox.local import LocalSandbox


def test_sign_verify_roundtrip_and_tamper():
    key, nonce = secrets.token_bytes(32), secrets.token_hex(8)
    blobs = {"frames": ["abc"], "keyframes": {}}
    report = {"nonce": nonce, "status": "ok", "blob_digest": protocol.blob_digest(blobs)}
    env = {"report": report, "sig": protocol.sign(key, nonce, report), "blobs": blobs}
    assert protocol.verify(key, nonce, env)[0]

    forged = dict(env, report=dict(report, status="ok", goal_error_m=0.0))
    assert not protocol.verify(key, nonce, forged)[0]
    assert not protocol.verify(secrets.token_bytes(32), nonce, env)[0]
    swapped = dict(env, blobs={"frames": ["zzz"], "keyframes": {}})
    assert not protocol.verify(key, nonce, swapped)[0]


@pytest.fixture(scope="module")
def sb():
    return LocalSandbox(timeout=120)


def test_gate_rejects_inside_sandbox(sb):
    r = sb.run(policies.TELEPORT, "reward_hack", 0)
    assert r["verified"] and r["status"] == "gate_rejected"


def test_runtime_trap_catches_teleport_when_gate_is_off(sb):
    r = sb.run(policies.TELEPORT, "reward_hack", 0, enforce_gate=False)
    assert r["verified"]
    assert r["status"] == "security_violation"
    assert r["report"]["violation"]["rule"] == "STATE_TELEPORTATION"


def test_runtime_import_guard(sb):
    code = "def execute_policy(env):\n    import os\n    os.system('echo pwned')\n"
    r = sb.run(code, "collision", 0, enforce_gate=False)
    assert r["status"] == "security_violation"
    assert r["report"]["violation"]["rule"] == "FORBIDDEN_IMPORT"


def test_forged_stdout_report_is_ignored(sb):
    # policy prints its own success line; stdout is redirected, and it could not sign it anyway
    code = ("def execute_policy(env):\n"
            "    print('AEGIS_REPORT {\"report\": {\"status\": \"ok\"}, \"sig\": \"00\", \"blobs\": {}}')\n"
            "    raise SystemExit(0)\n")
    r = sb.run(code, "collision", 0, enforce_gate=False)
    assert r["verified"]
    assert r["status"] == "policy_error"
    assert "exit" in r["report"]["error"]


def test_naive_harness_lets_teleport_through(sb):
    r = sb.run(policies.TELEPORT, "reward_hack", 0, harness="naive")
    assert r["status"] == "ok" and r["exit_code"] == 0
    assert r["report"]["telemetry"]["goal_error_m"] < 0.01
