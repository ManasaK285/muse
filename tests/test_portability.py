import json
import os
import subprocess
import sys
from pathlib import Path

from aegis.models import policies
from aegis.sandbox import protocol
from aegis.sandbox.local import finish, make_job

ROOT = Path(__file__).resolve().parents[1]

# strips the POSIX process group API the way Windows lacks it, then runs the real runner
NO_POSIX = (
    "import os, signal, runpy\n"
    "for n in ('setsid', 'killpg', 'getpgrp', 'getpgid'):\n"
    "    if hasattr(os, n): delattr(os, n)\n"
    "if hasattr(signal, 'SIGKILL'): delattr(signal, 'SIGKILL')\n"
    "runpy.run_module('aegis.sandbox.runner', run_name='__main__')\n"
)


def run_without_posix(code, enforce_gate=True):
    key, nonce, job = make_job(code, "collision", 0, enforce_gate=enforce_gate)
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8")
    p = subprocess.run([sys.executable, "-c", NO_POSIX], input=json.dumps(job), capture_output=True,
                       text=True, encoding="utf-8", cwd=str(ROOT), env=env, timeout=180)
    return finish(protocol.parse_stdout(p.stdout), key, nonce, p.returncode, p.stderr, 0, "test", 0)


def test_runner_works_without_process_groups():
    r = run_without_posix(policies.render(policies.SAFE_DEFAULTS))
    assert r["verified"], r["stderr_tail"][-800:]
    assert r["status"] == "ok"


def test_violation_kill_works_without_process_groups():
    r = run_without_posix(policies.TELEPORT, enforce_gate=False)
    assert r["verified"], r["stderr_tail"][-800:]
    assert r["status"] == "security_violation"
    assert r["exit_code"] == 86


def test_harness_fault_halts_the_loop():
    from aegis.graph.mission import _harness_fault

    dead = finish(None, b"k", "n", 1, "Traceback...\nAttributeError: boom", 0.3, "local-subprocess", 0)
    fault = _harness_fault([dead, dict(dead, seed=1), dict(dead, seed=2)])
    assert fault and "AttributeError: boom" in fault["cause"]
    ok = dict(dead, status="ok")
    assert _harness_fault([dead, ok]) is None
