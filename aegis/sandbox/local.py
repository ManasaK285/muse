import json
import os
import secrets
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aegis.sandbox import protocol

ROOT = Path(__file__).resolve().parents[2]


def stderr_summary(stderr):
    # last traceback line is usually the one that matters
    lines = [ln for ln in (stderr or "").splitlines() if ln.strip() and "pybullet build time" not in ln]
    return lines[-1].strip()[:300] if lines else ""


def finish(envelope, key, nonce, exit_code, stderr, elapsed, backend, seed=None):
    if envelope is None:
        why = stderr_summary(stderr)
        return {
            "verified": False, "seed": seed,
            "verify_msg": f"no signed report: {why}" if why else "no signed report (process died or spoofed exit)",
            "status": "no_report", "exit_code": exit_code, "report": {"seed": seed}, "frames": [],
            "keyframes": {}, "stderr_tail": stderr[-3000:], "elapsed_s": elapsed, "backend": backend,
        }
    ok, msg = protocol.verify(key, nonce, envelope)
    report = envelope.get("report", {})
    if report.get("status") == "harness_error":
        stderr = (report.get("error") or "") + "\n" + (stderr or "")
    return {
        "verified": ok, "verify_msg": msg, "seed": seed,
        "status": report.get("status") if ok else "unverified",
        "exit_code": exit_code, "report": report,
        "frames": envelope.get("blobs", {}).get("frames", []),
        "keyframes": envelope.get("blobs", {}).get("keyframes", {}),
        "stderr_tail": (stderr or "")[-3000:], "elapsed_s": elapsed, "backend": backend,
    }


def _spawn_kwargs():
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _kill_tree(proc):
    if os.name != "nt" and hasattr(os, "killpg"):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError):
            pass
    proc.kill()


def make_job(code, scenario, seed, enforce_gate=True, scene_graph=None, max_sim_seconds=20.0,
             harness="aegis"):
    key = secrets.token_bytes(32)
    nonce = secrets.token_hex(16)
    job = {"key": key.hex(), "nonce": nonce, "code": code, "scenario": scenario, "seed": seed,
           "enforce_gate": enforce_gate, "scene_graph": scene_graph or {},
           "max_sim_seconds": max_sim_seconds, "harness": harness}
    return key, nonce, job


class LocalSandbox:
    name = "local-subprocess"

    def __init__(self, timeout=180):
        self.timeout = timeout

    def run(self, code, scenario, seed, enforce_gate=True, scene_graph=None, harness="aegis"):
        key, nonce, job = make_job(code, scenario, seed, enforce_gate, scene_graph, harness=harness)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONIOENCODING"] = "utf-8"
        t0 = time.time()
        proc = subprocess.Popen([sys.executable, "-m", "aegis.sandbox.runner"], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(ROOT), env=env,
                                encoding="utf-8", errors="replace", **_spawn_kwargs())
        try:
            out, err = proc.communicate(json.dumps(job), timeout=self.timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            out, err = proc.communicate()
            res = finish(None, key, nonce, proc.returncode, err or "", round(time.time() - t0, 2),
                         self.name, seed)
            res["status"] = "timeout"
            return res
        return finish(protocol.parse_stdout(out), key, nonce, proc.returncode, err or "",
                      round(time.time() - t0, 2), self.name, seed)

    def run_parallel(self, code, scenario, seeds=(0, 1, 2), enforce_gate=True, scene_graph=None):
        with ThreadPoolExecutor(max_workers=len(seeds)) as ex:
            futs = [ex.submit(self.run, code, scenario, s, enforce_gate, scene_graph) for s in seeds]
            return [f.result() for f in futs]


def get_sandbox(kind=None):
    kind = (kind or os.environ.get("AEGIS_SANDBOX", "local")).lower()
    if kind == "modal":
        from aegis.sandbox.modal_backend import ModalSandbox
        return ModalSandbox()
    return LocalSandbox()
