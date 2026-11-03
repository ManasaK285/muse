# sandbox entrypoint: one JSON job on stdin, one signed report line on stdout
import base64
import builtins
import hashlib
import json
import os
import signal
import struct
import sys
import traceback

import numpy as np
import pybullet

from aegis import scenarios
from aegis.sandbox import gate, protocol
from aegis.sim.env import BudgetExceeded, SafePandaEnv
from aegis.sim.world import PandaWorld, jpeg_bytes

POLICY_IMPORTS = {"math", "numpy", "typing"}
BLOCKED_EVENTS = (
    "open", "exec", "compile", "subprocess.", "os.system", "os.exec", "os.posix_spawn", "os.spawn",
    "os.fork", "os.kill", "os.remove", "os.rename", "os.chmod", "ctypes.", "socket.", "shutil.",
    "sys._getframe", "sys.settrace", "sys.setprofile", "code.__new__", "function.__new__",
    "builtins.input", "urllib.", "http.", "pickle.", "marshal.loads",
)


class _Guard:
    active = False


def _isolate():
    # POSIX: own process group so a violation can take down anything the policy spawned.
    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            pass  # already a session leader when launched with start_new_session


def _hard_stop(code):
    if hasattr(os, "killpg") and hasattr(signal, "SIGKILL"):
        try:
            os.killpg(os.getpgrp(), signal.SIGKILL)
        except OSError:
            pass
    os._exit(code)  # Windows path; subprocess spawning is already blocked by the audit hook


def _keyframe(snap, world):
    seg_ids = world.seg_ids()
    return {
        "arrays": protocol.pack_array_bundle(rgb=snap["rgb"], depth=snap["depth"], seg=snap["seg"]),
        "t": snap["t"], "ee": list(snap["ee"]), "grip_closed": snap["grip_closed"],
        "poses": {k: list(v) for k, v in snap["poses"].items()},
        "view": list(world.view), "proj": list(world.proj),
        "seg_ids": seg_ids, "labels": world.labels,
    }


def main():
    _isolate()
    job = json.loads(sys.stdin.read())
    key = bytes.fromhex(job.pop("key"))
    nonce = job["nonce"]
    sc = scenarios.get(job["scenario"])
    seed = int(job.get("seed", 0))

    # keep a private handle on real stdout, then send everything else to stderr
    out = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    jpeg_bytes(np.zeros((8, 8, 3), np.uint8))  # warm PIL plugins so they never import mid-policy

    chain = {"h": hashlib.sha256(nonce.encode()).digest(), "steps": 0}
    world_ref = {}
    finished = {"done": False}

    def emit(status, extra=None, world=None, keyframes=None):
        if finished["done"]:
            return
        finished["done"] = True
        blobs = {"frames": [], "keyframes": keyframes or {}}
        telemetry = None
        if world is not None:
            blobs["frames"] = [base64.b64encode(f).decode() for f in world.frames]
            telemetry = world.telemetry()
        report = {
            "nonce": nonce, "status": status, "scenario": sc.key, "seed": seed,
            "physics_steps": chain["steps"], "chain_head": chain["h"].hex(),
            "telemetry": telemetry, "blob_digest": protocol.blob_digest(blobs),
        }
        report.update(extra or {})
        envelope = {"report": report, "sig": protocol.sign(key, nonce, report), "blobs": blobs}
        out.write(protocol.MARKER + json.dumps(envelope) + "\n")
        out.flush()

    def violation(rule, detail):
        _Guard.active = False
        w = world_ref.get("w")
        kf = {}
        if w is not None:
            try:
                kf["final"] = _keyframe(w.snapshot("violation"), w)
            except Exception:
                pass
        emit("security_violation",
             {"violation": {"rule": rule, "detail": detail,
                            "message": f"SECURITY_VIOLATION: {rule} ({detail})"}},
             world=w, keyframes=kf)
        _hard_stop(86)

    try:
        enforce = job.get("enforce_gate", True) and job.get("harness") != "naive"
        report = gate.check(job["code"])
        if enforce and not report.passed:
            emit("gate_rejected", {"gate": report.to_dict(),
                                   "violation": {"rule": report.violations[0].rule,
                                                 "detail": report.summary(),
                                                 "message": "SECURITY_VIOLATION: AST integrity gate"}})
            return

        world = PandaWorld(sc, seed=seed, frame_every=int(job.get("frame_every", 16)))
        world_ref["w"] = world

        def chain_hook(w):
            # the physics loop is the only thing that advances this hash chain
            pos = w.pose(sc.payload.name)[0]
            chain["h"] = hashlib.sha256(chain["h"] + struct.pack("<I3d", w.steps, *pos)).digest()
            chain["steps"] = w.steps

        world.step_hooks.append(chain_hook)
        keyframes = {"initial": _keyframe(world.snapshot("initial"), world)}

        def trap(name, fn):
            def wrapped(*a, **k):
                if _Guard.active:
                    sys.audit("aegis.teleport", name)
                return fn(*a, **k)
            return wrapped

        naive = job.get("harness") == "naive"

        for name in gate.TELEPORT_API:
            if not naive and hasattr(pybullet, name):
                setattr(pybullet, name, trap(name, getattr(pybullet, name)))

        def audit(event, args):
            if not _Guard.active:
                return
            if event == "aegis.teleport":
                rule = "STATE_TELEPORTATION" if str(args[0]).startswith("reset") else "SIMULATOR_STATE_WRITE"
            violation(rule, f"pybullet.{args[0]} called during policy rollout")
            if event == "import":
                root = str(args[0]).split(".")[0]
                if root not in POLICY_IMPORTS:
                    violation("FORBIDDEN_IMPORT", f"runtime import of {args[0]!r}")
                return
            if event.startswith(BLOCKED_EVENTS):
                violation("SANDBOX_ESCAPE", f"audit event {event}")

        if not naive:
            sys.addaudithook(audit)

        real_import = builtins.__import__

        def guarded_import(name, *a, **k):
            if name.split(".")[0] not in POLICY_IMPORTS:
                violation("FORBIDDEN_IMPORT", f"import {name}")
            return real_import(name, *a, **k)

        safe_builtins = dict(vars(builtins))
        for banned in ("open", "exec", "eval", "compile", "input", "breakpoint", "exit", "quit"):
            safe_builtins.pop(banned, None)
        if not naive:
            safe_builtins["__import__"] = guarded_import
        else:
            safe_builtins = vars(builtins)  # the naive harness trusts the policy completely

        env = SafePandaEnv(world, scene_graph=job.get("scene_graph"),
                           max_sim_seconds=float(job.get("max_sim_seconds", 20.0)))
        ns = {"__builtins__": safe_builtins, "__name__": "policy"}
        status, extra = "ok", {}
        try:
            code_obj = builtins.compile(job["code"], "<muse_code_policy>", "exec")
            builtins.exec(code_obj, ns)
            fn = ns.get("execute_policy")
            if not callable(fn):
                raise RuntimeError("execute_policy not defined")
            _Guard.active = True
            fn(env)
        except BudgetExceeded as e:
            status, extra = "policy_error", {"error": str(e)}
        except BaseException as e:  # noqa: BLE001 policy code can raise anything
            if isinstance(e, SystemExit):
                status, extra = "policy_error", {"error": f"policy tried to exit with code {e.code}"}
            else:
                status = "policy_error"
                extra = {"error": "".join(traceback.format_exception_only(type(e), e)).strip()[:400]}
        finally:
            _Guard.active = False

        # hold still for a beat so a grasp that is about to fail actually fails
        try:
            env.wait(0.75)
        except BudgetExceeded:
            pass

        keyframes["final"] = _keyframe(world.snapshot("final"), world)
        for i, snap in enumerate(world.carry_frames[:10]):
            keyframes[f"carry_{i:02d}"] = _keyframe(snap, world)
        extra["gate"] = report.to_dict()
        emit(status, extra, world=world, keyframes=keyframes)
        world.close()

    except Exception:
        emit("harness_error", {"error": traceback.format_exc()[-2000:]},
             world=world_ref.get("w"))


if __name__ == "__main__":
    main()
