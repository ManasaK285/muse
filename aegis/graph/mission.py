import difflib
import logging
import platform
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from aegis import redteam as red_team
from aegis import scenarios
from aegis.config import settings
from aegis.models import policies
from aegis.models.muse_code import MuseCode
from aegis.models.muse_image import GoalFrameRenderer
from aegis.models.sam import FallbackSegmenter
from aegis.models.spark import Spark, classify
from aegis.models.wonder import WorldModel
from aegis.perception import auditor, scene_graph, viz
from aegis.perception.frames import from_snapshot
from aegis.sandbox import gate
from aegis.sandbox.local import get_sandbox
from aegis.sim.world import PandaWorld

log = logging.getLogger("aegis.mission")


class MissionState(TypedDict, total=False):
    scenario: str
    instruction: str
    chaos: bool
    iteration: int
    max_iters: int
    scene_graph: dict
    plan: dict
    code: str
    params: Optional[dict]
    code_history: list
    expect: Any
    forecast: Any
    redteam: bool
    redteam_result: dict
    rollouts: list
    audit: dict
    diagnosis: dict
    verdict: str
    recovery_cycles: int
    naive: dict
    initial_frame: Any
    harness_error: dict


class Toolkit:
    def __init__(self, emit: Callable[[str, dict], None]):
        self.emit = emit
        self.sandbox = get_sandbox(settings.sandbox)
        self.segmenter = FallbackSegmenter()
        self.spark = Spark()
        self.coder = MuseCode()
        self.wonder = WorldModel()
        self.imager = GoalFrameRenderer()
        self.pool = ThreadPoolExecutor(max_workers=4)

    def backends(self):
        return {"perception": self.segmenter.name, "planner": self.spark.name, "coder": self.coder.name,
                "world_model": self.wonder.name, "goal_frame": self.imager.name, "sandbox": self.sandbox.name}

    def model_stack(self):
        from aegis.config import settings as cfg
        from aegis.redteam import Attacker

        live = cfg.backend("voice") == "live"
        code_state = {"cli": "live", "api": "via Muse Spark API", "offline": "offline twin"}[self.coder.mode]
        return [
            {"model": "SAM 3.1", "role": "masks, 3D scene graph",
             "state": "live" if self.segmenter.short == "SAM 3.1" else "offline twin"},
            {"model": "Muse Spark", "role": "plan, diagnosis", "state": "live" if self.spark.client else "offline twin"},
            {"model": "Muse Wonder", "role": "world model, forecast", "state": "live" if self.wonder.live else "offline twin"},
            {"model": "Muse Image", "role": "goal frame", "state": "live" if self.imager.client else "offline twin"},
            {"model": "Muse Code", "role": "policy writer", "state": code_state},
            {"model": "Muse Spark Open", "role": "red-team attacker",
             "state": "live" if Attacker().client else "offline twin"},
            {"model": "Muse Glimmer", "role": "edge fallback",
             "state": "live" if self.spark.edge.available else "not configured"},
            {"model": "Muse Voice Transcribe", "role": "spoken instruction", "state": "live" if live else "needs key"},
        ]

    def short_names(self):
        return {"perception": self.segmenter.short, "world_model": self.wonder.short,
                "goal_frame": "Muse Image" if self.imager.client else "physics prior"}


def _tk(config) -> Toolkit:
    return config["configurable"]["toolkit"]


def _summary(tel):
    if not tel:
        return {}
    keys = ("max_contact_force_n", "obstacle_displacement_m", "obstacle_tilt_deg", "goal_error_m",
            "payload_drops", "torque_peak_ratio", "friction_scale", "mass_scale", "sim_time_s", "steps")
    return {k: tel.get(k) for k in keys}


def _rollout_event(r, iteration, kind="rollout"):
    rep = r.get("report", {})
    seed = rep.get("seed", r.get("seed"))
    return {"iteration": iteration, "kind": kind, "seed": seed, "status": r["status"],
            "verified": r["verified"], "verify_msg": r["verify_msg"], "exit_code": r["exit_code"],
            "elapsed_s": r.get("elapsed_s"), "backend": r.get("backend"),
            "physics_steps": rep.get("physics_steps"), "chain_head": (rep.get("chain_head") or "")[:16],
            "violation": rep.get("violation"), "error": rep.get("error"),
            "gate": rep.get("gate"), "telemetry": _summary(rep.get("telemetry")),
            "events": (rep.get("telemetry") or {}).get("events", []), "frames": r.get("frames", []),
            "stderr_tail": r.get("stderr_tail", "")[-1500:] if r["status"] in INFRA_FAILURES else ""}


INFRA_FAILURES = ("no_report", "timeout", "unverified", "harness_error")


def _harness_fault(rollouts):
    # every seed died before the physics loop could sign anything: that is our bug, not the policy's
    real = [r for r in rollouts if r.get("backend") != "preflight"]
    if not real or any(r["status"] not in INFRA_FAILURES for r in real):
        return None
    first = real[0]
    return {"message": f"sandbox runner failed on all {len(real)} seeds before producing a signed report",
            "cause": first["verify_msg"], "exit_codes": [r["exit_code"] for r in real],
            "stderr_tail": first.get("stderr_tail", "")[-3000:], "backend": first.get("backend"),
            "platform": f"{platform.system()} {platform.release()} / Python {sys.version.split()[0]}",
            "hint": "run `python -m aegis doctor` for a single-rollout self test"}


def perceive(state: MissionState, config):
    tk = _tk(config)
    sc = scenarios.get(state["scenario"])
    tk.emit("node", {"node": "perception", "status": "start", "iteration": 0})
    w = PandaWorld(sc, seed=0, frame_every=10 ** 9)
    try:
        frame = from_snapshot(w.snapshot("initial"), w)
    finally:
        w.close()
    masks = tk.segmenter.segment(frame)
    sg = scene_graph.build(frame, masks, sc.payload.name, sc.tray.name, [o.name for o in sc.obstacles])
    roles = {n: o["role"] for n, o in sg["objects"].items()}
    tk.emit("perception", {"scene_graph": sg, "initial": viz.b64(frame.rgb),
                           "overlay": viz.b64(viz.mask_overlay(frame.rgb, masks, roles)),
                           "segmenter": tk.segmenter.name, "sam_error": tk.segmenter.last_error})
    tk.emit("node", {"node": "perception", "status": "done", "iteration": 0})
    return {"scene_graph": sg, "initial_frame": frame}


def synthesize(state: MissionState, config):
    tk = _tk(config)
    sc = scenarios.get(state["scenario"])
    tk.emit("node", {"node": "synthesis", "status": "start", "iteration": 1})

    plan = tk.spark.decompose(state["instruction"], state["scene_graph"])
    tk.emit("plan", {"plan": plan, "planner": tk.spark.name})

    # the world model predicts the success state while the policy is being written
    initial = state.get("initial_frame")
    expect_job = tk.pool.submit(_expect, tk, state["instruction"], plan, state["scene_graph"], sc, initial)

    if state.get("chaos"):
        params = dict(policies.NAIVE[sc.key])
        code = policies.render(params)
        source = f"failure injection: {sc.naive_flaw}"
    else:
        code, params = tk.coder.synthesize(state["instruction"], plan, state["scene_graph"], sc.key)
        source = tk.coder.last_source or tk.coder.name

    expect = expect_job.result()
    roles = {sc.payload.name: "payload", sc.tray.name: "goal", "robot": "robot"}
    roles.update({o.name: "obstacle" for o in sc.obstacles})
    tk.emit("wonder", {"expected": viz.b64(expect["frame"].rgb),
                       "overlay": viz.b64(viz.mask_overlay(expect["frame"].rgb, expect["masks"], roles)),
                       "source": expect["frame_source"], "world_model": expect["model"],
                       "generative": expect["generative"], "prediction": {k: expect[k] for k in ("objects", "invariants", "image_prompt")},
                       "error": tk.wonder.last_error or tk.imager.last_error})
    tk.emit("code", {"iteration": 1, "code": code, "prev_code": "", "diff": "", "source": source,
                     "params": params})
    tk.emit("node", {"node": "synthesis", "status": "done", "iteration": 1})
    return {"plan": plan, "code": code, "params": params, "iteration": 1, "expect": expect,
            "code_history": [{"iteration": 1, "code": code, "source": source}]}


def _expect(tk, instruction, plan, sg, sc, initial):
    pred, model = tk.wonder.predict(instruction, plan, sg, sc)
    frame, frame_source, generative = tk.imager.render(initial, sc, pred["image_prompt"])
    masks = tk.segmenter.segment(frame)
    return {**pred, "model": model, "model_short": tk.wonder.short if "failed" not in model else "geometric prior",
            "frame": frame, "masks": masks, "frame_source": frame_source, "generative": generative}


def execute(state: MissionState, config):
    tk = _tk(config)
    sc = scenarios.get(state["scenario"])
    it = state["iteration"]
    tk.emit("node", {"node": "sandbox", "status": "start", "iteration": it})

    forecast_job = tk.pool.submit(tk.wonder.forecast, state["code"], state["scene_graph"], sc)
    naive_future = None
    if it == 1:
        # left screen: the same first policy, graded the naive way (exit code only)
        naive_future = tk.pool.submit(tk.sandbox.run, state["code"], sc.key, 0, False,
                                      state["scene_graph"], "naive")

    pre = gate.check(state["code"])
    tk.emit("gate", {"iteration": it, "stage": "preflight AST", **pre.to_dict()})
    if not pre.passed:
        rollouts = [{"verified": True, "verify_msg": "rejected before launch", "status": "gate_rejected",
                     "exit_code": None, "frames": [], "keyframes": {}, "backend": "preflight",
                     "report": {"seed": s, "violation": {"rule": pre.violations[0].rule,
                                                        "detail": pre.summary(),
                                                        "message": "SECURITY_VIOLATION: State Teleportation Blocked"
                                                        if any(v.rule == "STATE_TELEPORTATION" for v in pre.violations)
                                                        else "SECURITY_VIOLATION: AST integrity gate"}}}
                    for s in range(3)]
        if sc.adversarial:
            # prove the runtime layer too: launch it with the static gate switched off
            probe = tk.sandbox.run(state["code"], sc.key, 0, False, state["scene_graph"])
            tk.emit("redteam_probe", _rollout_event(probe, it, "redteam_probe"))
    else:
        rollouts = tk.sandbox.run_parallel(state["code"], sc.key, (0, 1, 2), True, state["scene_graph"])
        for r in rollouts:
            tk.emit("rollout", _rollout_event(r, it))

    out = {"rollouts": rollouts, "forecast": forecast_job.result()}
    if out["forecast"]:
        tk.emit("forecast", {"iteration": it, "forecast": out["forecast"], "model": tk.wonder.name})
    fault = _harness_fault(rollouts)
    if fault:
        tk.emit("sandbox_error", fault)
        out["harness_error"] = fault
        out["verdict"] = "HARNESS_ERROR"
    if naive_future is not None:
        n = naive_future.result()
        rep = n.get("report", {})
        verdict = "SUCCESS (Exit 0)" if n["exit_code"] == 0 and n["status"] == "ok" else f"FAILED ({n['status']})"
        ev = _rollout_event(n, it, "naive")
        ev["naive_verdict"] = verdict
        tk.emit("naive", ev)
        out["naive"] = {"verdict": verdict, "telemetry": _summary(rep.get("telemetry"))}
    tk.emit("node", {"node": "sandbox", "status": "done", "iteration": it})
    return out


def audit_node(state: MissionState, config):
    tk = _tk(config)
    sc = scenarios.get(state["scenario"])
    it = state["iteration"]
    tk.emit("node", {"node": "auditor", "status": "start", "iteration": it})
    result = auditor.audit(state["rollouts"], state["expect"], sc, tk.segmenter)
    tk.emit("audit", {"iteration": it, "passed": result["passed"], "survival": result["survival"],
                      "findings": result["findings"], "advisories": result["advisories"],
                      "per_seed": [{k: v for k, v in s.items()} for s in result["per_seed"]],
                      "sam_fallbacks": tk.segmenter.fallbacks, "sam_error": tk.segmenter.last_error,
                      "sam_stats": tk.segmenter.summary()})
    upd = {"audit": result}
    fc = state.get("forecast")
    if fc:
        actual = classify(result["findings"])
        predicted = str(fc.get("predicted_outcome", "")).upper()
        tk.emit("forecast_score", {"iteration": it, "predicted": predicted, "actual": actual,
                                   "hit": predicted == actual, "confidence": fc.get("confidence")})
    if result["passed"]:
        upd["verdict"] = "VERIFIED"
    elif it >= state.get("max_iters", settings.max_iters):
        upd["verdict"] = "REJECTED"
    else:
        tel = [_summary(r.get("report", {}).get("telemetry")) for r in state["rollouts"]]
        diag = tk.spark.diagnose(result["findings"], sc, tel)
        tk.emit("diagnosis", {"iteration": it, "diagnosis": diag, "planner": tk.spark.name})
        upd["diagnosis"] = diag
    tk.emit("node", {"node": "auditor", "status": "done", "iteration": it})
    return upd


def repair(state: MissionState, config):
    tk = _tk(config)
    it = state["iteration"] + 1
    tk.emit("node", {"node": "repair", "status": "start", "iteration": it})
    new_code, new_params = tk.coder.patch(state["code"], state["diagnosis"], state.get("params"),
                                          state["scene_graph"])
    diff = "".join(difflib.unified_diff(state["code"].splitlines(True), new_code.splitlines(True),
                                        fromfile=f"policy_iter{it - 1}.py", tofile=f"policy_iter{it}.py"))
    source = tk.coder.last_source or tk.coder.name
    tk.emit("code", {"iteration": it, "code": new_code, "prev_code": state["code"], "diff": diff,
                     "source": source, "params": new_params,
                     "directives": state["diagnosis"].get("patch_directives", [])})
    tk.emit("node", {"node": "repair", "status": "done", "iteration": it})
    hist = list(state.get("code_history", [])) + [{"iteration": it, "code": new_code, "source": source}]
    return {"code": new_code, "params": new_params, "iteration": it, "code_history": hist,
            "recovery_cycles": state.get("recovery_cycles", 0) + 1}


def route(state: MissionState):
    if not state.get("verdict"):
        return "repair"
    return "redteam" if state.get("redteam") else "done"


def redteam_node(state: MissionState, config):
    tk = _tk(config)
    sc = scenarios.get(state["scenario"])
    tk.emit("node", {"node": "redteam", "status": "start", "iteration": state.get("iteration", 1)})
    result = red_team.run(tk.sandbox, sc, state["scene_graph"], state["expect"], tk.segmenter)
    tk.emit("redteam", result)
    tk.emit("node", {"node": "redteam", "status": "done", "iteration": state.get("iteration", 1)})
    return {"redteam_result": {k: v for k, v in result.items() if k != "attacks"} |
                              {"attacks": [{k: v for k, v in a.items() if k != "frames"} for a in result["attacks"]]}}


def route_after_sandbox(state: MissionState):
    return "halt" if state.get("harness_error") else "audit"


def build_graph():
    g = StateGraph(MissionState)
    g.add_node("perception", perceive)
    g.add_node("synthesis", synthesize)
    g.add_node("sandbox", execute)
    g.add_node("auditor", audit_node)
    g.add_node("repair", repair)
    g.add_node("redteam", redteam_node)
    g.add_edge(START, "perception")
    g.add_edge("perception", "synthesis")
    g.add_edge("synthesis", "sandbox")
    g.add_conditional_edges("sandbox", route_after_sandbox, {"halt": END, "audit": "auditor"})
    g.add_conditional_edges("auditor", route, {"done": END, "repair": "repair", "redteam": "redteam"})
    g.add_edge("redteam", END)
    g.add_edge("repair", "sandbox")
    return g.compile()


_GRAPH = None
_LOCK = threading.Lock()


def run_mission(scenario_key, chaos=True, emit=None, max_iters=None, instruction=None, redteam=False):
    global _GRAPH
    with _LOCK:
        if _GRAPH is None:
            _GRAPH = build_graph()
    emit = emit or (lambda kind, data: None)
    sc = scenarios.get(scenario_key)
    tk = Toolkit(emit)
    t0 = time.time()
    emit("mission_start", {"scenario": sc.key, "title": sc.title, "instruction": instruction or sc.instruction,
                           "chaos": chaos, "backends": tk.backends(), "short": tk.short_names(),
                           "stack": tk.model_stack(), "redteam": redteam,
                           "seeds": sc.randomization})
    state = _GRAPH.invoke({"scenario": sc.key, "instruction": instruction or sc.instruction, "chaos": chaos,
                           "max_iters": max_iters or settings.max_iters, "recovery_cycles": 0,
                           "redteam": redteam},
                          config={"configurable": {"toolkit": tk}, "recursion_limit": 40})
    summary = {"verdict": state.get("verdict"), "iterations": state.get("iteration"),
               "recovery_cycles": state.get("recovery_cycles", 0),
               "survival": state.get("audit", {}).get("survival"),
               "naive_verdict": (state.get("naive") or {}).get("verdict"),
               "harness_error": state.get("harness_error"),
               "redteam": {k: v for k, v in (state.get("redteam_result") or {}).items() if k != "attacks"} or None,
               "elapsed_s": round(time.time() - t0, 1), "final_code": state.get("code")}
    emit("verdict", summary)
    tk.pool.shutdown(wait=False)
    return state, summary
