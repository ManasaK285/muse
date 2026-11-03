import json
import logging
import math

from aegis.config import settings
from aegis.models.glimmer import EdgeModel
from aegis.models.meta_client import MetaClient, chat_json

log = logging.getLogger("aegis.spark")

DECOMPOSE_PROMPT = """You plan robot manipulation for a Franka Panda arm on a tabletop.
Break the instruction into ordered kinematic sub-goals. Use only these sub-goal types:
pre_grasp, grasp_lift, obstacle_clearance_spline, place.
Reply with JSON only:
{"subgoals": [{"name": "...", "target": "<object name>", "intent": "<one sentence>",
  "constraints": ["..."]}], "success_condition": "<one sentence the camera can verify>"}"""

DIAGNOSE_PROMPT = """You are the failure analyst for a robot policy verification harness.
You get physics telemetry from domain randomized rollouts plus a visual audit that compared
SAM 3.1 masks of the real outcome against a Muse Wonder counterfactual of the expected outcome.
Find the root cause and tell the code writer exactly what to change.
Reply with JSON only:
{"failure_mode": "MICRO_COLLISION|PAYLOAD_SLIP|SECURITY_VIOLATION|PLACEMENT_ERROR|POLICY_CRASH",
 "root_cause": "<two sentences max, cite numbers>",
 "evidence": ["<short fact>", "..."],
 "patch_directives": [{"type": "raise_clearance", "margin_m": 0.04} |
                      {"type": "increase_grip_force", "newtons": 35} |
                      {"type": "slow_lift", "speed": 0.12} |
                      {"type": "legitimate_kinematics_only"} |
                      {"type": "reduce_speed"}]}"""


def offline_subgoals(instruction, scene_graph):
    objs = scene_graph.get("objects", {})
    payload = scene_graph.get("payload")
    tray = scene_graph.get("tray")
    obstacles = [n for n, o in objs.items() if o.get("role") == "obstacle"]
    return {
        "subgoals": [
            {"name": "pre_grasp", "target": payload,
             "intent": f"hover above {payload}, open gripper, descend top down",
             "constraints": ["top down approach", "fingers straddle the cylinder axis"]},
            {"name": "grasp_lift", "target": payload,
             "intent": f"close on {payload} and lift clear of the table",
             "constraints": ["grip force sized for payload mass and friction"]},
            {"name": "obstacle_clearance_spline", "target": tray,
             "intent": f"carry {payload} toward {tray}",
             "constraints": [f"do not contact {o}" for o in obstacles]},
            {"name": "place", "target": tray,
             "intent": f"lower {payload} into {tray} and release",
             "constraints": ["release only once supported"]},
        ],
        "success_condition": f"{payload} upright inside {tray}, every obstacle unmoved",
    }


def _required_grip(sc):
    # Coulomb friction with two contacts: 2 * mu * F >= m * g, sized for the worst seed, x2.5 margin
    mu_min = sc.payload.friction * min(f for f, _ in sc.randomization)
    m_max = sc.payload.mass * max(m for _, m in sc.randomization)
    need = m_max * 9.81 / (2 * mu_min)
    return float(5 * math.ceil(need * 2.5 / 5)), need, mu_min, m_max


def classify(findings):
    kinds = {f["kind"] for f in findings}
    if not findings:
        return "SUCCESS"
    if kinds & {"security_violation", "gate_rejected"}:
        return "SECURITY_VIOLATION"
    if kinds & {"payload_lost", "payload_drop", "goal_miss_slip"}:
        return "PAYLOAD_SLIP"
    if kinds & {"contact", "obstacle_shift", "obstacle_mask_shift"}:
        return "MICRO_COLLISION"
    if kinds & {"policy_error", "no_report", "timeout"}:
        return "POLICY_CRASH"
    return "PLACEMENT_ERROR"


def offline_diagnosis(findings, sc):
    kinds = {f["kind"] for f in findings}
    evidence = [f["detail"] for f in findings][:8]
    if kinds & {"security_violation", "gate_rejected"}:
        return {"failure_mode": "SECURITY_VIOLATION",
                "root_cause": "The policy tried to write simulator state directly instead of moving the arm. "
                              "Rewrite it to reach the goal through IK waypoints and the gripper only.",
                "evidence": evidence,
                "patch_directives": [{"type": "legitimate_kinematics_only"},
                                     {"type": "raise_clearance", "margin_m": 0.04}]}
    if kinds & {"payload_lost", "payload_drop", "goal_miss_slip"}:
        newtons, need, mu, m = _required_grip(sc)
        return {"failure_mode": "PAYLOAD_SLIP",
                "root_cause": (f"Grip force is below the friction limit for the worst seed: holding "
                               f"{m:.2f} kg at mu={mu:.3f} needs {need:.1f} N per finger before inertia. "
                               f"Raise to {newtons:.0f} N and lift gently."),
                "evidence": evidence,
                "patch_directives": [{"type": "increase_grip_force", "newtons": newtons},
                                     {"type": "slow_lift", "speed": 0.12}]}
    if kinds & {"contact", "obstacle_shift", "obstacle_mask_shift"}:
        tallest = max((o.height for o in sc.obstacles), default=0.0)
        return {"failure_mode": "MICRO_COLLISION",
                "root_cause": (f"The transport segment runs below the obstacle top ({tallest:.3f} m), so the "
                               f"carried payload sweeps through it. Lift above it and follow a clearance spline."),
                "evidence": evidence,
                "patch_directives": [{"type": "raise_clearance", "margin_m": 0.04}]}
    if kinds & {"policy_error", "no_report", "timeout"}:
        return {"failure_mode": "POLICY_CRASH", "root_cause": "The policy raised or overran its step budget.",
                "evidence": evidence, "patch_directives": [{"type": "legitimate_kinematics_only"}]}
    return {"failure_mode": "PLACEMENT_ERROR", "root_cause": "Payload landed outside the goal tolerance.",
            "evidence": evidence, "patch_directives": [{"type": "reduce_speed"}]}


class Spark:
    # cloud Muse Spark first, then Muse Glimmer on the edge, then deterministic rules
    def __init__(self):
        self.client = None
        self.last_error = None
        self.edge = EdgeModel()
        self.used = None
        if settings.backend("spark") == "live":
            try:
                self.client = MetaClient()
            except Exception as e:
                self.last_error = str(e)

    @property
    def name(self):
        if self.used:
            return self.used
        if self.client:
            return f"Muse Spark ({settings.spark_model})"
        return self.edge.name if self.edge.available else "offline planner"

    def _cloud(self, system, user, max_tokens):
        return chat_json(self.client, settings.spark_model, system, user, max_tokens)

    def _ask(self, system, user, ok, max_tokens=2000):
        # returns (result, backend name) or (None, None) when every model tier failed the check
        if self.client:
            try:
                out = self._cloud(system, user, max_tokens)
                if ok(out):
                    return out, f"Muse Spark ({settings.spark_model})"
            except Exception as e:
                self.last_error = str(e)[:300]
                log.warning("Muse Spark failed: %s", e)
        if self.edge.available:
            try:
                out = self.edge.ask_json(system, user, max_tokens)
                if ok(out):
                    return out, self.edge.name
            except Exception as e:
                self.last_error = f"edge: {str(e)[:280]}"
                log.warning("Muse Glimmer failed: %s", e)
        return None, None

    def decompose(self, instruction, scene_graph):
        out, used = self._ask(DECOMPOSE_PROMPT, json.dumps({"instruction": instruction, "scene_graph": scene_graph}),
                              lambda o: bool(o.get("subgoals")), 4000)
        self.used = used or "offline planner"
        return out or offline_subgoals(instruction, scene_graph)

    def diagnose(self, findings, sc, telemetry_summary):
        user = json.dumps({
            "instruction": sc.instruction, "findings": findings, "telemetry": telemetry_summary,
            "payload": {"mass_kg": sc.payload.mass, "friction": sc.payload.friction,
                        "randomization": sc.randomization},
            "obstacles": [{"name": o.name, "height_m": o.height} for o in sc.obstacles]})
        out, used = self._ask(DIAGNOSE_PROMPT, user, lambda o: bool(o.get("patch_directives")), 3000)
        self.used = used or "offline planner"
        return out or offline_diagnosis(findings, sc)
