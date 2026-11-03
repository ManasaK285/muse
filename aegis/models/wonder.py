import json
import logging

import pybullet as p

from aegis import scenarios
from aegis.config import settings
from aegis.models.meta_client import MetaClient, chat_json
from aegis.perception.frames import Frame, from_snapshot
from aegis.sim.world import PandaWorld

log = logging.getLogger("aegis.wonder")

PREDICT_PROMPT = """You are the counterfactual world model inside a robot policy verification harness.
You get a manipulation instruction, the plan, and a scene graph built from SAM 3.1 masks deprojected
through a depth camera (world frame, meters, z up, table at z=0). Predict the scene AFTER the task
succeeds cleanly. Be physically precise: an object resting on a surface has its centroid at
surface height plus half its own height.
Reply with JSON only:
{"objects": {"<object name>": {"centroid": [x, y, z], "must_not_move": true|false,
   "tolerance_m": <float>, "upright": true|false}},
 "invariants": ["<one checkable statement per line>"],
 "image_prompt": "<one paragraph instructing an image editor to turn the current camera frame into the success frame>"}
List every object from the scene graph."""

FORECAST_PROMPT = """You are the counterfactual world model inside a robot policy verification harness.
Before a generated Python policy is executed in physics, read it together with the scene graph and the
payload's physical priors, and forecast what will actually happen when it runs. Think about swept volume
versus obstacle tops, grip force versus friction and mass (two finger contacts: 2 * mu * F must exceed m * g
with margin), and any attempt to touch simulator state directly.
Reply with JSON only:
{"predicted_outcome": "SUCCESS|MICRO_COLLISION|PAYLOAD_SLIP|SECURITY_VIOLATION|PLACEMENT_ERROR|POLICY_CRASH",
 "confidence": <0..1>,
 "reasoning": "<two sentences, cite numbers from the code and the scene>",
 "risky_lines": [<line numbers>]}"""


def imagine_goal(sc) -> Frame:
    # physics prior: same scene with the payload at the goal, rendered in the trusted process
    w = PandaWorld(sc, seed=0, frame_every=10 ** 9)
    try:
        gx, gy, gz = scenarios.goal_position(sc)
        p.resetBasePositionAndOrientation(w.ids[sc.payload.name], [gx, gy, gz + 0.001], [0, 0, 0, 1],
                                          physicsClientId=w.cid)
        for _ in range(30):
            p.stepSimulation(physicsClientId=w.cid)
        return from_snapshot(w.snapshot("expected"), w)
    finally:
        w.close()


def offline_prediction(scene_graph, sc):
    objs = scene_graph.get("objects", {})
    out = {}
    tray = objs.get(sc.tray.name, {})
    tray_xy = (tray.get("centroid") or list(scenarios.goal_position(sc)))[:2]
    for name, o in objs.items():
        if name == sc.payload.name:
            z = sc.tray.thickness + sc.payload.height / 2
            out[name] = {"centroid": [round(tray_xy[0], 4), round(tray_xy[1], 4), round(z, 4)],
                         "must_not_move": False, "tolerance_m": 0.04, "upright": True}
        else:
            out[name] = {"centroid": o["centroid"], "must_not_move": True, "tolerance_m": 0.002,
                         "upright": True}
    inv = [f"{sc.payload.name} rests upright inside {sc.tray.name}"]
    inv += [f"{o.name} does not move more than 2 mm and stays upright" for o in sc.obstacles]
    prompt = (f"Same camera, same lighting. The {sc.payload.prompt} now stands upright in the center of the "
              f"{sc.tray.prompt}. Every other object is exactly where it was. The robot arm is out of the way.")
    return {"objects": out, "invariants": inv, "image_prompt": prompt}


class WorldModel:
    def __init__(self):
        self.client = None
        self.last_error = None
        if settings.backend("wonder") == "live":
            try:
                self.client = MetaClient()
            except Exception as e:
                self.last_error = str(e)

    @property
    def live(self):
        return self.client is not None

    @property
    def name(self):
        return f"Muse Wonder ({settings.wonder_model})" if self.client else "offline (geometric prior)"

    @property
    def short(self):
        return "Muse Wonder" if self.client else "geometric prior"

    def _ask(self, system, user, max_tokens=6000):
        return chat_json(self.client, settings.wonder_model, system, user, max_tokens)

    def predict(self, instruction, plan, scene_graph, sc):
        fallback = offline_prediction(scene_graph, sc)
        if not self.client:
            return fallback, self.name
        try:
            out = self._ask(PREDICT_PROMPT, json.dumps({"instruction": instruction, "plan": plan,
                                                        "scene_graph": scene_graph}))
            objs = out.get("objects") or {}
            if sc.payload.name in objs and len(objs.get(sc.payload.name, {}).get("centroid", [])) == 3:
                for k, v in fallback["objects"].items():
                    objs.setdefault(k, v)
                out["objects"] = objs
                out.setdefault("invariants", fallback["invariants"])
                out.setdefault("image_prompt", fallback["image_prompt"])
                return out, self.name
            self.last_error = "Wonder prediction missing the payload centroid"
        except Exception as e:
            self.last_error = str(e)[:300]
            log.warning("Muse Wonder predict failed: %s", e)
        return fallback, "offline (geometric prior, Wonder call failed)"

    def forecast(self, code, scene_graph, sc):
        if not self.client:
            return None
        numbered = "\n".join(f"{i + 1:3d} {line}" for i, line in enumerate(code.splitlines()))
        try:
            return self._ask(FORECAST_PROMPT, json.dumps({
                "instruction": sc.instruction, "scene_graph": scene_graph,
                "payload_priors": {"mass_kg": sc.payload.mass, "friction": sc.payload.friction,
                                   "randomization_friction_x_mass": sc.randomization},
                "obstacles": [{"name": o.name, "height_m": o.height} for o in sc.obstacles],
                "policy": numbered}), max_tokens=4000)
        except Exception as e:
            self.last_error = str(e)[:300]
            log.warning("Muse Wonder forecast failed: %s", e)
            return None
