import io
import math

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image

from aegis import scenarios
from aegis.sim import camera

DT = 1.0 / 240.0
ARM_JOINTS = list(range(7))
FINGERS = (9, 10)
EE_LINK = 11
TORQUE_LIMITS = np.array([87, 87, 87, 87, 12, 12, 12], dtype=np.float64)
LOWER = [-2.9671, -1.8326, -2.9671, -3.0718, -2.9671, -0.0175, -2.9671]
UPPER = [2.9671, 1.8326, 2.9671, -0.0698, 2.9671, 3.7525, 2.9671]
RANGES = [u - l for l, u in zip(LOWER, UPPER)]
HOME = [0.0, -0.4, 0.0, -2.3, 0.0, 1.95, 0.785]


def jpeg_bytes(rgb, quality=82):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class PandaWorld:
    def __init__(self, scenario, seed=0, frame_every=16, gui=False):
        self.sc = scenario
        self.seed = seed
        f_scale, m_scale = scenario.randomization[seed % len(scenario.randomization)]
        self.f_scale, self.m_scale = f_scale, m_scale
        self.frame_every = frame_every
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.setTimeStep(DT, physicsClientId=self.cid)
        p.setPhysicsEngineParameter(numSolverIterations=80, physicsClientId=self.cid)
        self.view, self.proj = camera.matrices(p)
        self.ids = {}
        self.labels = {}
        self._build()
        self.steps = 0
        self.t = 0.0
        self.grip_cmd = {"closed": False, "force": 0.0}
        self.frames = []
        self.carry_frames = []
        self.events = []
        self.contact_log = []
        self.max_contact = {name: 0.0 for name in self.obstacle_names}
        self.torque_peak = 0.0
        self.torque_sat_steps = 0
        self.payload_track = []
        self.step_hooks = []
        self.initial_poses = {n: self.pose(n) for n in self.ids}

    def _cyl(self, body: scenarios.Body, mass_scale=1.0, friction_scale=1.0):
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=body.radius, height=body.height,
                                     physicsClientId=self.cid)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=body.radius, length=body.height,
                                  rgbaColor=body.color, physicsClientId=self.cid)
        uid = p.createMultiBody(body.mass * mass_scale, col, vis,
                                [body.xy[0], body.xy[1], body.height / 2 + 0.0005],
                                physicsClientId=self.cid)
        p.changeDynamics(uid, -1, lateralFriction=body.friction * friction_scale,
                         spinningFriction=0.002, rollingFriction=0.0005,
                         physicsClientId=self.cid)
        return uid

    def _build(self):
        sc = self.sc
        cid = self.cid
        self.ids["table"] = p.loadURDF("plane.urdf", physicsClientId=cid)
        p.changeVisualShape(self.ids["table"], -1, rgbaColor=(0.16, 0.17, 0.19, 1), physicsClientId=cid)
        self.robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True,
                                flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT, physicsClientId=cid)
        self.ids["robot"] = self.robot
        for j, q in enumerate(HOME):
            p.resetJointState(self.robot, j, q, physicsClientId=cid)
        for f in FINGERS:
            p.resetJointState(self.robot, f, 0.04, physicsClientId=cid)
            p.changeDynamics(self.robot, f, lateralFriction=1.0, spinningFriction=0.01,
                             physicsClientId=cid)
        p.setJointMotorControlArray(self.robot, ARM_JOINTS, p.POSITION_CONTROL, targetPositions=HOME,
                                    forces=TORQUE_LIMITS.tolist(), physicsClientId=cid)
        p.setJointMotorControlArray(self.robot, list(FINGERS), p.POSITION_CONTROL,
                                    targetPositions=[0.04, 0.04], forces=[20, 20], physicsClientId=cid)

        t = sc.tray
        half = [t.half_extent, t.half_extent, t.thickness / 2]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cid)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=t.color, physicsClientId=cid)
        self.ids[t.name] = p.createMultiBody(0, col, vis, [t.xy[0], t.xy[1], t.thickness / 2],
                                             physicsClientId=cid)

        self.ids[sc.payload.name] = self._cyl(sc.payload, self.m_scale, self.f_scale)
        self.obstacle_names = []
        for ob in sc.obstacles:
            self.ids[ob.name] = self._cyl(ob)
            self.obstacle_names.append(ob.name)

        self.labels = {sc.payload.name: sc.payload.prompt, t.name: t.prompt, "robot": "robot arm"}
        for ob in sc.obstacles:
            self.labels[ob.name] = ob.prompt
        # let everything settle before anyone looks at it
        for _ in range(60):
            p.stepSimulation(physicsClientId=cid)

    def pose(self, name):
        pos, orn = p.getBasePositionAndOrientation(self.ids[name], physicsClientId=self.cid)
        return tuple(pos), tuple(orn)

    def ee_pose(self):
        st = p.getLinkState(self.robot, EE_LINK, computeForwardKinematics=True, physicsClientId=self.cid)
        return tuple(st[4]), tuple(st[5])

    def finger_width(self):
        a = p.getJointState(self.robot, FINGERS[0], physicsClientId=self.cid)[0]
        b = p.getJointState(self.robot, FINGERS[1], physicsClientId=self.cid)[0]
        return a + b

    def tilt_deg(self, name):
        _, orn = self.pose(name)
        r = p.getMatrixFromQuaternion(orn)
        return math.degrees(math.acos(max(-1.0, min(1.0, r[8]))))

    def snapshot(self, tag):
        rgb, depth, seg = camera.render(p, self.cid, self.view, self.proj)
        return {
            "tag": tag, "t": round(self.t, 3), "rgb": rgb, "depth": depth, "seg": seg,
            "ee": self.ee_pose()[0], "grip_closed": self.grip_cmd["closed"],
            "poses": {n: self.pose(n)[0] for n in self.ids if n not in ("robot", "table")},
        }

    def seg_ids(self):
        return {n: uid for n, uid in self.ids.items() if n != "table"}

    def step(self):
        p.stepSimulation(physicsClientId=self.cid)
        self.steps += 1
        self.t = self.steps * DT
        self._record()
        for hook in self.step_hooks:
            hook(self)

    def _record(self):
        cid = self.cid
        pl = self.ids[self.sc.payload.name]
        for name in self.obstacle_names:
            oid = self.ids[name]
            total = 0.0
            where = None
            for other in (self.robot, pl):
                for c in p.getContactPoints(other, oid, physicsClientId=cid):
                    total += c[9]
                    where = c[5]
            if total > self.max_contact[name]:
                self.max_contact[name] = total
            if total > 0.5:
                self.contact_log.append({"t": round(self.t, 3), "obstacle": name,
                                         "force_n": round(total, 2),
                                         "at": [round(v, 4) for v in where]})
        if self.steps % 4 == 0:
            states = p.getJointStates(self.robot, ARM_JOINTS, physicsClientId=cid)
            tau = np.abs([s[3] for s in states]) / TORQUE_LIMITS
            self.torque_peak = max(self.torque_peak, float(tau.max()))
            if tau.max() > 0.98:
                self.torque_sat_steps += 1
            ee = self.ee_pose()[0]
            pp = self.pose(self.sc.payload.name)[0]
            self.payload_track.append((round(self.t, 3), ee, pp, self.grip_cmd["closed"]))
        if self.steps % self.frame_every == 0:
            rgb, _, _ = camera.render(p, cid, self.view, self.proj, width=camera.VIDEO_W,
                                      height=camera.VIDEO_H, with_seg=False)
            self.frames.append(jpeg_bytes(rgb))
        if self.grip_cmd["closed"] and self.steps % 120 == 0:
            ee = self.ee_pose()[0]
            if ee[2] > 0.12:
                self.carry_frames.append(self.snapshot("carry"))

    def telemetry(self):
        payload_final = self.pose(self.sc.payload.name)[0]
        disp = {}
        tilt = {}
        for n in self.obstacle_names:
            a = np.array(self.initial_poses[n][0])
            b = np.array(self.pose(n)[0])
            disp[n] = float(np.linalg.norm(b - a))
            tilt[n] = self.tilt_deg(n)
        gx, gy, gz = scenarios.goal_position(self.sc)
        err = float(np.linalg.norm(np.array(payload_final) - np.array([gx, gy, gz])))
        drops = 0
        prev_attached = False
        for t, ee, pp, closed in self.payload_track:
            attached = closed and np.linalg.norm(np.array(ee) - np.array(pp)) < 0.06
            if prev_attached and not attached and closed:
                drops += 1
            prev_attached = attached
        return {
            "seed": self.seed,
            "friction_scale": self.f_scale,
            "mass_scale": self.m_scale,
            "steps": self.steps,
            "sim_time_s": round(self.t, 3),
            "max_contact_force_n": {k: round(v, 2) for k, v in self.max_contact.items()},
            "obstacle_displacement_m": {k: round(v, 5) for k, v in disp.items()},
            "obstacle_tilt_deg": {k: round(v, 2) for k, v in tilt.items()},
            "payload_final": [round(v, 4) for v in payload_final],
            "goal_error_m": round(err, 4),
            "payload_drops": drops,
            "torque_peak_ratio": round(self.torque_peak, 3),
            "torque_saturation_samples": self.torque_sat_steps,
            "contact_log": self.contact_log[:40],
            "ee_path": [list(map(lambda v: round(v, 4), r[1])) for r in self.payload_track[::6]],
            "payload_path": [list(map(lambda v: round(v, 4), r[2])) for r in self.payload_track[::6]],
            "events": self.events,
        }

    def close(self):
        try:
            p.disconnect(self.cid)
        except Exception:
            pass
