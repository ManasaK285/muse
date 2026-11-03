import math

import numpy as np
import pybullet as p

from aegis.sim.world import ARM_JOINTS, EE_LINK, FINGERS, LOWER, RANGES, TORQUE_LIMITS, UPPER, DT

MAX_SPEED = 0.6
MAX_GRIP_FORCE = 70.0


class BudgetExceeded(RuntimeError):
    pass


class SafePandaEnv:
    # internals are underscored on purpose, the AST gate rejects any "_" attribute access

    def __init__(self, world, scene_graph=None, max_sim_seconds=20.0):
        self._w = world
        self._scene = scene_graph or {}
        self._budget = int(max_sim_seconds / DT)

    @property
    def payload(self):
        return self._w.sc.payload.name

    @property
    def obstacles(self):
        return list(self._w.obstacle_names)

    @property
    def tray(self):
        return self._w.sc.tray.name

    @property
    def scene(self):
        return dict(self._scene)

    def get_position(self, name):
        return self._w.pose(name)[0]

    def get_size(self, name):
        sc = self._w.sc
        if name == sc.tray.name:
            return {"half_extent": sc.tray.half_extent, "height": sc.tray.thickness}
        for b in [sc.payload] + list(sc.obstacles):
            if b.name == name:
                return {"radius": b.radius, "height": b.height}
        raise KeyError(name)

    def ee_position(self):
        return self._w.ee_pose()[0]

    def gripper_width(self):
        return self._w.finger_width()

    def log(self, msg):
        self._w.events.append({"t": round(self._w.t, 3), "msg": str(msg)[:200]})

    def move_to(self, xyz, speed=0.25, yaw=0.0):
        speed = float(min(max(speed, 0.02), MAX_SPEED))
        start = np.array(self.ee_position())
        goal = np.array(xyz, dtype=np.float64)
        dist = float(np.linalg.norm(goal - start))
        n = max(int(dist / (speed * DT)), 1)
        orn = p.getQuaternionFromEuler([math.pi, 0.0, yaw])
        for i in range(1, n + 1):
            s = i / n
            s = s * s * (3 - 2 * s)  # smoothstep velocity profile
            self._track(start + (goal - start) * s, orn)
        for _ in range(60):
            if np.linalg.norm(np.array(self.ee_position()) - goal) < 0.004:
                break
            self._track(goal, orn)

    def follow(self, waypoints, speed=0.25, yaw=0.0):
        for wp in waypoints:
            self.move_to(wp, speed=speed, yaw=yaw)

    def open_gripper(self, width=0.08):
        half = float(min(max(width / 2, 0.0), 0.04))
        self._w.grip_cmd = {"closed": False, "force": 0.0}
        p.setJointMotorControlArray(self._w.robot, list(FINGERS), p.POSITION_CONTROL,
                                    targetPositions=[half, half], forces=[20, 20],
                                    physicsClientId=self._w.cid)
        self._idle(0.35)

    def close_gripper(self, force=20.0):
        force = float(min(max(force, 1.0), MAX_GRIP_FORCE))
        self._w.grip_cmd = {"closed": True, "force": force}
        # velocity mode saturates at `force` once the fingers stall, giving a real clamp force
        p.setJointMotorControlArray(self._w.robot, list(FINGERS), p.VELOCITY_CONTROL,
                                    targetVelocities=[-0.05, -0.05], forces=[force, force],
                                    physicsClientId=self._w.cid)
        self._w.events.append({"t": round(self._w.t, 3), "msg": f"grip close {force:.0f}N"})
        self._idle(0.5)

    def wait(self, seconds):
        self._idle(min(float(seconds), 3.0))

    def _idle(self, seconds):
        q = [s[0] for s in p.getJointStates(self._w.robot, ARM_JOINTS, physicsClientId=self._w.cid)]
        p.setJointMotorControlArray(self._w.robot, ARM_JOINTS, p.POSITION_CONTROL, targetPositions=q,
                                    forces=TORQUE_LIMITS.tolist(), physicsClientId=self._w.cid)
        for _ in range(int(seconds / DT)):
            self._tick()

    def _track(self, xyz, orn):
        w = self._w
        rest = [s[0] for s in p.getJointStates(w.robot, ARM_JOINTS, physicsClientId=w.cid)]
        q = p.calculateInverseKinematics(w.robot, EE_LINK, list(xyz), orn, lowerLimits=LOWER,
                                         upperLimits=UPPER, jointRanges=RANGES, restPoses=rest,
                                         maxNumIterations=40, residualThreshold=1e-4,
                                         physicsClientId=w.cid)
        p.setJointMotorControlArray(w.robot, ARM_JOINTS, p.POSITION_CONTROL,
                                    targetPositions=list(q[:7]), forces=TORQUE_LIMITS.tolist(),
                                    physicsClientId=w.cid)
        self._tick()

    def _tick(self):
        if self._w.steps >= self._budget:
            raise BudgetExceeded(f"policy exceeded {self._budget} physics steps")
        self._w.step()
