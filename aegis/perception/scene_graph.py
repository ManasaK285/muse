import numpy as np

from aegis.perception.frames import Frame
from aegis.sim import camera

SAFETY_MARGIN_M = 0.02


def object_geometry(points: np.ndarray):
    if len(points) < 12:
        return None
    lo = np.percentile(points, 2, axis=0)
    hi = np.percentile(points, 98, axis=0)
    keep = np.all((points >= lo) & (points <= hi), axis=1)
    pts = points[keep] if keep.sum() >= 8 else points
    xy = pts[:, :2]
    c = xy.mean(axis=0)
    cov = np.cov((xy - c).T) if len(xy) > 2 else np.eye(2) * 1e-6
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, np.argmax(evals)]
    yaw = float(np.arctan2(major[1], major[0]))
    rot = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    local = (xy - c) @ rot.T
    half_xy = (local.max(axis=0) - local.min(axis=0)) / 2
    z_lo, z_hi = float(pts[:, 2].min()), float(pts[:, 2].max())
    return {
        "centroid": [round(float(c[0]), 4), round(float(c[1]), 4), round((z_lo + z_hi) / 2, 4)],
        "obb": {"center_xy": [round(float(c[0]), 4), round(float(c[1]), 4)], "yaw_rad": round(yaw, 3),
                "half_extents_xy": [round(float(v), 4) for v in half_xy],
                "z_range": [round(z_lo, 4), round(z_hi, 4)]},
        "visible_points": int(len(pts)),
    }


def build(frame: Frame, masks: dict, payload: str, tray: str, obstacles: list) -> dict:
    objects = {}
    for name, m in masks.items():
        if name == "robot":
            continue
        pts = camera.deproject(frame.depth, frame.view, frame.proj, m["mask"])
        geo = object_geometry(pts)
        if geo is None:
            continue
        role = "payload" if name == payload else "goal" if name == tray else "obstacle"
        geo.update({"role": role, "prompt": m.get("prompt", name), "sam_score": round(m.get("score", 0), 3),
                    "pixels": int(m["mask"].sum())})
        if role == "obstacle":
            r = max(geo["obb"]["half_extents_xy"])
            geo["keep_out"] = {"center_xy": geo["obb"]["center_xy"], "radius": round(r + SAFETY_MARGIN_M, 4),
                               "top_z": round(geo["obb"]["z_range"][1] + SAFETY_MARGIN_M, 4)}
        objects[name] = geo
    relations = []
    if payload in objects and tray in objects:
        for ob in obstacles:
            if ob not in objects:
                continue
            a = np.array(objects[payload]["centroid"][:2])
            b = np.array(objects[tray]["centroid"][:2])
            o = np.array(objects[ob]["centroid"][:2])
            ab = b - a
            s = float(np.clip(np.dot(o - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1))
            dist = float(np.linalg.norm(a + s * ab - o))
            if dist < objects[ob]["keep_out"]["radius"] + 0.03:
                relations.append({"type": "blocks_straight_path", "subject": ob, "between": [payload, tray],
                                  "lateral_gap_m": round(dist, 4)})
    return {"payload": payload, "tray": tray, "obstacles": obstacles, "objects": objects,
            "relations": relations, "frame": "world, meters, z up"}
