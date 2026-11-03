import numpy as np

from aegis.config import settings
from aegis.perception import viz
from aegis.perception.frames import from_keyframe
from aegis.perception.scene_graph import object_geometry
from aegis.sim import camera

LIMITS = {
    "contact_n": 1.0,          # any real push on a fragile obstacle
    "obstacle_disp_m": 0.002,  # 2 mm micro collision threshold
    "obstacle_tilt_deg": 3.0,
    "visual_shift_m": 0.01,
    "payload_iou": 0.3,
    "goal_xy_m": 0.04,
    "carry_radius_m": 0.08,
}


def iou(a, b):
    if a is None or b is None:
        return 0.0
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def _centroid(frame, mask):
    if mask is None or mask.sum() < 12:
        return None
    geo = object_geometry(camera.deproject(frame.depth, frame.view, frame.proj, mask))
    return np.array(geo["centroid"]) if geo else None


def _src(*entries):
    # name the evidence by what actually produced it: "SAM 3.1" only if every mask involved came from SAM
    srcs = {(e or {}).get("source", "renderer") for e in entries}
    return "SAM 3.1" if srcs == {"SAM 3.1"} else "renderer"


def _mask(masks, name):
    return masks.get(name, {}).get("mask")


def _evenly(items, k):
    if k <= 0 or len(items) <= k:
        return items
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in sorted(set(idx))]


def audit_rollout(r, expect, sc, segmenter):
    # expect: {"objects", "frame", "masks", "generative", "frame_source", "model_short"}
    expected, expected_masks = expect["frame"], expect["masks"]
    exp_name = expect.get("model_short", "expected state")
    seed = r.get("report", {}).get("seed", r.get("seed", "?"))
    out = {"seed": seed, "passed": False, "findings": [], "advisories": [], "metrics": {}, "images": {}}
    f = out["findings"]
    status = r["status"]
    rep = r.get("report", {})

    if not r["verified"]:
        f.append({"kind": "no_report", "detail": f"seed {seed}: {r['verify_msg']} (exit {r['exit_code']})"})
        return out
    if status in ("security_violation", "gate_rejected"):
        f.append({"kind": status, "detail": rep.get("violation", {}).get("message", status)})
        return out
    if status != "ok":
        f.append({"kind": "policy_error", "detail": f"seed {seed}: {rep.get('error', status)}"})

    tel = rep.get("telemetry") or {}
    m = out["metrics"]
    m["max_contact_n"] = max(tel.get("max_contact_force_n", {}).values(), default=0.0)
    m["obstacle_disp_mm"] = round(1000 * max(tel.get("obstacle_displacement_m", {}).values(), default=0.0), 2)
    m["goal_error_m"] = tel.get("goal_error_m")
    m["torque_peak_ratio"] = tel.get("torque_peak_ratio")

    for ob, force in tel.get("max_contact_force_n", {}).items():
        if force > LIMITS["contact_n"]:
            first = next((c for c in tel.get("contact_log", []) if c["obstacle"] == ob), None)
            peak = max((c for c in tel.get("contact_log", []) if c["obstacle"] == ob),
                       key=lambda c: c["force_n"], default=first)
            where = f" at t={peak['t']}s, xyz={peak['at']}" if peak else ""
            f.append({"kind": "contact", "detail": f"seed {seed}: {force:.1f} N lateral contact on {ob}{where}",
                      "at": peak["at"] if peak else None})
    for ob, d in tel.get("obstacle_displacement_m", {}).items():
        tilt = tel.get("obstacle_tilt_deg", {}).get(ob, 0.0)
        if d > LIMITS["obstacle_disp_m"] or tilt > LIMITS["obstacle_tilt_deg"]:
            f.append({"kind": "obstacle_shift",
                      "detail": f"seed {seed}: {ob} moved {d * 1000:.1f} mm and tilted {tilt:.0f} deg"})
    if tel.get("payload_drops"):
        f.append({"kind": "payload_drop",
                  "detail": f"seed {seed}: payload left the gripper while it was closed "
                            f"(friction x{tel.get('friction_scale')}, mass x{tel.get('mass_scale')})"})

    kfs = r.get("keyframes", {})
    if "initial" not in kfs or "final" not in kfs:
        return out
    init = from_keyframe(kfs["initial"], "initial")
    final = from_keyframe(kfs["final"], "final")
    m_init = segmenter.segment(init)
    m_final = segmenter.segment(final)
    roles = {sc.payload.name: "payload", sc.tray.name: "goal", "robot": "robot"}
    roles.update({o.name: "obstacle" for o in sc.obstacles})

    payload_iou = iou(_mask(m_final, sc.payload.name), _mask(expected_masks, sc.payload.name))
    m["payload_mask_iou"] = round(payload_iou, 3)
    m["iou_is_advisory"] = bool(expect.get("generative"))
    pred = expect["objects"].get(sc.payload.name, {})
    goal_xy = np.array(pred.get("centroid", [0, 0, 0])[:2], dtype=float)
    tol = float(np.clip(pred.get("tolerance_m", LIMITS["goal_xy_m"]), 0.02, 0.06))
    pc = _centroid(final, _mask(m_final, sc.payload.name))
    vis_goal = float(np.linalg.norm(pc[:2] - goal_xy)) if pc is not None else None
    m["visual_goal_error_m"] = round(vis_goal, 4) if vis_goal is not None else None
    m["goal_tolerance_m"] = round(tol, 3)
    iou_fail = payload_iou < LIMITS["payload_iou"]
    if expect.get("generative") and iou_fail:
        # a generated image does not preserve exact geometry, so mask IoU on it can only warn
        out["advisories"].append(f"seed {seed}: payload mask IoU vs {expect['frame_source']} frame = "
                                 f"{payload_iou:.2f} (advisory)")
        iou_fail = False
    if vis_goal is None or vis_goal > tol or iou_fail:
        kind = "goal_miss_slip" if tel.get("payload_drops") else "goal_miss"
        where = f"{vis_goal * 100:.1f} cm from the {exp_name} goal" if vis_goal is not None else "not visible"
        f.append({"kind": kind, "detail": f"seed {seed}: {_src(m_final.get(sc.payload.name))} payload mask {where} "
                                          f"(mask IoU vs expected frame {payload_iou:.2f})"})

    fixed = {n for n, o in expect["objects"].items() if o.get("must_not_move")}
    fixed |= {o.name for o in sc.obstacles}
    shifts = {}
    for ob in [o for o in sc.obstacles if o.name in fixed]:
        a, b = _mask(m_init, ob.name), _mask(m_final, ob.name)
        ca, cb = _centroid(init, a), _centroid(final, b)
        shift = float(np.linalg.norm(cb - ca)) if ca is not None and cb is not None else None
        ov = iou(a, b)
        shifts[ob.name] = {"shift_m": round(shift, 4) if shift is not None else None, "mask_iou": round(ov, 3)}
        if shift is None or shift > LIMITS["visual_shift_m"] or ov < 0.6:
            f.append({"kind": "obstacle_mask_shift",
                      "detail": f"seed {seed}: {_src(m_init.get(ob.name), m_final.get(ob.name))} mask of {ob.name} moved "
                                f"{(shift or 0) * 1000:.0f} mm (mask IoU {ov:.2f} vs pre-rollout)"})
    m["obstacle_visual"] = shifts

    carry_checks = []
    carry_keys = sorted(k for k in kfs if k.startswith("carry_"))
    if not isinstance(segmenter.primary, type(segmenter.backup)):
        carry_keys = _evenly(carry_keys, settings.sam_max_carry)  # live SAM is billed per frame
    for k in carry_keys:
        cf = from_keyframe(kfs[k], k)
        cm = segmenter.segment(cf, only=[sc.payload.name])
        c = _centroid(cf, _mask(cm, sc.payload.name))
        d = float(np.linalg.norm(c - np.array(cf.ee))) if c is not None and cf.ee else None
        ok = d is not None and d < LIMITS["carry_radius_m"]
        carry_checks.append({"t": cf.t, "payload_to_ee_m": round(d, 3) if d is not None else None, "held": ok})
        if not ok:
            f.append({"kind": "payload_lost",
                      "detail": f"seed {seed}: at t={cf.t:.2f}s no {_src(cm.get(sc.payload.name))} payload mask inside the end effector box"})
            break
    m["carry_checks"] = carry_checks

    path = tel.get("ee_path") or []
    overlay = viz.mask_overlay(final.rgb, m_final, roles, np.array(path) if path else None, final.view, final.proj)
    focus = [sc.payload.name] + [o.name for o in sc.obstacles]
    heat = viz.divergence_heatmap(expected.rgb, final.rgb, expected_masks, m_final, focus)
    out["images"] = {"overlay": viz.b64(overlay), "heatmap": viz.b64(heat), "final": viz.b64(final.rgb)}
    out["passed"] = not f
    return out


def audit(rollouts, expect, sc, segmenter):
    per_seed = [audit_rollout(r, expect, sc, segmenter) for r in rollouts]
    survived = sum(1 for a in per_seed if a["passed"])
    findings = [x for a in per_seed for x in a["findings"]]
    return {"passed": survived == len(rollouts) and len(rollouts) > 0, "survival": f"{survived}/{len(rollouts)}",
            "survived": survived, "per_seed": per_seed, "findings": findings,
            "advisories": [x for a in per_seed for x in a.get("advisories", [])]}
