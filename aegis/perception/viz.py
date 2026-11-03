import numpy as np
from PIL import Image, ImageDraw

from aegis.perception.frames import to_b64
from aegis.sim import camera

PALETTE = {
    "payload": (255, 70, 70),
    "obstacle": (80, 200, 255),
    "goal": (160, 255, 120),
    "robot": (200, 160, 255),
}


def mask_overlay(rgb, masks, roles, path_xyz=None, view=None, proj=None, alpha=0.5):
    out = rgb.astype(np.float32).copy()
    for name, m in masks.items():
        color = np.array(PALETTE.get(roles.get(name, "robot"), (255, 255, 0)), dtype=np.float32)
        mm = m["mask"]
        out[mm] = out[mm] * (1 - alpha) + color * alpha
        edge = mm ^ np.roll(mm, 1, 0) | mm ^ np.roll(mm, 1, 1)
        out[edge] = color
    img = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for name, m in masks.items():
        ys, xs = np.nonzero(m["mask"])
        if len(xs):
            # flat things get labelled underneath so they don't collide with what sits on them
            y = int(ys.max()) + 2 if roles.get(name) == "goal" else max(int(ys.min()) - 11, 0)
            draw.text((int(xs.min()), y), name, fill=(255, 255, 255))
    if path_xyz is not None and len(path_xyz) > 1 and view is not None:
        uv = camera.project(path_xyz, view, proj, rgb.shape[1], rgb.shape[0])
        draw.line([tuple(map(float, q)) for q in uv], fill=(255, 220, 40), width=2)
        for q in uv[:: max(len(uv) // 12, 1)]:
            draw.ellipse([q[0] - 2, q[1] - 2, q[0] + 2, q[1] + 2], fill=(255, 220, 40))
    return np.asarray(img)


def divergence_heatmap(expected_rgb, actual_rgb, expected_masks, actual_masks, focus):
    diff = np.abs(expected_rgb.astype(np.float32) - actual_rgb.astype(np.float32)).mean(axis=2) / 255.0
    # the arm is supposed to be somewhere else at the end, so its pixels are not evidence
    for masks in (expected_masks, actual_masks):
        robot = masks.get("robot", {}).get("mask")
        if robot is not None:
            diff[robot] = 0.0
    base = (actual_rgb.astype(np.float32) * 0.25)
    heat = np.zeros_like(base)
    heat[..., 0] = np.clip(diff * 3.0, 0, 1) * 255
    heat[..., 1] = np.clip(diff * 1.2, 0, 1) * 90
    out = base + heat * 0.6
    for name in focus:
        e = expected_masks.get(name, {}).get("mask")
        a = actual_masks.get(name, {}).get("mask")
        if e is None or a is None:
            continue
        missing = e & ~a
        extra = a & ~e
        out[missing] = (255, 60, 60)
        out[extra] = (255, 200, 40)
        out[e & a] = out[e & a] * 0.3 + np.array((60, 220, 120)) * 0.7
    return out.clip(0, 255).astype(np.uint8)


def b64(img):
    return to_b64(img, "JPEG", 88)
