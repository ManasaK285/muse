import numpy as np

WIDTH, HEIGHT = 320, 240
VIDEO_W, VIDEO_H = 240, 180
NEAR, FAR = 0.05, 3.0
FOV = 50.0

TARGET = (0.5, -0.04, 0.08)
DISTANCE = 0.95
YAW, PITCH = 60.0, -44.0


def matrices(p):
    view = p.computeViewMatrixFromYawPitchRoll(TARGET, DISTANCE, YAW, PITCH, 0.0, 2)
    proj = p.computeProjectionMatrixFOV(FOV, WIDTH / HEIGHT, NEAR, FAR)
    return view, proj


def render(p, client, view, proj, width=WIDTH, height=HEIGHT, with_seg=True):
    flags = 0 if with_seg else p.ER_NO_SEGMENTATION_MASK
    _, _, rgba, depth, seg = p.getCameraImage(
        width, height, view, proj, renderer=p.ER_TINY_RENDERER, flags=flags,
        physicsClientId=client)
    rgb = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    depth = np.asarray(depth, dtype=np.float32).reshape(height, width)
    seg = np.asarray(seg, dtype=np.int32).reshape(height, width) if with_seg else None
    return rgb, depth, seg


def deproject(depth, view, proj, mask=None):
    # depth buffer -> world xyz for every (masked) pixel
    h, w = depth.shape
    vs, us = np.nonzero(mask) if mask is not None else np.indices((h, w)).reshape(2, -1)
    if len(us) == 0:
        return np.zeros((0, 3))
    x = 2.0 * (us + 0.5) / w - 1.0
    y = 1.0 - 2.0 * (vs + 0.5) / h
    z = 2.0 * depth[vs, us] - 1.0
    clip = np.stack([x, y, z, np.ones_like(x)], axis=0)
    V = np.asarray(view, dtype=np.float64).reshape(4, 4).T
    P = np.asarray(proj, dtype=np.float64).reshape(4, 4).T
    world = np.linalg.inv(P @ V) @ clip
    world /= world[3]
    return world[:3].T


def project(points, view, proj, width=WIDTH, height=HEIGHT):
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    V = np.asarray(view, dtype=np.float64).reshape(4, 4).T
    P = np.asarray(proj, dtype=np.float64).reshape(4, 4).T
    hom = np.c_[pts, np.ones(len(pts))].T
    clip = P @ V @ hom
    ndc = clip[:3] / clip[3]
    u = (ndc[0] + 1.0) * 0.5 * width
    v = (1.0 - ndc[1]) * 0.5 * height
    return np.stack([u, v], axis=1)
