import base64
import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from aegis.sandbox import protocol


@dataclass
class Frame:
    rgb: np.ndarray
    depth: np.ndarray
    view: list
    proj: list
    labels: dict  # object name -> text prompt
    seg: np.ndarray = None
    seg_ids: dict = field(default_factory=dict)
    t: float = 0.0
    ee: tuple = None
    grip_closed: bool = False
    poses: dict = field(default_factory=dict)
    tag: str = ""


def from_keyframe(kf: dict, tag="") -> Frame:
    arr = protocol.unpack_array_bundle(kf["arrays"])
    return Frame(rgb=arr["rgb"], depth=arr["depth"], seg=arr.get("seg"), view=kf["view"],
                 proj=kf["proj"], labels=kf["labels"], seg_ids=kf["seg_ids"], t=kf.get("t", 0.0),
                 ee=tuple(kf["ee"]) if kf.get("ee") else None, grip_closed=kf.get("grip_closed", False),
                 poses=kf.get("poses", {}), tag=tag)


def from_snapshot(snap: dict, world) -> Frame:
    return Frame(rgb=snap["rgb"], depth=snap["depth"], seg=snap["seg"], view=list(world.view),
                 proj=list(world.proj), labels=dict(world.labels), seg_ids=world.seg_ids(),
                 t=snap["t"], ee=snap["ee"], grip_closed=snap["grip_closed"], poses=snap["poses"],
                 tag=snap["tag"])


def to_b64(img: np.ndarray, fmt="JPEG", quality=85) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode()
