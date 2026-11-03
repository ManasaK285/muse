import logging

import numpy as np
from PIL import Image

from aegis.config import settings
from aegis.models.meta_client import MetaClient, decode_image, fetch_image, png_data_url
from aegis.models.wonder import imagine_goal
from aegis.perception.frames import Frame

log = logging.getLogger("aegis.muse_image")


def _first_image(data):
    items = []
    if isinstance(data, dict):
        items = data.get("data") or data.get("images") or data.get("output") or []
        if isinstance(items, dict):
            items = [items]
    for it in items if isinstance(items, list) else []:
        if isinstance(it, str):
            return ("b64", it)
        if isinstance(it, dict):
            if it.get("b64_json") or it.get("b64") or it.get("image"):
                return ("b64", it.get("b64_json") or it.get("b64") or it.get("image"))
            if it.get("url"):
                return ("url", it["url"])
            for c in it.get("content", []) if isinstance(it.get("content"), list) else []:
                if isinstance(c, dict) and (c.get("b64_json") or c.get("image_url") or c.get("url")):
                    v = c.get("b64_json") or c.get("image_url") or c.get("url")
                    return ("url" if str(v).startswith("http") else "b64", v)
    return None


class GoalFrameRenderer:
    # expected success frame: Muse Image edit of the real camera frame, or the physics prior offline
    def __init__(self):
        self.client = None
        self.last_error = None
        if settings.backend("image") == "live":
            try:
                self.client = MetaClient()
            except Exception as e:
                self.last_error = str(e)

    @property
    def name(self):
        return f"Muse Image ({settings.image_model})" if self.client else "offline (physics prior render)"

    def render(self, initial: Frame, sc, prompt: str):
        if self.client:
            try:
                h, w = initial.rgb.shape[:2]
                # upscale the 320x240 sim frame so the editor has detail to keep; 4:3 is passed as size
                big = np.asarray(Image.fromarray(initial.rgb).resize((w * 3, h * 3), Image.BICUBIC))
                data = self.client.post(settings.image_endpoint, {
                    "model": settings.image_model,
                    "prompt": prompt + " Keep the camera angle, lighting, colors and every unmentioned object "
                                       "exactly as they are. Photorealistic simulator render, no text.",
                    "images": [{"image_url": png_data_url(big)}],
                    "n": 1,
                    "size": f"{w * 3}x{h * 3}",
                    "output_format": "png",
                    "reasoning_strength": "low",
                    "tool_enablement": {"enable_image_search": False, "enable_web_search": False,
                                        "enable_shell": False},
                })
                hit = _first_image(data)
                if not hit:
                    raise RuntimeError(f"no image in response keys {list(data)[:6]}")
                rgb = fetch_image(hit[1]) if hit[0] == "url" else decode_image(hit[1])
                if rgb.shape[:2] != (h, w):
                    rgb = np.asarray(Image.fromarray(rgb).resize((w, h), Image.BILINEAR))
                prior = imagine_goal(sc)  # depth is not generated, borrow it from the prior
                frame = Frame(rgb=rgb, depth=prior.depth, view=initial.view, proj=initial.proj,
                              labels=initial.labels, seg=None, seg_ids={}, tag="expected")
                return frame, self.name, True
            except Exception as e:
                self.last_error = str(e)[:300]
                log.warning("Muse Image failed (%s), using physics prior render", e)
        return imagine_goal(sc), "offline (physics prior render)", False
