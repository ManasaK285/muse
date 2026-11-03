import base64
import io
import json
import time
from pathlib import Path

from PIL import Image

RUNS = Path(__file__).resolve().parents[1] / "runs"


def _gif(frames_b64, path, fps=15):
    if not frames_b64:
        return
    imgs = [Image.open(io.BytesIO(base64.b64decode(f))).convert("P", palette=Image.ADAPTIVE)
            for f in frames_b64]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)


def _img(b64, path):
    path.write_bytes(base64.b64decode(b64))


class Recorder:
    def __init__(self, scenario, save_media=True):
        self.dir = RUNS / f"{time.strftime('%Y%m%d_%H%M%S')}_{scenario}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.save_media = save_media
        self.log = open(self.dir / "events.jsonl", "w", encoding="utf-8")

    def __call__(self, kind, data):
        slim = {k: v for k, v in data.items() if k not in ("frames",)}
        if kind in ("perception", "wonder"):
            slim = {k: v for k, v in slim.items() if not (isinstance(v, str) and len(v) > 500)}
        if kind == "audit":
            slim = dict(slim)
            slim["per_seed"] = [{k: v for k, v in s.items() if k != "images"} for s in data["per_seed"]]
        self.log.write(json.dumps({"t": time.time(), "kind": kind, "data": slim}, default=str) + "\n")
        self.log.flush()
        if not self.save_media:
            return
        try:
            if kind == "code":
                (self.dir / f"policy_iter{data['iteration']}.py").write_text(data["code"], encoding="utf-8")
                if data.get("diff"):
                    (self.dir / f"patch_iter{data['iteration']}.diff").write_text(data["diff"], encoding="utf-8")
            elif kind in ("rollout", "naive", "redteam_probe"):
                name = "naive" if kind == "naive" else f"iter{data['iteration']}_seed{data['seed']}"
                if kind == "redteam_probe":
                    name = "redteam_probe"
                _gif(data.get("frames"), self.dir / f"{name}.gif")
            elif kind == "wonder":
                _img(data["expected"], self.dir / "wonder_expected.jpg")
            elif kind == "perception":
                _img(data["overlay"], self.dir / "sam_initial_overlay.jpg")
            elif kind == "audit":
                for s in data["per_seed"]:
                    for k, v in (s.get("images") or {}).items():
                        _img(v, self.dir / f"audit_iter{data['iteration']}_seed{s['seed']}_{k}.jpg")
            elif kind == "verdict":
                (self.dir / "summary.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        except Exception as e:  # media is a nice to have, never fail a run over it
            self.log.write(json.dumps({"kind": "recorder_error", "error": str(e)}) + "\n")

    def close(self):
        self.log.close()
