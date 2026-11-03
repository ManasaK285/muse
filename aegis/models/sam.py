import logging
import os
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from aegis.config import settings
from aegis.models.meta_client import MetaClient, png_data_url
from aegis.perception.frames import Frame

log = logging.getLogger("aegis.sam")


class OfflineSegmenter:
    # no key configured: use the renderer's instance buffer, pixel exact and reproducible
    name = "offline (renderer instance masks)"
    short = "renderer"

    def segment(self, frame: Frame, only=None) -> dict:
        if frame.seg is None:
            raise ValueError("offline segmenter needs the renderer's segmentation buffer")
        out = {}
        for name, uid in frame.seg_ids.items():
            if name not in frame.labels or (only and name not in only):
                continue
            m = (frame.seg & ((1 << 24) - 1)) == uid
            out[name] = {"mask": m, "score": 1.0, "prompt": frame.labels[name], "source": "renderer"}
        return out


class SAM31:
    name = f"SAM 3.1 (Model API, {settings.sam_model})"
    short = "SAM 3.1"

    def __init__(self):
        try:
            import meta_sam_parser  # noqa: F401
        except ImportError as e:
            raise RuntimeError("live SAM needs Meta's parser: pip install meta-sam-parser") from e
        self.client = MetaClient()
        self.pool = ThreadPoolExecutor(max_workers=6)
        self.lock = threading.Lock()
        self.stats = {"calls": 0, "hits": Counter(), "misses": Counter(), "alt_hits": Counter()}
        self.debug_dir = Path(os.environ.get("AEGIS_SAM_DEBUG", Path(__file__).resolve().parents[2] / "runs" / "sam_debug"))
        self.debug_saved = 0
        self.preferred = {}  # object -> phrase that last worked, tried first from then on

    def request(self, rgb, prompt):
        # shape from the Media segmentation guide: one message, noun phrase first, then the image
        return {
            "model": settings.sam_model,
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": png_data_url(rgb)},
            ]}],
        }

    def _call(self, big, h, w, prompt):
        text, status = self.client.stream_output_text(settings.sam_endpoint, self.request(big, prompt))
        with self.lock:
            self.stats["calls"] += 1
        if not text.strip():
            return [], text  # zero matches is a valid answer
        return parse_output(text, h, w, status), text

    def _one(self, rgb, name, prompt):
        from aegis.scenarios import ALT_PROMPTS

        h, w = rgb.shape[:2]
        big = np.asarray(Image.fromarray(rgb).resize((w * 2, h * 2), Image.BICUBIC))  # small sim frames
        raw = ""
        phrases = [prompt] + ALT_PROMPTS.get(name, [])
        if self.preferred.get(name) in phrases:
            phrases.remove(self.preferred[name])
            phrases.insert(0, self.preferred[name])
        for i, phrase in enumerate(phrases):
            matches, raw = self._call(big, h, w, phrase)
            if matches:
                best = max(matches, key=lambda m: m["mask"].sum())  # several hits: keep the biggest
                best.update(prompt=phrase, matches=len(matches), source="SAM 3.1")
                with self.lock:
                    self.stats["hits"][name] += 1
                    self.preferred[name] = phrase
                    if phrase != prompt:
                        self.stats["alt_hits"][f"{name} <- {phrase!r}"] += 1
                return name, best
        with self.lock:
            self.stats["misses"][name] += 1
            save = self.debug_saved < 24
            self.debug_saved += save
        if save:
            # keep a few misses so the prompt or the frame can be inspected later
            try:
                self.debug_dir.mkdir(parents=True, exist_ok=True)
                stem = f"miss_{self.debug_saved:02d}_{name}"
                Image.fromarray(big).save(self.debug_dir / f"{stem}.png")
                (self.debug_dir / f"{stem}.txt").write_text(f"prompts: {[prompt] + ALT_PROMPTS.get(name, [])}\nraw: {raw[:2000]!r}\n", encoding="utf-8")
            except OSError:
                pass
        return name, None

    def summary(self):
        s = self.stats
        hits, misses = sum(s["hits"].values()), sum(s["misses"].values())
        return {"calls": s["calls"], "objects_found": hits, "objects_missed": misses,
                "hit_rate": round(hits / max(hits + misses, 1), 3), "missed_by_object": dict(s["misses"]),
                "found_with_alt_phrase": dict(s["alt_hits"])}

    def segment(self, frame: Frame, only=None) -> dict:
        targets = {n: t for n, t in frame.labels.items() if n != "robot" and (not only or n in only)}
        futs = [self.pool.submit(self._one, frame.rgb, n, t) for n, t in targets.items()]
        out = {}
        for f in futs:
            name, best = f.result()
            if best is not None:
                out[name] = best
        return out


def parse_output(text, h, w, status="completed"):
    # SAM 3.1 special-token lines -> full-frame boolean masks, via Meta's meta_sam_parser
    import meta_sam_parser as msp

    parser = msp.image_segmentation_format().create_parser()
    parser.push(text)
    outcome = msp.CompletedOutcome() if status == "completed" else msp.IncompleteOutcome(reason="eof")
    result = parser.finish(outcome).result
    boxes = {r.object_id: r for r in result.records if isinstance(r, msp.SegmentationBoxRecord)}
    src = text_frame_size(text) or (w, h)
    sx, sy = w / src[0], h / src[1]  # bounds are in the frame SAM saw; rescale if ours differs
    out = []
    for rec in result.records:
        if not isinstance(rec, msp.SegmentationMaskRecord):
            continue
        local = np.frombuffer(msp.decode_mask_to_raster(rec.mask), dtype=np.uint8)
        local = local.reshape(rec.mask.height, rec.mask.width).astype(bool)
        b = rec.bounds
        x0, y0 = max(int(round(b.left * sx)), 0), max(int(round(b.top * sy)), 0)
        x1, y1 = min(int(round(b.right * sx)), w), min(int(round(b.bottom * sy)), h)
        if x1 <= x0 or y1 <= y0:
            continue
        if local.shape != (y1 - y0, x1 - x0):
            img = Image.fromarray(local.astype(np.uint8) * 255).resize((x1 - x0, y1 - y0), Image.NEAREST)
            local = np.asarray(img) > 127
        full = np.zeros((h, w), dtype=bool)
        full[y0:y1, x0:x1] = local
        box = boxes.get(rec.object_id)
        out.append({"mask": full, "object_id": rec.object_id,
                    "score": float(rec.confidence) if rec.confidence is not None else 1.0,
                    "box": [box.left, box.top, box.right, box.bottom] if box else None})
    if not out and result.diagnostics:
        d = result.diagnostics[0]
        raise RuntimeError(f"SAM 3.1 output not parseable ({d.code}: {d.message}): {text[:300]!r}")
    return out


def text_frame_size(text):
    m = re.search(r";w=(\d+);h=(\d+)", text)
    return (int(m.group(1)), int(m.group(2))) if m else None


def get_segmenter():
    if settings.backend("sam") == "live":
        try:
            return SAM31()
        except Exception as e:
            log.warning("SAM 3.1 unavailable (%s), using offline segmenter", e)
    return OfflineSegmenter()


class FallbackSegmenter:
    def __init__(self):
        self.primary = get_segmenter()
        self.backup = OfflineSegmenter()
        self.last_error = None
        self.fallbacks = 0

    @property
    def name(self):
        return self.primary.name

    @property
    def short(self):
        return self.primary.short

    def summary(self):
        return self.primary.summary() if hasattr(self.primary, "summary") else None

    def segment(self, frame, only=None):
        if isinstance(self.primary, OfflineSegmenter):
            return self.primary.segment(frame, only)
        try:
            out = self.primary.segment(frame, only)
            wanted = [n for n in frame.labels if n != "robot" and (not only or n in only)]
            missing = [n for n in wanted if n not in out]
            if missing and frame.seg is not None:
                # SAM found nothing for these phrases; a missing mask must not read as "object moved"
                filled = self.backup.segment(frame, missing)
                for n, m in filled.items():
                    out[n] = {**m, "source": "renderer"}
                self.fallbacks += len(filled)
                self.last_error = f"SAM 3.1 found no match for {', '.join(missing)} on a {frame.tag or 'frame'}"
            if out:
                return out
            self.last_error = "SAM 3.1 returned no masks"
        except Exception as e:
            self.last_error = str(e)[:300]
            log.warning("SAM call failed: %s", e)
        if frame.seg is not None:
            self.fallbacks += 1
            return self.backup.segment(frame, only)
        return {}
