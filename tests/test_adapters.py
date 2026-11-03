import base64
import io

import numpy as np
from PIL import Image

from aegis.models import muse_image


# the image line from Meta's "Read segmentation output" page, two objects in a 320x334 frame
DOC_LINE = ("<0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334|><|mask;x=0;y=0;data=27,60,~!!!!M!0c[0o91w?q1!pIH4pPRVp2B3'7`e.ioeAf6-k/#Xd8%dX9x(|>,"
            "1<|box;x1=155;y1=228;x2=202;y2=254;w=320;h=334|><|mask;x=0;y=0;data=27,48,~!!!!J!0c[q=Pj_zs=*4C4(/./x#:/`S_GnD`=o3?X{emCgO$y@|>")


def test_sam_grammar_decodes_with_meta_parser():
    from aegis.models.sam import parse_output

    masks = parse_output(DOC_LINE, 334, 320)
    assert [m["object_id"] for m in masks] == ["0", "1"]
    m0 = masks[0]["mask"]
    ys, xs = np.nonzero(m0)
    assert xs.min() >= 211 and xs.max() <= 270 and ys.min() >= 228 and ys.max() <= 254
    assert m0.sum() == 1539


def test_sam_rescales_when_frame_size_differs():
    from aegis.models.sam import parse_output

    masks = parse_output(DOC_LINE, 167, 160)  # same image at half size
    xs = np.nonzero(masks[0]["mask"])[1]
    assert 104 <= xs.min() <= 107 and xs.max() <= 136


def test_sam_stream_end_to_end(monkeypatch):
    from aegis.config import settings
    from aegis.models import meta_client, sam
    from aegis.perception.frames import Frame

    sent = {}

    def fake_stream(self, url, payload):
        sent.update(url=url, payload=payload)
        return (DOC_LINE if payload["input"][0]["content"][0]["text"] == "red vial" else ""), "completed"

    monkeypatch.setattr(settings, "api_key", "k")
    monkeypatch.setattr(meta_client.MetaClient, "stream_output_text", fake_stream)
    frame = Frame(rgb=np.zeros((334, 320, 3), np.uint8), depth=np.ones((334, 320), np.float32), view=[], proj=[],
                  labels={"red_vial": "red vial", "glass_beaker": "glass beaker", "robot": "robot arm"})
    out = sam.SAM31().segment(frame)
    assert set(out) == {"red_vial"}  # zero matches for the beaker is a valid empty answer
    assert out["red_vial"]["matches"] == 2 and out["red_vial"]["mask"].sum() == 1539
    assert sent["url"].endswith("/responses")
    assert sent["payload"]["input"][0]["type"] == "message"


def test_muse_image_response_shapes():
    assert muse_image._first_image({"data": [{"b64_json": "QUJD"}]}) == ("b64", "QUJD")
    assert muse_image._first_image({"data": [{"url": "https://x/y.png"}]}) == ("url", "https://x/y.png")
    assert muse_image._first_image({"images": ["QUJD"]}) == ("b64", "QUJD")
    assert muse_image._first_image({"data": []}) is None


def test_muse_image_edit_uses_json_body(monkeypatch):
    from aegis import scenarios
    from aegis.config import settings
    from aegis.models import meta_client
    from aegis.models.wonder import imagine_goal

    sent = {}

    def fake_post(self, url, payload=None, retries=3, files=None, data=None):
        sent.update(url=url, payload=payload, files=files)
        buf = io.BytesIO()
        Image.new("RGB", (960, 720), (40, 40, 40)).save(buf, format="PNG")
        return {"created": 1, "data": [{"b64_json": base64.b64encode(buf.getvalue()).decode()}]}

    monkeypatch.setattr(settings, "api_key", "k")
    monkeypatch.setenv("AEGIS_IMAGE_BACKEND", "live")
    monkeypatch.setattr(meta_client.MetaClient, "post", fake_post)
    sc = scenarios.get("collision")
    initial = imagine_goal(sc)
    frame, source, generative = muse_image.GoalFrameRenderer().render(initial, sc, "vial in tray")
    assert generative and source.startswith("Muse Image")
    assert sent["files"] is None and sent["url"].endswith("/images/edits")
    assert sent["payload"]["images"][0]["image_url"].startswith("data:image/png;base64,")
    assert set(sent["payload"]) <= {"model", "prompt", "images", "n", "size", "output_format",
                                    "reasoning_strength", "tool_enablement"}
    assert frame.rgb.shape == initial.rgb.shape


def test_sse_reader_collects_output_text(monkeypatch):
    import json

    import httpx

    from aegis.config import settings
    from aegis.models import meta_client

    events = [{"type": "response.created"},
              {"type": "response.output_text.delta", "delta": DOC_LINE[:90]},
              {"type": "response.output_text.delta", "delta": DOC_LINE[90:]},
              {"type": "response.completed"}]
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    seen = {}

    def handler(request):
        seen["json"] = json.loads(request.content)
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(settings, "api_key", "k")
    client = meta_client.MetaClient()
    client.http = httpx.Client(transport=httpx.MockTransport(handler))
    text, status = client.stream_output_text("https://api.meta.ai/v1/responses", {"model": "sam-3.1", "input": []})
    assert text == DOC_LINE and status == "completed"
    assert seen["json"]["stream"] is True


def test_sam_miss_is_filled_from_renderer(monkeypatch):
    from aegis import scenarios
    from aegis.config import settings
    from aegis.models import meta_client, sam
    from aegis.models.wonder import imagine_goal

    monkeypatch.setattr(settings, "api_key", "k")
    monkeypatch.setenv("AEGIS_SAM_BACKEND", "live")
    monkeypatch.setattr(meta_client.MetaClient, "stream_output_text", lambda self, url, payload: ("", "completed"))
    seg = sam.FallbackSegmenter()
    frame = imagine_goal(scenarios.get("collision"))
    out = seg.segment(frame)
    assert {"red_vial", "glass_beaker", "containment_tray"} <= set(out)
    assert out["glass_beaker"]["source"].startswith("renderer")
    assert seg.fallbacks >= 3 and "no match" in seg.last_error


def test_chat_json_retries_once_on_bad_json():
    from aegis.models.meta_client import chat_json

    replies = iter(['{"subgoals": [1, 2', '{"subgoals": [1, 2]}'])

    class C:
        calls = []

        def chat(self, model, messages, json_mode=False, max_tokens=0):
            self.calls.append(len(messages))
            return next(replies)

    c = C()
    assert chat_json(c, "m", "sys", "user") == {"subgoals": [1, 2]}
    assert c.calls == [2, 4]  # second call carries the bad reply and the correction


def test_sam_retries_alternate_phrase_and_counts(monkeypatch):
    from aegis.config import settings
    from aegis.models import meta_client, sam
    from aegis.perception.frames import Frame

    asked = []

    def fake_stream(self, url, payload):
        phrase = payload["input"][0]["content"][0]["text"]
        asked.append(phrase)
        return (DOC_LINE if phrase == "light blue cylinder" else ""), "completed"

    monkeypatch.setattr(settings, "api_key", "k")
    monkeypatch.setenv("AEGIS_SAM_DEBUG", str(__import__("tempfile").mkdtemp()))
    monkeypatch.setattr(meta_client.MetaClient, "stream_output_text", fake_stream)
    frame = Frame(rgb=np.zeros((334, 320, 3), np.uint8), depth=np.ones((334, 320), np.float32), view=[], proj=[],
                  labels={"glass_beaker": "glass beaker", "red_vial": "red vial"})
    seg = sam.SAM31()
    out = seg.segment(frame)
    assert out["glass_beaker"]["prompt"] == "light blue cylinder" and out["glass_beaker"]["source"] == "SAM 3.1"
    st = seg.summary()
    assert st["objects_found"] == 1 and st["missed_by_object"] == {"red_vial": 1}
    assert "glass_beaker <- 'light blue cylinder'" in st["found_with_alt_phrase"]
