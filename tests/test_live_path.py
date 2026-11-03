import json

import pytest

from aegis.config import settings
from aegis.models import meta_client, policies


@pytest.fixture
def fake_api(monkeypatch):
    calls = []

    def fake_chat(self, model, messages, json_mode=False, max_tokens=4096):
        system = messages[0]["content"]
        user = messages[-1]["content"]
        calls.append((model, system[:40]))
        if "plan robot manipulation" in system:
            return json.dumps({"subgoals": [{"name": "pre_grasp", "target": "red_vial", "intent": "hover"}],
                               "success_condition": "vial in tray"})
        if "AFTER the task" in system:
            return json.dumps({"objects": {"red_vial": {"centroid": [0.59, -0.243, 0.06], "must_not_move": False,
                                                        "tolerance_m": 0.04, "upright": True},
                                           "glass_beaker": {"centroid": [0.58, -0.04, 0.08], "must_not_move": True,
                                                            "tolerance_m": 0.002, "upright": True}},
                               "invariants": ["beaker unmoved"], "image_prompt": "vial in tray"})
        if "forecast what will actually happen" in system:
            outcome = "MICRO_COLLISION" if "CLEARANCE_MARGIN_M = 0.000" in user else "SUCCESS"
            return json.dumps({"predicted_outcome": outcome, "confidence": 0.8, "reasoning": "r", "risky_lines": []})
        if "failure analyst" in system:
            return json.dumps({"failure_mode": "MICRO_COLLISION", "root_cause": "path clips the beaker",
                               "evidence": [], "patch_directives": [{"type": "raise_clearance", "margin_m": 0.05}]})
        if "failed verification" in user:
            return "Here is the fix:\n```python\n" + policies.render(policies.SAFE_DEFAULTS) + "```"
        return "```python\n" + policies.render(policies.NAIVE["collision"]) + "```"

    monkeypatch.setattr(settings, "api_key", "test-key")
    monkeypatch.setenv("AEGIS_SPARK_BACKEND", "live")
    monkeypatch.setenv("AEGIS_CODE_BACKEND", "live")
    monkeypatch.setenv("AEGIS_SAM_BACKEND", "offline")
    monkeypatch.setenv("AEGIS_WONDER_BACKEND", "live")
    monkeypatch.setenv("AEGIS_IMAGE_BACKEND", "offline")
    monkeypatch.setattr(meta_client.MetaClient, "chat", fake_chat)
    return calls


@pytest.mark.slow
def test_live_models_drive_the_loop(fake_api):
    from aegis.graph.mission import run_mission

    events = []
    _, summary = run_mission("collision", chaos=False, emit=lambda k, d: events.append((k, d)))
    used = {c[0] for c in fake_api}
    assert settings.spark_model in used and settings.wonder_model in used
    assert summary["verdict"] == "VERIFIED"
    assert summary["recovery_cycles"] == 1
    scores = [d for k, d in events if k == "forecast_score"]
    assert [s["hit"] for s in scores] == [True, True]  # collision forecast, then success forecast
    wonder = next(d for k, d in events if k == "wonder")
    assert wonder["world_model"].startswith("Muse Wonder")


def test_extractors():
    assert meta_client.extract_json('noise ```json\n{"a": 1}\n``` tail') == {"a": 1}
    code = meta_client.extract_python("```python\nx = 1\n```\n```python\ndef execute_policy(env):\n    pass\n```")
    assert code.startswith("def execute_policy")
