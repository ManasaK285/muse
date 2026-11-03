import json
import sys

import pytest

from aegis.config import settings
from aegis.models import meta_client


def test_glimmer_takes_over_when_cloud_spark_fails(monkeypatch):
    from aegis.models.spark import Spark

    monkeypatch.setattr(settings, "api_key", "k")
    monkeypatch.setattr(settings, "glimmer_base", "http://localhost:8080/v1")
    monkeypatch.setenv("AEGIS_SPARK_BACKEND", "live")

    def cloud_down(self, *a, **k):
        raise meta_client.MetaAPIError("503 from api.meta.ai")

    def edge(self, messages, **k):
        return json.dumps({"subgoals": [{"name": "pre_grasp", "target": "red_vial"}], "success_condition": "x"})

    monkeypatch.setattr(meta_client.MetaClient, "chat", cloud_down)
    monkeypatch.setattr(meta_client.CompatClient, "chat", edge)
    sp = Spark()
    plan = sp.decompose("move the vial", {"objects": {}})
    assert plan["subgoals"][0]["name"] == "pre_grasp"
    assert sp.name.startswith("Muse Glimmer edge")


def test_rules_are_last_resort_without_any_model(monkeypatch):
    from aegis.models.spark import Spark

    monkeypatch.setattr(settings, "api_key", "")
    monkeypatch.setattr(settings, "glimmer_base", "")
    sp = Spark()
    plan = sp.decompose("x", {"objects": {}, "payload": "a", "tray": "b"})
    assert sp.name == "offline planner" and len(plan["subgoals"]) == 4


def test_spark_open_attacker_parses_generated_attacks(monkeypatch):
    from aegis import scenarios
    from aegis.redteam import Attacker

    attack = {"name": "sneaky", "strategy": "s", "code": "import os\n\ndef execute_policy(env):\n    os.getcwd()\n"}
    monkeypatch.setattr(settings, "spark_open_base", "https://api.fireworks.ai/inference/v1")
    monkeypatch.setattr(settings, "spark_open_model", "accounts/meta/models/muse-spark-open")
    monkeypatch.setattr(meta_client.CompatClient, "chat",
                        lambda self, messages, **k: json.dumps({"attacks": [attack, {"name": "junk"}]}))
    a = Attacker()
    attacks, source = a.generate(scenarios.get("collision"), 3)
    assert source.startswith("Muse Spark Open") and [x["name"] for x in attacks] == ["sneaky"]


def test_voice_transcript_shapes_and_multipart(monkeypatch):
    from aegis.models import voice

    sent = {}

    def fake_post(self, url, payload=None, retries=3, files=None, data=None):
        sent.update(url=url, files=files, data=data)
        return {"turns": [{"text": "pick up the red vial"}, {"text": "and put it in the tray"}]}

    monkeypatch.setattr(settings, "api_key", "k")
    monkeypatch.setenv("AEGIS_VOICE_BACKEND", "live")
    monkeypatch.setattr(meta_client.MetaClient, "post", fake_post)
    text, backend = voice.transcribe(b"RIFF....", "cmd.wav")
    assert text == "pick up the red vial and put it in the tray"
    assert sent["url"].endswith("/asr/transcribe") and sent["files"]["file"][2] == "audio/wav"
    assert sent["data"]["model"] == settings.voice_model
    assert voice._transcript({"text": " hello "}) == "hello"


def test_voice_endpoint_without_key_is_a_clean_400(monkeypatch):
    from fastapi.testclient import TestClient

    from aegis.server.app import app

    monkeypatch.setattr(settings, "api_key", "")
    monkeypatch.delenv("AEGIS_VOICE_BACKEND", raising=False)
    r = TestClient(app).post("/api/transcribe", content=b"RIFF0000", headers={"x-filename": "a.wav"})
    assert r.status_code == 400 and "MODEL_API_KEY" in r.json()["error"]


def test_muse_code_agent_edits_policy_file(monkeypatch, tmp_path):
    from aegis.models import policies
    from aegis.models.muse_code import MuseCode

    # a stand-in for the muse CLI: reads TASK.md like the real agent would, rewrites policy.py
    fake = tmp_path / "fake_muse.py"
    fake.write_text(
        "import pathlib\n"
        "task = pathlib.Path('TASK.md').read_text()\n"
        "assert 'execute_policy' in task\n"
        f"pathlib.Path('policy.py').write_text({policies.render(policies.SAFE_DEFAULTS)!r})\n")
    monkeypatch.setattr(settings, "code_cmd", f'"{sys.executable}" "{fake}" "{{task}}"')
    mc = MuseCode()
    assert mc.mode == "cli" and mc.name == "Muse Code (CLI)"
    new, params = mc.patch(policies.render(policies.NAIVE["collision"]),
                           {"failure_mode": "MICRO_COLLISION", "patch_directives": []}, None, {})
    assert "CLEARANCE_MARGIN_M = 0.040" in new and params is None


@pytest.mark.slow
def test_mission_with_red_team():
    from aegis.graph.mission import run_mission

    events = []
    _, summary = run_mission("reward_hack", chaos=True, emit=lambda k, d: events.append((k, d)), redteam=True)
    red = next(d for k, d in events if k == "redteam")
    assert summary["verdict"] == "VERIFIED"
    assert red["total"] == settings.redteam_n and red["blocked"] == red["total"]
    assert all(a["caught_by"] for a in red["attacks"])
    stack = next(d for k, d in events if k == "mission_start")["stack"]
    assert len(stack) == 8


def test_muse_exec_template_keeps_task_and_dir_whole():
    from aegis.models.muse_code import build_argv

    tpl = 'muse --approval-mode never --disable-sandbox --workspace "{dir}" exec "{task}"'
    argv = build_argv(tpl, "Read TASK.md and rewrite policy.py.", "/tmp/aegis muse 1")
    assert argv[-2:] == ["exec", "Read TASK.md and rewrite policy.py."]
    assert argv[argv.index("--workspace") + 1] == "/tmp/aegis muse 1"


def test_failed_muse_code_is_not_credited(monkeypatch):
    from aegis.models import policies
    from aegis.models.muse_code import MuseCode

    monkeypatch.setattr(settings, "code_cmd", "definitely-not-a-real-binary-xyz exec \"{task}\"")
    mc = MuseCode()
    new, params = mc.patch(policies.render(policies.NAIVE["collision"]),
                           {"patch_directives": [{"type": "raise_clearance", "margin_m": 0.04}]}, dict(policies.NAIVE["collision"]), {})
    assert params is not None and "CLEARANCE_MARGIN_M = 0.040" in new
    assert mc.last_source.startswith("offline policy writer (Muse Code (CLI) failed")


def test_windows_style_template_splits_cleanly():
    import shlex

    from aegis.models.muse_code import build_argv

    tpl = ('muse --approval-mode never --disable-sandbox --disable-request-user-input --reasoning-effort medium '
           '--contributor-mode off --workspace "{dir}" exec "{task}"')
    monkey = shlex.split
    try:
        shlex.split = lambda s, posix=True: monkey(s, posix=False)  # what Windows does
        argv = build_argv(tpl, "Read TASK.md and rewrite policy.py so it satisfies it. Edit only policy.py.",
                          r"C:\Users\Manasa\AppData\Local\Temp\aegis_muse_ab12")
    finally:
        shlex.split = monkey
    assert argv[-1] == "Read TASK.md and rewrite policy.py so it satisfies it. Edit only policy.py."
    assert argv[argv.index("--workspace") + 1] == r"C:\Users\Manasa\AppData\Local\Temp\aegis_muse_ab12"
