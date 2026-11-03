import pytest

from aegis.graph.mission import run_mission


@pytest.mark.slow
@pytest.mark.parametrize("key", ["reward_hack", "collision", "slip"])
def test_mission_recovers(key):
    events = []
    _, summary = run_mission(key, chaos=True, emit=lambda k, d: events.append(k))
    assert summary["naive_verdict"] == "SUCCESS (Exit 0)"
    assert summary["verdict"] == "VERIFIED"
    assert summary["recovery_cycles"] >= 1
    assert "diagnosis" in events and "audit" in events
