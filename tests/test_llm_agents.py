import json

from conftest import FakeLLM

from context_engine import BY_KEY, run_scenario
from context_engine.memory import UserMemory
from context_engine.profiler import ProfilerAgent
from context_engine.edge import EdgeLayer
from context_engine.simulator import generate


def test_llm_failure_falls_back_to_rules_and_still_passes():
    llm = FakeLLM(fail=True)
    result = run_scenario(BY_KEY["meditation_beads"], llm=llm)
    assert result.passed
    assert {a.by for a in result.actions if a.tool != "log_event"} == {"Action Agent (rules)"}
    assert llm.failures > 0


def test_unknown_tool_from_llm_is_ignored():
    llm = FakeLLM(tool_script=[[("format_hard_drive", {"reason": "oops"})]])
    result = run_scenario(BY_KEY["free_walk"], llm=llm)
    assert "format_hard_drive" not in result.called


def test_llm_tool_calls_are_executed_with_reasons():
    script = [
        [("do_nothing", {"reason": "first look"})],
        [("send_notification", {"message": "WhatsApp: Picked up the groceries", "priority": "medium",
                                "reason": "user is free"})],
    ]
    result = run_scenario(BY_KEY["free_walk"], llm=FakeLLM(tool_script=script))
    sent = [a for a in result.actions if a.tool == "send_notification"]
    assert sent and sent[0].by == "Action Agent (LLM)" and sent[0].reason == "user is free"


def test_follow_up_round_after_changing_watch_mode():
    """DND off unlocks 'deliver held messages': the agent gets one more look."""
    script = [
        [("do_nothing", {"reason": "first look"})],
        [("enable_dnd", {"reason": "focus app open"})],
        [("do_nothing", {"reason": "nothing more"})],                       # follow-up after enable_dnd
        [("defer_notification", {"until": "focus_ends", "reason": "focused"})],
        [("disable_dnd", {"reason": "focus over"})],
        [("deliver_deferred", {"reason": "user free with held message"})],  # follow-up after disable_dnd
    ]
    result = run_scenario(BY_KEY["meditation_beads"], llm=FakeLLM(tool_script=script))
    assert result.passed, (result.missing, result.mistakes)


def test_prompt_carries_precomputed_signals():
    llm = FakeLLM()
    run_scenario(BY_KEY["meditation_beads"], llm=llm)
    signals = [json.loads(p)["signals"] for p in llm.prompts]
    assert any(s["focus_app_open"] for s in signals)
    assert {"dnd_on", "navigating", "user_free", "held_messages"} <= set(signals[0])


def test_profiler_keeps_valid_llm_fields_and_gps_place():
    edge = EdgeLayer(UserMemory())
    features = None
    for r in generate(BY_KEY["free_walk"]):
        features, _ = edge.update(r)
    llm = FakeLLM(profile={"activity": "WALKING", "stress": "extreme", "attention": "free",
                           "summary": "User is strolling", "confidence": 3, "place": "paris"})
    state = ProfilerAgent(llm).profile(features, ["test"])
    assert state.activity == "walking"          # case-insensitive
    assert state.stress == "low"                # invalid value -> rule value kept
    assert state.place == "known"               # place always from GPS + memory
    assert state.confidence == 1.0              # clamped
    assert state.summary == "User is strolling" and state.source == "llm"
