import copy

from context_engine import BY_KEY, run_scenario


def _tools(result):
    return [a.tool for a in result.actions]


def test_spike_while_still_triggers_emergency_once_then_alerts_contact():
    tools = _tools(run_scenario(BY_KEY["hr_spike_still"]))
    assert tools.count("alert_emergency_contact") == 1
    assert "start_emergency_protocol" in tools
    starts = [a for a in run_scenario(BY_KEY["hr_spike_still"]).actions
              if a.tool == "start_emergency_protocol" and a.result.startswith("Countdown")]
    assert len(starts) == 1


def test_safety_guard_acts_without_any_agent():
    by = {a.by for a in run_scenario(BY_KEY["hr_spike_still"]).actions if a.tool == "start_emergency_protocol"}
    assert "Safety Guard" in by


def test_running_never_triggers_emergency():
    assert "start_emergency_protocol" not in _tools(run_scenario(BY_KEY["hr_high_running"]))


def test_user_can_cancel_with_im_ok():
    sc = copy.deepcopy(BY_KEY["hr_spike_still"])
    sc.phases[1].user_inputs = {30: "im_ok"}  # a few seconds into the countdown
    tools = _tools(run_scenario(sc))
    assert "cancel_emergency" in tools
    assert "alert_emergency_contact" not in tools
