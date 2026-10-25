import pytest

from context_engine import SCENARIOS, run_scenario


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.key)
def test_rules_mode_makes_the_right_calls(scenario):
    result = run_scenario(scenario)  # no LLM: the rules version of both agents
    if scenario.needs_llm:
        assert not result.passed, "rules cannot read message text; this story documents why the LLM matters"
    else:
        assert result.passed, f"missing={result.missing} mistakes={result.mistakes}"
        assert result.false_alarms == 0


def test_every_action_has_a_reason():
    for scenario in SCENARIOS:
        for a in run_scenario(scenario).actions:
            assert a.reason and a.reason != "(no reason given)", a
