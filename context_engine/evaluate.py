"""Run scenarios and score them: did the engine make the right calls?"""
from dataclasses import dataclass
from typing import Callable

from .engine import ContextEngine, StepResult, Stats
from .memory import UserMemory
from .schemas import Action
from .simulator import Scenario, generate

# Tools that interrupt the user. Using one by mistake counts as a false alarm.
INTERRUPTING = {"send_notification", "suggest_breathing", "start_emergency_protocol", "alert_emergency_contact"}


@dataclass
class ScenarioResult:
    scenario: Scenario
    actions: list[Action]
    stats: Stats
    mode: str

    @property
    def called(self) -> set[str]:
        return {a.tool for a in self.actions}

    @property
    def missing(self) -> set[str]:
        return self.scenario.must_call - self.called

    @property
    def mistakes(self) -> set[str]:
        return self.scenario.must_not_call & self.called

    @property
    def false_alarms(self) -> int:
        return sum(1 for a in self.actions if a.tool in self.scenario.must_not_call and a.tool in INTERRUPTING)

    @property
    def passed(self) -> bool:
        return not self.missing and not self.mistakes


def run_scenario(scenario: Scenario, llm=None, memory: UserMemory | None = None,
                 on_step: Callable[[StepResult, ContextEngine], None] | None = None,
                 log_path=None) -> ScenarioResult:
    engine = ContextEngine(memory=memory, llm=llm, log_path=log_path)
    for reading in generate(scenario):
        step = engine.process(reading)
        if on_step:
            on_step(step, engine)
    return ScenarioResult(scenario, engine.actions, engine.stats, engine.mode)
