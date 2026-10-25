"""The context engine: wires every block together, one reading at a time.

    reading -> Edge Layer -> (on change) Profiler Agent -> Action Agent -> Tools
                   \\-> Safety Guard (every reading, overrides the agents)
"""
import json
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from .action_agent import ActionAgent, DecisionContext
from .edge import EdgeLayer
from .memory import UserMemory
from .profiler import ProfilerAgent
from .safety import SafetyGuard
from .schemas import Action, Features, Reading, UserState
from .tools import Watch, run_tool

STATE_CHANGING_TOOLS = {"enable_dnd", "disable_dnd", "battery_saver_mode"}
FOLLOW_UP_ROUNDS = 1


@dataclass
class StepResult:
    reading: Reading
    features: Features | None
    state: UserState | None
    changes: list[str]
    actions: list[Action]


@dataclass
class Stats:
    readings: int = 0
    profiler_runs: int = 0
    action_runs: int = 0
    llm_calls: int = 0
    llm_failures: int = 0
    decision_ms: list[float] = field(default_factory=list)

    @property
    def agent_runs(self) -> int:
        return self.profiler_runs + self.action_runs

    @property
    def calls_saved_pct(self) -> float:
        """Without the Edge Layer, both agents would run on every reading."""
        naive = self.readings * 2
        return 100 * (1 - self.agent_runs / naive) if naive else 0.0

    @property
    def avg_decision_ms(self) -> float:
        return sum(self.decision_ms) / len(self.decision_ms) if self.decision_ms else 0.0


class ContextEngine:
    def __init__(self, memory: UserMemory | None = None, llm=None, log_path: str | Path | None = None):
        self.memory = memory or UserMemory()
        self.llm = llm
        self.edge = EdgeLayer(self.memory)
        self.safety = SafetyGuard()
        self.profiler = ProfilerAgent(llm, self.memory.focus_apps)
        self.agent = ActionAgent(llm)
        self.watch = Watch()
        self.state: UserState | None = None
        self.features: Features | None = None
        self.history: list[tuple] = []  # (time, UserState)
        self.actions: list[Action] = []
        self.stats = Stats()
        self.log_path = Path(log_path) if log_path else None

    @property
    def mode(self) -> str:
        return f"LLM ({self.llm.label})" if self.llm else "Rules only (offline)"

    def process(self, r: Reading) -> StepResult:
        started = time.perf_counter()
        self.stats.readings += 1
        new: list[Action] = []

        features, changes = self.edge.update(r)
        new += self._follow_emergency(r)
        if r.notification:
            self.watch.pending = r.notification
        if features is None:  # still warming up
            return StepResult(r, None, self.state, [], new)
        self.features = features

        alarm = self.safety.check(features)
        if alarm:
            new.append(self._run("start_emergency_protocol", {"reason": alarm}, "Safety Guard", r))

        if changes:
            self.state = self.profiler.profile(features, changes)
            self.history.append((r.time, self.state))
            self.stats.profiler_runs += 1

        triggers = list(changes)
        if r.notification:
            n = r.notification
            triggers.append(f"new message from {n.source} ({n.priority} priority)")
        if triggers and self.state:
            waited_before = self.llm.waited_seconds if self.llm else 0.0
            for round_no in range(1 + FOLLOW_UP_ROUNDS):
                calls, by = self.agent.decide(self._context(triggers, features, r))
                self.stats.action_runs += 1
                if round_no:  # a follow-up look only adds actions; "nothing more to do" is not logged
                    calls = [(n, a) for n, a in calls if n != "do_nothing"]
                for name, args in calls:
                    new.append(self._run(name, args, by, r))
                # Observe -> act -> observe: after changing the watch's mode, let the LLM
                # look once more, since one change often unlocks another (DND off -> deliver held).
                changed = [n for n, _ in calls if n in STATE_CHANGING_TOOLS]
                if not changed or by != "Action Agent (LLM)":
                    break
                triggers = [f"follow-up: you just called {', '.join(changed)}; check the rules again"]
            waited = (self.llm.waited_seconds if self.llm else 0.0) - waited_before
            self.stats.decision_ms.append((time.perf_counter() - started - waited) * 1000)
            if self.watch.pending:  # the agent chose not to act on the message
                self.watch.pending = None
                new.append(self._run("log_event", {"reason": "Message left silently in the inbox"}, "Engine", r))

        if self.llm:
            self.stats.llm_calls, self.stats.llm_failures = self.llm.calls, self.llm.failures
        return StepResult(r, features, self.state, changes, new)

    def log_event(self, action: Action) -> None:
        """Record every action and its reason (the action log)."""
        self.actions.append(action)
        if self.log_path:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(action.model_dump(mode="json")) + "\n")

    def _run(self, name: str, args: dict, by: str, r: Reading) -> Action:
        args = dict(args)
        reason = str(args.pop("reason", "")).strip() or "(no reason given)"
        result = run_tool(self.watch, name, args, r)
        action = Action(time=r.time, tool=name, args=args, reason=reason, by=by, result=result)
        self.log_event(action)
        return action

    def _follow_emergency(self, r: Reading) -> list[Action]:
        e = self.watch.emergency
        if not e or e.status != "countdown":
            return []
        if r.user_input == "im_ok":
            return [self._run("cancel_emergency", {"reason": "User tapped 'I'm OK'"}, "Safety Guard", r)]
        if r.time >= e.deadline:
            contact = self.memory.emergency_contact
            return [self._run("alert_emergency_contact", {
                "contact": contact["name"], "phone": contact["phone"], "lat": r.gps.lat, "lon": r.gps.lon,
                "reason": "No answer to 'Are you OK?' within 30 seconds",
            }, "Safety Guard", r)]
        return []

    def _context(self, triggers: list[str], f: Features, r: Reading) -> DecisionContext:
        recent = [(t, s) for t, s in self.history if r.time - t <= timedelta(minutes=5)]
        return DecisionContext(
            triggers=triggers,
            state=self.state,
            features=f,
            watch=self.watch,
            notification=self.watch.pending,
            focus_apps=self.memory.focus_apps,
            recent_actions=[f"{a.time:%H:%M:%S} {a.tool}: {a.reason}" for a in self.actions[-5:]],
            recently_commuting=any(s.activity == "commuting" for _, s in recent),
            recently_breathing=any(a.tool == "suggest_breathing" and r.time - a.time <= timedelta(minutes=10)
                                   for a in self.actions),
        )
