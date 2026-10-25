"""Profiler Agent: turns the Edge Layer's facts into a picture of the user's situation.

Simple cases are handled by rules. When an LLM is available it refines the rule
guess (stress, attention, a natural summary); if the LLM fails, rules are used.
"""
import json
from typing import get_args

from pydantic import ValidationError

from .schemas import Activity, Attention, Features, Place, Stress, UserState

SYSTEM_PROMPT = """You are the Profiler Agent of a smartwatch "context engine".
From sensor facts, describe what the user is doing right now.

Reply with JSON only, with exactly these keys:
  activity: one of sleeping, sitting, walking, running, commuting
  stress: one of low, medium, high
  attention: one of free, busy, focused, asleep
  summary: one short sentence that starts with "User is"
  confidence: number from 0 to 1

Meaning of attention:
  asleep  - the user is sleeping
  focused - a focus app (e.g. meditation, beads counting) is open
  busy    - commuting, running, or navigating with maps
  free    - none of the above

Stress: a heart rate well above normal while NOT exercising suggests stress.
A high heart rate while walking or running is exercise, not stress.
A rule-based guess is included. Keep it unless the facts clearly say otherwise."""

_ALLOWED = {
    "activity": set(get_args(Activity)),
    "stress": set(get_args(Stress)),
    "attention": set(get_args(Attention)),
}

_STRESS_WORD = {"low": "calm", "medium": "a little stressed", "high": "highly stressed"}
_PLACE_PHRASE = {"home": "at home", "work": "at work", "known": "around town", "unfamiliar": "in an unfamiliar place"}


def rules_state(f: Features, focus_apps: set[str]) -> UserState:
    exercising = f.activity in ("walking", "running")
    if exercising or f.activity == "sleeping":
        stress = "low"
    elif f.hr_zone == "high" or f.hr_avg >= f.hr_baseline + 25:
        stress = "high"
    elif f.hr_zone == "elevated":
        stress = "medium"
    else:
        stress = "low"

    if f.activity == "sleeping":
        attention = "asleep"
    elif f.active_app in focus_apps:
        attention = "focused"
    elif f.activity in ("commuting", "running") or f.active_app == "maps":
        attention = "busy"
    else:
        attention = "free"

    return UserState(activity=f.activity, stress=stress, place=f.place, attention=attention,
                     summary=_summary(f, stress, attention), confidence=0.7, source="rules")


def _summary(f: Features, stress: str, attention: str) -> str:
    place = _PLACE_PHRASE[f.place]
    if f.hr_zone == "high" and f.activity in ("sitting", "sleeping"):
        return f"User is sitting still with a very high heart rate ({f.hr_latest} bpm, normal {f.hr_baseline:.0f})"
    if f.activity == "sleeping":
        return f"User is sleeping {place}"
    if attention == "focused":
        return f"User is focused on {f.active_app.replace('_', ' ')} {place}"
    if f.active_app == "maps":
        return f"User is {f.activity} {place}, navigating with Maps (battery {f.battery}%)"
    if f.activity == "running":
        return f"User is running {place} ({f.hr_latest} bpm, normal for exercise)"
    if f.activity == "commuting":
        return f"User is {_STRESS_WORD[stress]} and commuting"
    return f"User is {_STRESS_WORD[stress]} and {f.activity} {place}"


class ProfilerAgent:
    def __init__(self, llm=None, focus_apps: set[str] | None = None):
        self.llm = llm
        self.focus_apps = focus_apps or set()
        self.last_error: str | None = None

    def profile(self, f: Features, changes: list[str]) -> UserState:
        guess = rules_state(f, self.focus_apps)
        if not self.llm:
            return guess
        try:
            prompt = json.dumps({
                "facts": f.model_dump(mode="json", exclude={"lat", "lon", "time", "window_seconds"}),
                "what_changed": changes,
                "focus_app_open": f.active_app in self.focus_apps,
                "rule_guess": guess.model_dump(include={"activity", "stress", "attention", "summary"}),
            }, separators=(",", ":"))
            return _merge(self.llm.json_chat(SYSTEM_PROMPT, prompt), guess)
        except Exception as e:  # network, rate limit, bad JSON: fall back to rules
            self.last_error = f"{type(e).__name__}: {e}"[:200]
            return guess


def _merge(data: dict, guess: UserState) -> UserState:
    """Take each LLM field only if it is valid; place always comes from GPS + memory."""
    merged = guess.model_dump()
    for key, allowed in _ALLOWED.items():
        value = str(data.get(key, "")).strip().lower()
        if value in allowed:
            merged[key] = value
    summary = str(data.get("summary", "")).strip()
    if summary:
        merged["summary"] = summary[:160]
    try:
        merged["confidence"] = min(1.0, max(0.0, float(data.get("confidence", guess.confidence))))
    except (TypeError, ValueError):
        pass
    merged["source"] = "llm"
    try:
        return UserState(**merged)
    except ValidationError:
        return guess
