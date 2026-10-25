"""Action Agent: decides whether to interrupt the user, and acts through tools.

With an LLM it uses tool calling. Without one (or if the call fails) the same
policy runs as plain rules, so the watch never stops working. Both read the
same pre-computed signals, so they reason from identical facts.
"""
import json
from dataclasses import dataclass, field

from .schemas import Features, Notification, UserState
from .tools import AGENT_TOOLS, TOOL_SPECS, Watch

SYSTEM_PROMPT = """You are the Action Agent of a smartwatch. You are called when the user's
situation changes or a message arrives. Decide whether to interrupt the user and
act by calling tools. Interrupting breaks sleep and focus and adds stress; staying
quiet can make the user miss something important.

The "signals" field is already worked out for you. Check EVERY rule below and make
one tool call for each rule that matches (often more than one):
A. focus_app_open and not dnd_on -> enable_dnd
B. dnd_on and not focus_app_open -> disable_dnd
C. navigating and battery_percent <= 10 and not battery_saver_on -> battery_saver_mode.
   While navigating, never warn about battery.
D. not navigating and battery_percent <= 15 and not low_battery_warning_shown
   -> send_notification with a battery warning
E. incoming_message present -> first judge from its TEXT how urgent it really is
   (an emergency, accident or hospital message is urgent whatever its priority field says). Then:
   - urgent or high priority -> send_notification now (priority high)
   - else if asleep, focus_app_open, (busy and stress is high),
     (navigating and battery_percent <= 10), or (busy and priority is low) -> defer_notification
   - else -> send_notification with the message text
F. no incoming_message and held_messages > 0 and user_free -> deliver_deferred
G. stress is high and just_arrived_after_commute and not breathing_suggested_recently
   -> suggest_breathing
H. very_high_hr_while_still and not emergency_running -> start_emergency_protocol
   (a high heart rate while exercising is normal)
If no rule matches, call do_nothing. Give every call a short reason a person can read."""


@dataclass
class DecisionContext:
    triggers: list[str]
    state: UserState
    features: Features
    watch: Watch
    notification: Notification | None
    focus_apps: set[str]
    recent_actions: list[str] = field(default_factory=list)
    recently_commuting: bool = False
    recently_breathing: bool = False

    @property
    def signals(self) -> dict:
        s, f = self.state, self.features
        focus = f.active_app in self.focus_apps
        return {
            "focus_app_open": focus,
            "dnd_on": self.watch.dnd,
            "navigating": f.active_app == "maps" or s.place == "unfamiliar",
            "battery_percent": f.battery,
            "battery_saver_on": self.watch.battery_saver,
            "low_battery_warning_shown": self.watch.low_battery_warned,
            "asleep": s.attention == "asleep",
            "busy": s.attention == "busy",
            "user_free": s.attention == "free" and not focus,
            "held_messages": len(self.watch.deferred),
            "just_arrived_after_commute": self.recently_commuting and s.activity != "commuting",
            "breathing_suggested_recently": self.recently_breathing,
            "exercising": s.activity in ("walking", "running"),
            "very_high_hr_while_still": f.hr_zone == "high" and f.motion == "still",
            "emergency_running": bool(self.watch.emergency and self.watch.emergency.status == "countdown"),
        }

    def to_prompt(self) -> str:
        f, s = self.features, self.state
        return json.dumps({
            "time": f.time.strftime("%H:%M:%S"),
            "why_you_are_called": self.triggers,
            "user": s.model_dump(include={"activity", "stress", "attention", "place", "summary"}),
            "signals": self.signals,
            "heart_rate": {"now": f.hr_latest, "normal": f.hr_baseline, "zone": f.hr_zone},
            "active_app": f.active_app,
            "incoming_message": self.notification.model_dump(exclude={"id"}) if self.notification else None,
            "held_messages": self.watch.snapshot()["held_messages"],
            "recent_actions": self.recent_actions[-3:],
        }, separators=(",", ":"))


class ActionAgent:
    def __init__(self, llm=None):
        self.llm = llm
        self.last_error: str | None = None

    def decide(self, ctx: DecisionContext) -> tuple[list[tuple[str, dict]], str]:
        """Return the tool calls to make and who decided them."""
        if self.llm:
            try:
                calls = [(n, a) for n, a in self.llm.tool_chat(SYSTEM_PROMPT, ctx.to_prompt(), TOOL_SPECS)
                         if n in AGENT_TOOLS]
                if calls:
                    return calls, "Action Agent (LLM)"
                self.last_error = "LLM returned no usable tool call"
            except Exception as e:  # network, rate limit, bad output: fall back to rules
                self.last_error = f"{type(e).__name__}: {e}"[:200]
        return rules_decide(ctx), "Action Agent (rules)"


def rules_decide(c: DecisionContext) -> list[tuple[str, dict]]:
    """The same policy as the prompt, written as plain rules."""
    s, f, w, n, sig = c.state, c.features, c.watch, c.notification, c.signals
    calls: list[tuple[str, dict]] = []

    if sig["focus_app_open"] and not w.dnd:
        calls.append(("enable_dnd", {"reason": f"{f.active_app.replace('_', ' ')} is open; protect focus time"}))
    elif not sig["focus_app_open"] and w.dnd:
        calls.append(("disable_dnd", {"reason": "Focus session ended"}))

    if sig["navigating"]:
        if f.battery <= 10 and not w.battery_saver:
            calls.append(("battery_saver_mode", {"reason": f"Battery {f.battery}% while navigating: save power "
                                                           "quietly and keep Maps alive instead of warning"}))
    elif f.battery <= 15 and not w.low_battery_warned and s.attention in ("free", "busy"):
        calls.append(("send_notification", {"message": f"Battery low ({f.battery}%). Charge soon.",
                                            "priority": "medium", "reason": "Battery is low and the user can act on it"}))

    if n:
        hold = _hold(n, s, f, sig)
        if hold:
            calls.append(("defer_notification", {"until": hold[1], "reason": hold[0]}))
        else:
            calls.append(("send_notification", {"message": f"{n.source}: {n.text}", "priority": n.priority,
                                                "reason": f"{n.priority.capitalize()}-priority message and the user "
                                                          f"is free ({s.summary})"}))
    elif sig["held_messages"] and sig["user_free"]:
        calls.append(("deliver_deferred", {"reason": f"User is free again; deliver {sig['held_messages']} held "
                                                     "message(s) as one summary"}))

    if s.stress == "high" and sig["just_arrived_after_commute"] and not sig["breathing_suggested_recently"]:
        calls.append(("suggest_breathing", {"reason": "Still highly stressed right after the commute"}))

    if not calls:
        calls.append(("do_nothing", {"reason": _quiet_reason(s, f)}))
    return calls


def _hold(n: Notification, s: UserState, f: Features, sig: dict) -> tuple[str, str] | None:
    """(reason, until) if the message should wait, else None."""
    if n.priority == "high":
        return None
    if s.attention == "asleep":
        return "User is asleep and the message can wait", "user_wakes"
    if sig["focus_app_open"]:
        return "User is in a focus session (Do Not Disturb)", "focus_ends"
    if s.attention == "busy" and s.stress == "high":
        return "User is stressed and busy; don't add to it", "user_arrives"
    if sig["navigating"] and f.battery <= 10:
        return "Battery is very low while navigating; keep the screen for Maps", "later"
    if s.attention == "busy" and n.priority == "low":
        return "User is busy and this is low priority", "later"
    return None


def _quiet_reason(s: UserState, f: Features) -> str:
    if f.hr_zone == "high" and f.activity == "running":
        return "High heart rate while running is normal exercise"
    if f.hr_zone == "high" and f.activity in ("sitting", "sleeping"):
        return "Very high heart rate while still: the Safety Guard is watching this"
    if f.active_app == "maps" and f.battery <= 15:
        return "Low battery, but the user is navigating: stay quiet"
    return f"{s.summary}. Nothing needs the user's attention"
