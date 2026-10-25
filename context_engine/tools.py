"""Tools: the actions the system can take on the (simulated) watch.

The Action Agent may call the AGENT tools. The emergency follow-ups
(alert_emergency_contact, cancel_emergency) are used only by the Safety Guard.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .schemas import Notification, Reading

EMERGENCY_COUNTDOWN_SECONDS = 30


@dataclass
class Emergency:
    started: datetime
    deadline: datetime
    status: str = "countdown"  # countdown | cancelled | contact_alerted


@dataclass
class ScreenItem:
    time: datetime
    kind: str  # notification | summary | nudge | emergency | system
    text: str


@dataclass
class Watch:
    """Everything the tools can change on the watch."""
    dnd: bool = False
    battery_saver: bool = False
    low_battery_warned: bool = False
    pending: Notification | None = None  # message waiting for a decision
    deferred: list[Notification] = field(default_factory=list)
    screen: list[ScreenItem] = field(default_factory=list)
    emergency: Emergency | None = None

    def snapshot(self) -> dict:
        return {
            "do_not_disturb": self.dnd,
            "battery_saver": self.battery_saver,
            "low_battery_warning_shown": self.low_battery_warned,
            "held_messages": [f"{n.source}: {n.text}" for n in self.deferred],
            "emergency": self.emergency.status if self.emergency else None,
        }


def _spec(name: str, description: str, props: dict | None = None, required: tuple = ()) -> dict:
    props = {**(props or {}), "reason": {"type": "string", "description": "Short reason a person can read"}}
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": props, "required": [*required, "reason"]},
    }}


TOOL_SPECS = [
    _spec("send_notification", "Show a message on the watch now (an incoming message or a warning).",
          {"message": {"type": "string"}, "priority": {"type": "string", "enum": ["low", "medium", "high"]}},
          ("message", "priority")),
    _spec("defer_notification", "Hold the incoming message and deliver it later.",
          {"until": {"type": "string", "enum": ["user_wakes", "focus_ends", "user_arrives", "later"]}},
          ("until",)),
    _spec("deliver_deferred", "Show one summary of all held messages."),
    _spec("enable_dnd", "Turn on Do Not Disturb."),
    _spec("disable_dnd", "Turn off Do Not Disturb."),
    _spec("battery_saver_mode", "Turn on battery saver: dim screen, pause background sync, keep navigation running."),
    _spec("suggest_breathing", "Offer a 1-minute breathing exercise."),
    _spec("start_emergency_protocol",
          "Show 'Are you OK?' with a 30-second countdown, then alert the emergency contact with GPS if there is no answer."),
    _spec("do_nothing", "Take no action right now."),
]
AGENT_TOOLS = {s["function"]["name"] for s in TOOL_SPECS}


def run_tool(watch: Watch, name: str, args: dict, r: Reading) -> str:
    """Carry out one tool call on the watch and return what happened."""
    now = r.time
    show = lambda kind, text: watch.screen.append(ScreenItem(now, kind, text))  # noqa: E731

    if name == "send_notification":
        message, priority = str(args.get("message", "")), args.get("priority", "medium")
        if watch.pending and watch.pending.text.lower()[:25] in message.lower():
            watch.pending = None
        if "battery" in message.lower():
            watch.low_battery_warned = True
        show("notification", message)
        silent = watch.dnd and priority != "high"
        return "Shown silently (Do Not Disturb is on)" if silent else f"Shown on watch ({priority})"

    if name == "defer_notification":
        if not watch.pending:
            return "No incoming message to hold"
        n, watch.pending = watch.pending, None
        watch.deferred.append(n)
        return f"Held '{n.source}: {n.text}' until {args.get('until', 'later')} ({len(watch.deferred)} held)"

    if name == "deliver_deferred":
        if not watch.deferred:
            return "Nothing was held"
        lines = "; ".join(f"{n.source}: {n.text}" for n in watch.deferred)
        count = len(watch.deferred)
        watch.deferred.clear()
        show("summary", f"While you were busy ({count}): {lines}")
        return f"Delivered {count} held message(s) as one summary"

    if name in ("enable_dnd", "disable_dnd"):
        target = name == "enable_dnd"
        if watch.dnd == target:
            return f"Do Not Disturb already {'on' if target else 'off'}"
        watch.dnd = target
        return f"Do Not Disturb {'on' if target else 'off'}"

    if name == "battery_saver_mode":
        if watch.battery_saver:
            return "Battery saver already on"
        watch.battery_saver = True
        return "Battery saver on: screen dimmed, background sync paused, navigation kept running"

    if name == "suggest_breathing":
        show("nudge", "You seem tense. Take a 1-minute breathing break?")
        return "Breathing exercise offered"

    if name == "start_emergency_protocol":
        if watch.emergency and watch.emergency.status == "countdown":
            return "Emergency protocol already running"
        watch.emergency = Emergency(now, now + timedelta(seconds=EMERGENCY_COUNTDOWN_SECONDS))
        show("emergency", f"Are you OK? Tap 'I'm OK' within {EMERGENCY_COUNTDOWN_SECONDS}s "
                          "or your emergency contact will be called.")
        return f"Countdown started ({EMERGENCY_COUNTDOWN_SECONDS}s)"

    if name == "alert_emergency_contact":
        watch.emergency.status = "contact_alerted"
        show("emergency", f"Calling {args.get('contact')} - location shared")
        return f"Alerted {args.get('contact')} ({args.get('phone')}) with location {args.get('lat')}, {args.get('lon')}"

    if name == "cancel_emergency":
        watch.emergency.status = "cancelled"
        show("system", "Glad you're OK. Emergency cancelled.")
        return "Emergency cancelled by the user"

    if name in ("do_nothing", "log_event"):
        return "No action"

    return f"Unknown tool '{name}' ignored"
