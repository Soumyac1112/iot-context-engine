"""Data shapes passed between the blocks. Pydantic checks every field."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Priority = Literal["low", "medium", "high"]
Activity = Literal["sleeping", "sitting", "walking", "running", "commuting"]
Motion = Literal["still", "light", "moderate", "high"]
Place = Literal["home", "work", "known", "unfamiliar"]
HRZone = Literal["low", "normal", "elevated", "high"]
Stress = Literal["low", "medium", "high"]
Attention = Literal["free", "busy", "focused", "asleep"]


class Accel(BaseModel):
    x: float
    y: float
    z: float


class GPS(BaseModel):
    lat: float
    lon: float
    speed_kmh: float


class Notification(BaseModel):
    id: str
    source: str
    text: str
    priority: Priority


class Reading(BaseModel):
    """One JSON message from the (simulated) watch, sent every second."""
    time: datetime
    heart_rate: int
    accel: Accel
    steps: int
    gps: GPS
    battery: int
    active_app: str | None = None
    screen_on: bool = False
    notification: Notification | None = None  # a message that arrived this second
    user_input: str | None = None              # e.g. "im_ok" tapped on the watch


class Features(BaseModel):
    """Facts the Edge Layer works out from the rolling window."""
    time: datetime
    window_seconds: int
    hr_latest: int
    hr_avg: float
    hr_baseline: float
    hr_zone: HRZone
    motion: Motion
    still_seconds: int
    cadence_spm: float
    speed_kmh: float
    place: Place
    lat: float
    lon: float
    battery: int
    battery_drop_per_min: float
    active_app: str | None
    screen_on: bool
    activity: Activity  # rule-based first guess


class UserState(BaseModel):
    """The Profiler Agent's picture of the user right now."""
    activity: Activity
    stress: Stress
    place: Place
    attention: Attention
    summary: str
    confidence: float = Field(ge=0, le=1)
    source: Literal["rules", "llm"] = "rules"


class Action(BaseModel):
    """One tool call, who made it, and why. This is what the action log stores."""
    time: datetime
    tool: str
    args: dict = Field(default_factory=dict)
    reason: str
    by: str
    result: str = ""
