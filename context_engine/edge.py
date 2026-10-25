"""Edge Layer: quick thinking on the watch itself. No AI.

Keeps the last few minutes of readings, works out simple facts, and reports
only *meaningful* changes, so the (slow, costly) AI agents are woken up rarely.
"""
import math
import statistics
from collections import deque

from .memory import UserMemory
from .schemas import Activity, Features, HRZone, Motion, Reading

WINDOW_SECONDS = 300  # rolling window: last 5 minutes
WARMUP_READINGS = 10  # facts are unreliable before this
DEBOUNCE_READINGS = 3  # a change must hold this long before it counts (ignores sensor noise)
HR_HYSTERESIS = 3  # bpm past a zone boundary before the zone changes (stops flip-flopping)


def battery_band(level: int) -> str:
    if level > 20:
        return "ok"
    if level > 10:
        return "low"
    if level > 5:
        return "very low"
    return "critical"


class EdgeLayer:
    def __init__(self, memory: UserMemory, window_seconds: int = WINDOW_SECONDS):
        self.memory = memory
        self.window: deque[Reading] = deque(maxlen=window_seconds)
        self.still_seconds = 0
        self.hr_zone: HRZone | None = None
        self._reported: dict | None = None
        self._candidate: dict | None = None
        self._candidate_count = 0

    def update(self, r: Reading) -> tuple[Features | None, list[str]]:
        """Add one reading. Returns the current facts and a list of meaningful changes."""
        prev = self.window[-1] if self.window else None
        self.window.append(r)

        moved = prev is not None and r.steps > prev.steps
        if abs(_magnitude(r) - 1) < 0.04 and not moved and r.gps.speed_kmh < 1:
            self.still_seconds += 1
        else:
            self.still_seconds = 0

        if len(self.window) < WARMUP_READINGS:
            return None, []
        features = self._features(r)
        return features, self._changes(features)

    def _features(self, r: Reading) -> Features:
        recent = list(self.window)
        baseline = self.memory.resting_hr
        hr_avg = statistics.fmean(x.heart_rate for x in recent[-10:])
        motion_sd = statistics.pstdev(_magnitude(x) for x in recent[-10:])
        speed = statistics.fmean(x.gps.speed_kmh for x in recent[-10:])

        older = recent[-30] if len(recent) >= 30 else recent[0]
        secs = (r.time - older.time).total_seconds() or 1
        cadence = (r.steps - older.steps) / secs * 60

        older = recent[-60] if len(recent) >= 60 else recent[0]
        mins = (r.time - older.time).total_seconds() / 60
        battery_drop = (older.battery - r.battery) / mins if mins else 0.0

        motion = _motion(motion_sd)
        self.hr_zone = _hr_zone(hr_avg, baseline, self.hr_zone)
        return Features(
            time=r.time,
            window_seconds=len(recent),
            hr_latest=r.heart_rate,
            hr_avg=round(hr_avg, 1),
            hr_baseline=baseline,
            hr_zone=self.hr_zone,
            motion=motion,
            still_seconds=self.still_seconds,
            cadence_spm=round(cadence, 1),
            speed_kmh=round(speed, 1),
            place=self.memory.place_of(r.gps.lat, r.gps.lon),
            lat=r.gps.lat,
            lon=r.gps.lon,
            battery=r.battery,
            battery_drop_per_min=round(battery_drop, 2),
            active_app=r.active_app,
            screen_on=r.screen_on,
            activity=_activity(speed, cadence, motion, r.screen_on, hr_avg, baseline, self.still_seconds),
        )

    def _changes(self, f: Features) -> list[str]:
        signature = {
            "activity": f.activity,
            "heart-rate zone": f.hr_zone,
            "place": f.place,
            "battery": battery_band(f.battery),
            "app": f.active_app or "none",
        }
        if self._reported is None:
            self._reported = signature
            return ["first look at the user"]
        if signature == self._reported:
            self._candidate, self._candidate_count = None, 0
            return []

        if signature == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate, self._candidate_count = signature, 1

        app_changed = signature["app"] != self._reported["app"]  # app switches are certain, no debounce
        if not app_changed and self._candidate_count < DEBOUNCE_READINGS:
            return []

        changes = [f"{k}: {self._reported[k]} -> {v}" for k, v in signature.items() if self._reported[k] != v]
        self._reported = signature
        self._candidate, self._candidate_count = None, 0
        return changes


def _magnitude(r: Reading) -> float:
    return math.sqrt(r.accel.x ** 2 + r.accel.y ** 2 + r.accel.z ** 2)


def _motion(sd: float) -> Motion:
    if sd < 0.03:
        return "still"
    if sd < 0.12:
        return "light"
    if sd < 0.4:
        return "moderate"
    return "high"


def _hr_zone(hr: float, baseline: float, previous: HRZone | None = None) -> HRZone:
    # Keep the previous zone while the heart rate is within a few bpm of the boundary.
    if previous in (_raw_zone(hr - HR_HYSTERESIS, baseline), _raw_zone(hr + HR_HYSTERESIS, baseline)):
        return previous
    return _raw_zone(hr, baseline)


def _raw_zone(hr: float, baseline: float) -> HRZone:
    if hr >= baseline + 45:
        return "high"
    if hr >= baseline + 15:
        return "elevated"
    if hr <= baseline - 15:
        return "low"
    return "normal"


def _activity(speed: float, cadence: float, motion: Motion, screen_on: bool,
              hr: float, baseline: float, still_seconds: int) -> Activity:
    if speed >= 12 and cadence < 60:
        return "commuting"  # moving fast without steps: in a vehicle
    if cadence >= 140:
        return "running"
    if cadence >= 50 or speed >= 2.5:
        return "walking"
    if motion == "still" and not screen_on and hr <= baseline - 8 and still_seconds >= 30:
        return "sleeping"
    return "sitting"
