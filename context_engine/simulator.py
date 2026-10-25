"""Telemetry Simulator: a fake smartwatch that emits one JSON reading per second.

A scenario is a list of phases ("sit on the sofa for 60 s", "ride to work for
150 s", ...). Each phase sets targets; the simulator adds realistic noise.
"""
import asyncio
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import AsyncIterator, Iterator, Literal

from .schemas import GPS, Accel, Notification, Reading

MotionKind = Literal["still", "light", "walking", "running", "vehicle"]

# Fixed points the scenarios happen at (synthetic coordinates).
ANCHORS = {
    "home": (22.5726, 88.3639),
    "work": (22.5803, 88.4335),
    "park": (22.5480, 88.3426),
    "near_home": (22.5690, 88.3700),
    "route_to_work": (22.5803, 88.4180),  # ~1.6 km west of work
    "paris": (48.8606, 2.3376),
}

# accel noise (g) and step cadence (steps/min) per kind of movement
MOTION = {
    "still": (0.008, 0),
    "light": (0.05, 0),
    "walking": (0.25, 105),
    "running": (0.60, 165),
    "vehicle": (0.06, 0),
}


@dataclass
class Phase:
    seconds: int
    hr: float                            # heart rate the body drifts towards
    motion: MotionKind
    at: str = "home"                     # anchor the phase starts from
    speed_kmh: float = 0.0
    heading_deg: float = 90.0            # 0 = north, 90 = east
    app: str | None = None
    screen_on: bool = False
    battery: tuple[float, float] | None = None  # (start, end); None = slow normal drain
    hr_ramp: float = 0.15                # how fast heart rate moves towards the target
    notifications: dict[int, Notification] = field(default_factory=dict)  # second -> message
    user_inputs: dict[int, str] = field(default_factory=dict)


@dataclass
class Scenario:
    key: str
    title: str
    story: str
    start: datetime
    battery_start: float
    phases: list[Phase]
    must_call: set[str] = field(default_factory=set)      # tools that must be used
    must_not_call: set[str] = field(default_factory=set)  # tools that would be a mistake
    needs_llm: bool = False  # rules alone are not expected to pass (e.g. must read message text)

    @property
    def seconds(self) -> int:
        return sum(p.seconds for p in self.phases)


def _move(lat: float, lon: float, km: float, heading_deg: float) -> tuple[float, float]:
    h = math.radians(heading_deg)
    dlat = km * math.cos(h) / 111.32
    dlon = km * math.sin(h) / (111.32 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def generate(scenario: Scenario, seed: int = 7) -> Iterator[Reading]:
    """Yield the scenario's readings, one per simulated second."""
    rng = random.Random(seed)
    t = scenario.start
    hr = scenario.phases[0].hr
    battery = scenario.battery_start
    step_total = 0.0

    for phase in scenario.phases:
        noise, cadence = MOTION[phase.motion]
        lat0, lon0 = ANCHORS[phase.at]
        b_start, b_end = phase.battery or (battery, battery - phase.seconds * 0.0005)

        for i in range(phase.seconds):
            hr += (phase.hr - hr) * phase.hr_ramp + rng.gauss(0, 1.2)
            step_total += cadence / 60 * max(0.0, 1 + rng.gauss(0, 0.05))
            lat, lon = _move(lat0, lon0, phase.speed_kmh / 3600 * i, phase.heading_deg)
            speed = max(0.0, phase.speed_kmh + rng.gauss(0, 0.6)) if phase.speed_kmh else 0.0
            battery = b_start + (b_end - b_start) * i / phase.seconds

            yield Reading(
                time=t,
                heart_rate=round(hr),
                accel=Accel(x=rng.gauss(0, noise), y=rng.gauss(0, noise), z=1 + rng.gauss(0, noise)),
                steps=int(step_total),
                gps=GPS(lat=round(lat, 6), lon=round(lon, 6), speed_kmh=round(speed, 1)),
                battery=max(0, round(battery)),
                active_app=phase.app,
                screen_on=phase.screen_on,
                notification=phase.notifications.get(i),
                user_input=phase.user_inputs.get(i),
            )
            t += timedelta(seconds=1)


async def stream(scenario: Scenario, speed: float = 1.0, seed: int = 7) -> AsyncIterator[Reading]:
    """Async version: waits between readings, like a real watch (speed=10 -> 10x faster)."""
    for reading in generate(scenario, seed):
        yield reading
        if speed > 0:
            await asyncio.sleep(1 / speed)
