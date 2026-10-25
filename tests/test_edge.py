from collections import Counter
from datetime import datetime, timedelta

from context_engine import BY_KEY, UserMemory
from context_engine.edge import EdgeLayer
from context_engine.schemas import GPS, Accel, Reading
from context_engine.simulator import generate


def _activities(key: str) -> Counter:
    edge = EdgeLayer(UserMemory())
    seen = Counter()
    for r in generate(BY_KEY[key]):
        f, _ = edge.update(r)
        if f:
            seen[f.activity] += 1
    return seen


def test_detects_running():
    assert _activities("hr_high_running").most_common(1)[0][0] == "running"


def test_detects_commuting():
    assert _activities("stressed_commute")["commuting"] > 100


def test_detects_sleeping():
    assert _activities("sleeping_low_priority")["sleeping"] > 100


def test_detects_walking():
    assert _activities("free_walk").most_common(1)[0][0] == "walking"


def test_place_from_memory():
    m = UserMemory()
    assert m.place_of(22.5726, 88.3639) == "home"
    assert m.place_of(22.5803, 88.4335) == "work"
    assert m.place_of(22.60, 88.40) == "known"
    assert m.place_of(48.8606, 2.3376) == "unfamiliar"


def test_reports_few_changes():
    """The Edge Layer must wake the agents rarely, not on every reading."""
    for key in BY_KEY:
        edge = EdgeLayer(UserMemory())
        readings = list(generate(BY_KEY[key]))
        changes = sum(1 for r in readings if edge.update(r)[1])
        assert changes <= 10, (key, changes)


def _reading(t: datetime, hr: int) -> Reading:
    return Reading(time=t, heart_rate=hr, accel=Accel(x=0, y=0, z=1), steps=0,
                   gps=GPS(lat=22.5726, lon=88.3639, speed_kmh=0), battery=80)


def test_heart_rate_zone_does_not_flip_flop_at_boundary():
    edge = EdgeLayer(UserMemory())  # normal 70 -> "low" boundary at 55
    t = datetime(2026, 1, 1)
    zone_changes = 0
    for i in range(120):
        _, changes = edge.update(_reading(t + timedelta(seconds=i), 54 + (i % 3)))  # 54, 55, 56, ...
        zone_changes += sum("heart-rate zone" in c for c in changes)
    assert zone_changes <= 1
