"""User Memory: what is normal for this user (baseline heart rate, known places)."""
import json
import math
from pathlib import Path

from .schemas import Place

# Synthetic demo user. Never put real personal or health data here.
DEFAULT_PROFILE = {
    "name": "Demo user",
    "resting_hr": 70,
    "home": {"lat": 22.5726, "lon": 88.3639},
    "work": {"lat": 22.5803, "lon": 88.4335},
    "home_city": {"lat": 22.5726, "lon": 88.3639, "radius_km": 25},
    "place_radius_m": 250,
    "focus_apps": ["beads_counter", "meditation"],
    "emergency_contact": {"name": "Emergency contact (demo)", "phone": "+00 0000 000000"},
}


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres (haversine)."""
    r = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


class UserMemory:
    def __init__(self, profile: dict | None = None):
        self.profile = {**DEFAULT_PROFILE, **(profile or {})}

    @classmethod
    def load(cls, path: str | Path) -> "UserMemory":
        path = Path(path)
        if path.exists():
            return cls(json.loads(path.read_text(encoding="utf-8")))
        return cls()

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.profile, indent=2), encoding="utf-8")

    @property
    def resting_hr(self) -> float:
        return float(self.profile["resting_hr"])

    @property
    def focus_apps(self) -> set[str]:
        return set(self.profile["focus_apps"])

    @property
    def emergency_contact(self) -> dict:
        return self.profile["emergency_contact"]

    def place_of(self, lat: float, lon: float) -> Place:
        radius = self.profile["place_radius_m"]
        for name in ("home", "work"):
            spot = self.profile[name]
            if distance_m(lat, lon, spot["lat"], spot["lon"]) <= radius:
                return name
        city = self.profile["home_city"]
        if distance_m(lat, lon, city["lat"], city["lon"]) <= city["radius_km"] * 1000:
            return "known"
        return "unfamiliar"
