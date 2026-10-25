"""IoT Wearable Context Engine: a smartwatch 'brain' that knows when to speak and when to stay quiet."""
from .engine import ContextEngine
from .evaluate import run_scenario
from .llm import make_llm
from .memory import UserMemory
from .scenarios import BY_KEY, SCENARIOS

__all__ = ["ContextEngine", "run_scenario", "make_llm", "UserMemory", "SCENARIOS", "BY_KEY"]
