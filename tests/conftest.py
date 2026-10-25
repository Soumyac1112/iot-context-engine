import os
import sys
from pathlib import Path

# Tests never call the real LLM; the fake below stands in for it.
os.environ["LLM_DISABLED"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeLLM:
    """Stands in for the Groq/xAI client. tool_script is a list of tool-call lists, one per call."""

    def __init__(self, tool_script=None, profile=None, fail=False):
        self.tool_script = list(tool_script or [])
        self.profile = profile or {}
        self.fail = fail
        self.calls = 0
        self.failures = 0
        self.waited_seconds = 0.0
        self.label = "fake"
        self.prompts: list[str] = []

    def json_chat(self, system, user):
        self.calls += 1
        if self.fail:
            self.failures += 1
            raise RuntimeError("429 rate limit")
        return dict(self.profile)

    def tool_chat(self, system, user, tools):
        self.calls += 1
        self.prompts.append(user)
        if self.fail:
            self.failures += 1
            raise RuntimeError("429 rate limit")
        return self.tool_script.pop(0) if self.tool_script else [("do_nothing", {"reason": "fake"})]
