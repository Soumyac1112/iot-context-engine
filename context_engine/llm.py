"""Thin wrapper around the LLM provider (Groq or xAI Grok, both OpenAI-compatible)."""
import json
import os
import time
from collections import defaultdict, deque

from openai import OpenAI

from .config import LLMSettings, llm_settings

# Groq's free tier allows 8,000 tokens per minute per model. Stay a little under it.
TOKENS_PER_MINUTE = int(os.getenv("LLM_TOKENS_PER_MINUTE", "7500"))


class LLMClient:
    def __init__(self, settings: LLMSettings):
        self.settings = settings
        self.client = OpenAI(api_key=settings.api_key, base_url=settings.base_url, timeout=30, max_retries=2)
        self.calls = 0
        self.failures = 0
        self.tokens = 0
        self.waited_seconds = 0.0
        self._usage: dict[str, deque] = defaultdict(deque)  # model -> (time, tokens) in the last minute
        self._last_tokens: dict[str, int] = {}

    @property
    def label(self) -> str:
        return f"{self.settings.provider}: {self.settings.fast_model} / {self.settings.smart_model}"

    def json_chat(self, system: str, user: str) -> dict:
        """Ask the fast model for a JSON object (used by the Profiler Agent)."""
        resp = self._create(self.settings.fast_model, system, user, response_format={"type": "json_object"})
        return json.loads(resp.choices[0].message.content)

    def tool_chat(self, system: str, user: str, tools: list[dict]) -> list[tuple[str, dict]]:
        """Ask the smart model to pick tools (used by the Action Agent)."""
        resp = self._create(self.settings.smart_model, system, user, tools=tools, tool_choice="required")
        calls = resp.choices[0].message.tool_calls or []
        return [(c.function.name, json.loads(c.function.arguments or "{}")) for c in calls]

    def _create(self, model: str, system: str, user: str, **kwargs):
        self._wait_for_budget(model)
        self.calls += 1
        if "gpt-oss" in model:
            kwargs["extra_body"] = {"reasoning_effort": "low"}  # fewer hidden "thinking" tokens
        try:
            resp = self.client.chat.completions.create(
                model=model, temperature=0,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                **kwargs,
            )
        except Exception:
            self.failures += 1
            raise
        used = resp.usage.total_tokens if resp.usage else self._last_tokens.get(model, 1500)
        self.tokens += used
        self._last_tokens[model] = used
        self._usage[model].append((time.monotonic(), used))
        return resp

    def _wait_for_budget(self, model: str) -> None:
        """Sleep just long enough that the next call stays inside the per-minute token limit."""
        q, estimate = self._usage[model], self._last_tokens.get(model, 1500)
        while True:
            now = time.monotonic()
            while q and now - q[0][0] > 60:
                q.popleft()
            if not q or sum(t for _, t in q) + estimate <= TOKENS_PER_MINUTE:
                return
            pause = 60 - (now - q[0][0]) + 0.2
            self.waited_seconds += pause
            time.sleep(pause)


def make_llm() -> LLMClient | None:
    """An LLM client if a key is configured, else None (the engine then runs on rules)."""
    if os.getenv("LLM_DISABLED") == "1":
        return None
    settings = llm_settings()
    return LLMClient(settings) if settings.api_key else None
