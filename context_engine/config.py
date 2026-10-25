"""Settings read from .env: which LLM provider to use and which models."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Both providers speak the OpenAI chat-completions API, so one client works for either.
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "fast_model": "openai/gpt-oss-20b",
        "smart_model": "openai/gpt-oss-120b",
    },
    "xai": {
        "base_url": "https://api.x.ai/v1",
        "key_env": "XAI_API_KEY",
        "fast_model": "grok-4-1-fast-non-reasoning",
        "smart_model": "grok-4-1-fast-reasoning",
    },
}


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    base_url: str
    api_key: str | None
    fast_model: str   # Profiler Agent: runs on every situation change, must be quick
    smart_model: str  # Action Agent: does tool calling, worth a stronger model


def llm_settings() -> LLMSettings:
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {sorted(PROVIDERS)}, got {provider!r}")
    p = PROVIDERS[provider]
    return LLMSettings(
        provider=provider,
        base_url=p["base_url"],
        api_key=os.getenv(p["key_env"]) or None,
        fast_model=os.getenv("FAST_MODEL", p["fast_model"]),
        smart_model=os.getenv("SMART_MODEL", p["smart_model"]),
    )
