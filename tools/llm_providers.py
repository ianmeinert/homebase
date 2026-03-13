"""
llm_providers.py  -  Multi-provider LLM abstraction for HOMEBASE.

Supported providers
-------------------
- Groq / Llama-3.3-70b        (primary  -  subagents + fallback synthesizer)
- Anthropic / Claude Sonnet    (synthesizer only  -  activated by ANTHROPIC_API_KEY)

Provider selection
------------------
Synthesizer (orchestrator.py):
    1. If ANTHROPIC_API_KEY env var is set -> Claude Sonnet
    2. If anthropic_api_key passed via state -> Claude Sonnet
    3. Otherwise -> Groq/Llama (existing behavior, unchanged)

Subagents (llm_tools.py):
    Always Groq.  Claude is not used for parallel batch recommendation calls
    to avoid rate limits and cost at scale.

Usage
-----
    from tools.llm_providers import get_synthesizer_model, get_subagent_model, active_provider

    model = get_synthesizer_model(api_key=state.get("anthropic_api_key", ""))
    provider = active_provider(anthropic_key=...)   # -> "claude" | "groq"
"""

import os
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq


# ---------------------------------------------------------------------------
# Provider constants
# ---------------------------------------------------------------------------

GROQ_MODEL   = "llama-3.3-70b-versatile"
CLAUDE_MODEL = "claude-sonnet-4-20250514"

ProviderName = Literal["claude", "groq"]


# ---------------------------------------------------------------------------
# Key resolution helpers
# ---------------------------------------------------------------------------

def _resolve_anthropic_key(api_key: str | None = None) -> str:
    """Return the first non-empty Anthropic key from arg or env."""
    return (api_key or "").strip() or os.environ.get("ANTHROPIC_API_KEY", "").strip()


def _resolve_groq_key(api_key: str | None = None) -> str:
    """Return the first non-empty Groq key from arg or env."""
    return (api_key or "").strip() or os.environ.get("GROQ_API_KEY", "").strip()


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def active_provider(anthropic_key: str | None = None) -> ProviderName:
    """
    Return which provider will be used for synthesis.

    Returns "claude" if a valid Anthropic key is available, else "groq".
    Does NOT make any API calls.
    """
    return "claude" if _resolve_anthropic_key(anthropic_key) else "groq"


def is_claude_active(anthropic_key: str | None = None) -> bool:
    return active_provider(anthropic_key) == "claude"


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def get_synthesizer_model(groq_api_key: str | None = None, anthropic_api_key: str | None = None):
    """
    Return the appropriate LLM for the synthesizer node.

    Prefers Claude Sonnet when ANTHROPIC_API_KEY is available.
    Falls back to Groq/Llama transparently.
    """
    anthropic_key = _resolve_anthropic_key(anthropic_api_key)

    if anthropic_key:
        return ChatAnthropic(
            model=CLAUDE_MODEL,
            api_key=anthropic_key,
            max_tokens=1024,
            temperature=0,
        )

    groq_key = _resolve_groq_key(groq_api_key)
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=groq_key,
        max_tokens=1024,
        temperature=0,
    )


def get_subagent_model(groq_api_key: str | None = None) -> ChatGroq:
    """
    Return the Groq model for subagent batch recommendation calls.
    Always Groq — subagents are parallel and high-volume.
    """
    key = _resolve_groq_key(groq_api_key)
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=key,
        max_tokens=1024,
        temperature=0,
    )


# ---------------------------------------------------------------------------
# Provider metadata  -  used by sidebar and synthesizer message log
# ---------------------------------------------------------------------------

PROVIDER_META: dict[ProviderName, dict] = {
    "claude": {
        "label":   "Claude Sonnet",
        "model":   CLAUDE_MODEL,
        "vendor":  "Anthropic",
        "color":   "#d2a8ff",   # purple  -  matches Anthropic brand
    },
    "groq": {
        "label":   "Llama 3.3 70B",
        "model":   GROQ_MODEL,
        "vendor":  "Groq",
        "color":   "#56d364",   # green  -  existing Groq color
    },
}


def provider_meta(anthropic_key: str | None = None) -> dict:
    """Return display metadata for the active synthesizer provider."""
    return PROVIDER_META[active_provider(anthropic_key)]