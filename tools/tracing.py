"""
tools/tracing.py  -  LangSmith tracing integration for HOMEBASE.

LangChain/LangGraph auto-instruments all LLM calls and graph nodes when
LANGCHAIN_TRACING_V2=true is set in the environment. This module provides:

  - init_tracing()           Called once at app startup to activate tracing
  - is_tracing_enabled()     Returns True if tracing env vars are present
  - get_run_metadata()       Builds the metadata + tags dict for a graph invocation

Usage in app.py:
    from tools.tracing import init_tracing, is_tracing_enabled, get_run_metadata

    init_tracing()  # call once at module level

    # Merge into LangGraph config per run:
    config = {"configurable": {"thread_id": thread_id}, **get_run_metadata(trigger, filter)}
"""

import os
from datetime import datetime


# ---------------------------------------------------------------------------
# Tracing activation
# ---------------------------------------------------------------------------

def init_tracing() -> bool:
    """
    Activate LangSmith tracing if LANGCHAIN_API_KEY is present in the environment.
    Sets LANGCHAIN_TRACING_V2=true and defaults LANGCHAIN_PROJECT to 'homebase'
    if not already set.

    Returns True if tracing was activated, False otherwise.
    Safe to call multiple times — idempotent.
    """
    api_key = os.environ.get("LANGCHAIN_API_KEY", "").strip()
    if not api_key:
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    if not os.environ.get("LANGCHAIN_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = "homebase"

    return True


def is_tracing_enabled() -> bool:
    """Return True if LangSmith tracing is currently active."""
    return (
        os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
        and bool(os.environ.get("LANGCHAIN_API_KEY", "").strip())
    )


def get_project_name() -> str:
    """Return the active LangSmith project name."""
    return os.environ.get("LANGCHAIN_PROJECT", "homebase")


# ---------------------------------------------------------------------------
# Per-run metadata
# ---------------------------------------------------------------------------

def get_run_metadata(
    trigger: str,
    category_filter: list[str] | None = None,
) -> dict:
    """
    Build a LangGraph-compatible config dict fragment with LangSmith run metadata.

    Merges into the graph config:
        config = {"configurable": {"thread_id": thread_id}, **get_run_metadata(...)}

    Tags appear as filterable labels in the LangSmith UI.
    Metadata fields are searchable key-value pairs on each trace.

    Returns an empty dict if tracing is not enabled (safe to merge regardless).
    """
    if not is_tracing_enabled():
        return {}

    tags = ["homebase", f"trigger:{_slug(trigger)}"]
    if category_filter:
        tags += [f"category:{c}" for c in category_filter]
    else:
        tags.append("category:all")

    metadata = {
        "trigger":         trigger,
        "category_filter": category_filter or "all",
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
        "project":         get_project_name(),
        "system":          "homebase-v1.5.0",
        "model":           "llama-3.3-70b-versatile",
        "agents":          "orchestrator,hvac,plumbing,electrical,appliance,general,synthesizer",
    }

    return {
        "tags":     tags,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _slug(text: str) -> str:
    """Convert a trigger phrase to a URL-safe tag slug."""
    return text.lower().strip().replace(" ", "-")[:40]