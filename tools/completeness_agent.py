"""
completeness_agent.py  -  Completeness Scorer + Prompt Agent for HOMEBASE

Given a free-text issue description and a predicted category, scores the description
against a per-category completeness rubric and surfaces targeted follow-up questions
for any missing or underspecified fields.

Returns:
    category:        canonical category used for scoring
    score:           float 0.0–1.0 (1.0 = all high-value fields present)
    missing_fields:  list of field names that are absent or vague
    questions:       list of targeted follow-up questions for the user
    error:           str | None

Provider: Groq (Llama 3.3 70B)
"""

import json
import os
import re
from typing import TypedDict

from langchain_groq import ChatGroq


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class CompletenessResult(TypedDict):
    category:       str          # canonical category used for scoring
    score:          float        # 0.0–1.0
    missing_fields: list[str]    # field names absent or vague
    questions:      list[str]    # targeted follow-up questions
    error:          str | None


# ---------------------------------------------------------------------------
# Per-category rubric
# Defines the high-value fields for each category that improve classification
# accuracy, routing confidence, and downstream agent recommendation quality.
# ---------------------------------------------------------------------------

CATEGORY_RUBRICS: dict[str, dict] = {
    "hvac": {
        "description": "HVAC systems (heating, ventilation, air conditioning, filters, ductwork)",
        "fields": [
            {"name": "symptom",        "description": "specific symptom or failure mode (noise, no heat, no cool, smell, etc.)"},
            {"name": "location",       "description": "which unit or zone is affected (furnace, AC, vent, thermostat, specific room)"},
            {"name": "duration",       "description": "how long the issue has been occurring"},
            {"name": "season_context", "description": "current season or temperature context (relevant for urgency)"},
            {"name": "last_service",   "description": "when the system was last serviced or filters changed"},
        ],
    },
    "plumbing": {
        "description": "Plumbing systems (pipes, drains, water heater, fixtures, hose bibs)",
        "fields": [
            {"name": "symptom",        "description": "specific symptom (leak, clog, low pressure, no hot water, noise, etc.)"},
            {"name": "location",       "description": "specific fixture or area affected (kitchen, bathroom, basement, yard)"},
            {"name": "duration",       "description": "how long the issue has been occurring"},
            {"name": "water_damage",   "description": "whether visible water damage or moisture is present"},
            {"name": "severity",       "description": "active drip vs slow leak vs intermittent vs no flow"},
        ],
    },
    "electrical": {
        "description": "Electrical systems (outlets, panels, fixtures, GFCI, wiring)",
        "fields": [
            {"name": "symptom",        "description": "specific symptom (tripping breaker, flickering, no power, sparking, burning smell)"},
            {"name": "location",       "description": "specific outlet, circuit, room, or panel affected"},
            {"name": "duration",       "description": "how long the issue has been occurring"},
            {"name": "safety_signal",  "description": "any signs of heat, burning smell, sparking, or shock risk"},
            {"name": "scope",          "description": "single outlet/fixture vs circuit vs whole-house"},
        ],
    },
    "appliance": {
        "description": "Home appliances (dishwasher, dryer, refrigerator, washer, oven)",
        "fields": [
            {"name": "appliance",      "description": "which specific appliance is affected"},
            {"name": "symptom",        "description": "specific failure mode (not starting, not heating, leaking, noise, error code)"},
            {"name": "duration",       "description": "how long the issue has been occurring"},
            {"name": "error_code",     "description": "any error code or indicator light shown"},
            {"name": "age",            "description": "approximate age of the appliance (helps assess repair vs replace)"},
        ],
    },
    "general": {
        "description": "General home maintenance (exterior, gutters, paint, caulk, safety devices, landscaping)",
        "fields": [
            {"name": "symptom",        "description": "specific issue or observed condition"},
            {"name": "location",       "description": "specific area of the home affected (exterior, interior, roof, foundation, etc.)"},
            {"name": "duration",       "description": "how long the condition has been present or when first noticed"},
            {"name": "scope",          "description": "size or extent of the affected area"},
            {"name": "safety_risk",    "description": "whether the issue poses any safety or structural risk"},
        ],
    },
}

_CANONICAL_CATEGORIES = set(CATEGORY_RUBRICS.keys())

_SYSTEM_PROMPT_TEMPLATE = """\
You are a home maintenance intake assistant. Your job is to evaluate how complete a user's issue description is, given that it has been classified into the "{category}" category ({category_description}).

The high-value fields for this category are:
{fields_list}

Evaluate the user's description against these fields. A field is "present" if the description contains clear, specific information for it. A field is "missing" or "vague" if the information is absent, ambiguous, or too general to be useful.

Respond ONLY with valid JSON — no preamble, no markdown. Use exactly these keys:
{{
  "score": <0.0–1.0, proportion of high-value fields that are clearly present>,
  "missing_fields": [<list of field names that are absent or vague>],
  "questions": [<list of specific, targeted follow-up questions — one per missing field, phrased naturally>]
}}

Scoring guidelines:
  1.0 = all fields clearly present
  0.8 = 1 field missing or vague
  0.6 = 2 fields missing or vague
  0.4 = 3 fields missing or vague
  0.2 = 4+ fields missing or vague
  0.0 = description too vague to assess any field

Question guidelines:
  - Ask one question per missing field
  - Phrase questions naturally, as a knowledgeable assistant would ask a homeowner
  - Be specific to the category and symptom described
  - Do not ask about fields that are already clearly present
  - If all fields are present, return an empty questions list
"""


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _get_model(api_key: str | None = None) -> ChatGroq:
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=key,
        max_tokens=512,
        temperature=0,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_category(category: str) -> str:
    """Map predicted quadrant-preview category or free-form string to canonical rubric key."""
    c = category.strip().lower()
    if c in _CANONICAL_CATEGORIES:
        return c
    # Fuzzy fallback
    for key in _CANONICAL_CATEGORIES:
        if key in c:
            return key
    return "general"


def _build_system_prompt(category: str) -> str:
    rubric = CATEGORY_RUBRICS[category]
    fields_list = "\n".join(
        f"  - {f['name']}: {f['description']}"
        for f in rubric["fields"]
    )
    return _SYSTEM_PROMPT_TEMPLATE.format(
        category=category,
        category_description=rubric["description"],
        fields_list=fields_list,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_completeness(
    description: str,
    category: str,
    api_key: str | None = None,
) -> CompletenessResult:
    """
    Score the completeness of a free-text issue description for a given category.

    Returns a CompletenessResult dict. Never raises — errors captured in 'error' field.

    Args:
        description: free-text issue description from the user
        category:    predicted or user-specified category (hvac/plumbing/electrical/appliance/general)
        api_key:     Groq API key (falls back to GROQ_API_KEY env var)
    """
    description = description.strip()
    if len(description) < 8:
        return CompletenessResult(
            category=category,
            score=0.0,
            missing_fields=[],
            questions=[],
            error="Description too short to score.",
        )

    canonical = _normalize_category(category)

    try:
        model = _get_model(api_key=api_key)
        system_prompt = _build_system_prompt(canonical)

        response = model.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": description},
        ])

        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()

        parsed = json.loads(raw)

        try:
            score = float(parsed.get("score", 0.5))
            score = max(0.0, min(1.0, score))
        except (TypeError, ValueError):
            score = 0.5

        missing_fields = [str(f) for f in parsed.get("missing_fields", [])]
        questions      = [str(q) for q in parsed.get("questions", [])]

        return CompletenessResult(
            category=canonical,
            score=score,
            missing_fields=missing_fields,
            questions=questions,
            error=None,
        )

    except Exception as e:
        return CompletenessResult(
            category=canonical,
            score=0.0,
            missing_fields=[],
            questions=[],
            error=f"{type(e).__name__}: {e}",
        )