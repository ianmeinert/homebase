"""
quadrant_preview.py  -  Predictive Quadrant Preview for HOMEBASE

Given a free-text issue description, predicts the Urgency × Impact quadrant
classification before any agent run is triggered.

Returns:
    quadrant: one of HU/HI, HU/LI, LU/HI, LU/LI
    confidence: float 0.0–1.0
    rationale: one-sentence explanation

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

class QuadrantPreview(TypedDict):
    quadrant: str       # HU/HI | HU/LI | LU/HI | LU/LI
    confidence: float   # 0.0–1.0
    rationale: str      # one sentence
    error: str | None


_VALID_QUADRANTS = {"HU/HI", "HU/LI", "LU/HI", "LU/LI"}

_SYSTEM_PROMPT = """\
You are a home maintenance triage assistant. Given a brief description of a home issue, classify it into one of four urgency/impact quadrants and estimate your confidence.

Quadrant definitions:
  HU/HI — High Urgency, High Impact: Safety risk or active damage; must act immediately (e.g. furnace out in winter, major leak, no power)
  HU/LI — High Urgency, Low Impact:  Annoying but not damaging; time-sensitive but limited consequences (e.g. broken door handle, minor pest, cosmetic crack)
  LU/HI — Low Urgency, High Impact:  Significant long-term consequence but not immediately critical; plan and schedule (e.g. aging roof, foundation hairline, deferred HVAC service)
  LU/LI — Low Urgency, Low Impact:   Routine maintenance or cosmetic; address when convenient (e.g. touch-up paint, minor caulk gap, lightbulb replacement)

Respond ONLY with valid JSON — no preamble, no markdown. Use exactly these keys:
{
  "quadrant": "<HU/HI|HU/LI|LU/HI|LU/LI>",
  "confidence": <0.0–1.0>,
  "rationale": "<one sentence explaining the classification>"
}

Confidence guidelines:
  0.85–1.0 = clear signals (safety risk, seasonal context, active damage described)
  0.65–0.84 = moderate signals (issue described but severity unclear)
  0.40–0.64 = weak signals (vague description, multiple possible classifications)
  <0.40 = insufficient information to classify reliably
"""


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _get_model(api_key: str | None = None) -> ChatGroq:
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=key,
        max_tokens=256,
        temperature=0,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_quadrant(description: str, api_key: str | None = None) -> QuadrantPreview:
    """
    Predict the urgency/impact quadrant for a free-text issue description.

    Returns a QuadrantPreview dict. Never raises — errors are captured in
    the 'error' field and the caller should handle gracefully.
    """
    description = description.strip()
    if len(description) < 8:
        return QuadrantPreview(
            quadrant="",
            confidence=0.0,
            rationale="",
            error="Description too short to classify.",
        )

    try:
        model = _get_model(api_key=api_key)
        response = model.invoke([
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": description},
        ])

        raw = response.content.strip()

        # Strip markdown fences if model adds them
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()

        parsed = json.loads(raw)

        quadrant = parsed.get("quadrant", "").strip()
        if quadrant not in _VALID_QUADRANTS:
            raise ValueError(f"Unexpected quadrant value: {quadrant!r}")

        try:
            confidence = float(parsed.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5

        rationale = str(parsed.get("rationale", "")).strip()

        return QuadrantPreview(
            quadrant=quadrant,
            confidence=confidence,
            rationale=rationale,
            error=None,
        )

    except Exception as e:
        return QuadrantPreview(
            quadrant="",
            confidence=0.0,
            rationale="",
            error=f"{type(e).__name__}: {e}",
        )