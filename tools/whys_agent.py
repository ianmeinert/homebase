"""
tools/whys_agent.py  —  5 Whys causal chain agent.

Operates directly on registry items for a given category (or the highest-severity
open category if none specified). Builds a structured causal chain and produces
a root cause — no prior RCA result required.

Public API:
  run_whys(instruction, category=None, api_key=None) -> dict

Returns:
  - category:           the category analyzed
  - problem_statement:  dominant problem pattern across items
  - causal_chain:       list of {level, why, because} dicts (5 levels)
  - root_cause:         final root cause statement
  - corrective_action:  recommended systemic fix
  - confidence:         float 0.0-1.0
  - confidence_rationale: one-sentence explanation
  - item_count:         number of items analyzed
  - error:              None or error string
"""

import json
import datetime
from tools.llm_tools import get_model
from tools.db import get_conn


# ---------------------------------------------------------------------------
# Registry loaders
# ---------------------------------------------------------------------------

def _load_category_items(category: str) -> list[dict]:
    """Load all open/in-progress registry items for a given category."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, category, title, description, urgency, impact, updated_at, status "
        "FROM registry WHERE category = ? AND status != 'closed'",
        (category,)
    ).fetchall()
    keys = ["id", "category", "title", "description", "urgency", "impact", "updated_at", "status"]
    records = []
    for r in rows:
        d = dict(zip(keys, r))
        try:
            updated = datetime.datetime.fromisoformat(d["updated_at"])
            d["days_since_update"] = (datetime.datetime.now() - updated).days
        except (ValueError, TypeError):
            d["days_since_update"] = 0
        records.append(d)
    return records


def _highest_severity_category() -> str | None:
    """Return the category with the highest average urgency * impact among open items."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT category, AVG(urgency * impact) as score "
        "FROM registry WHERE status != 'closed' "
        "GROUP BY category ORDER BY score DESC LIMIT 1"
    ).fetchall()
    return rows[0][0] if rows else None


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

WHYS_SYSTEM_PROMPT = """You are an expert root cause analysis facilitator performing a
structured 5 Whys causal chain analysis on a set of home maintenance registry items
within a specific category.

You will receive:
- The category being analyzed
- All open/in-progress registry items in that category (title, description, urgency,
  impact, days since last update, status)
- The user's instruction (may specify focus area within the category)

Your task:
1. Identify the dominant problem pattern across the items in this category
2. Perform a rigorous 5 Whys causal chain — each "because" becomes the next "why"
3. The chain must reach a systemic root cause, not just re-describe symptoms
4. Recommend a corrective action that addresses the root cause

Return ONLY a JSON object with this exact structure:

{
  "category": "<the category analyzed>",
  "problem_statement": "<one sentence describing the dominant problem pattern across these items>",
  "causal_chain": [
    {"level": 1, "why": "<observable problem statement>", "because": "<first causal factor>"},
    {"level": 2, "why": "<restate the because from level 1 as a why>", "because": "<deeper cause>"},
    {"level": 3, "why": "<restate the because from level 2 as a why>", "because": "<deeper cause>"},
    {"level": 4, "why": "<restate the because from level 3 as a why>", "because": "<deeper cause>"},
    {"level": 5, "why": "<restate the because from level 4 as a why>", "because": "<root cause — systemic, not symptomatic>"}
  ],
  "root_cause": "<one clear sentence stating the systemic root cause reached at level 5>",
  "corrective_action": "<specific, actionable systemic fix that addresses the root cause>",
  "confidence": <float 0.0-1.0>,
  "confidence_rationale": "<one sentence explaining confidence level>"
}

Rules:
- Each "because" at level N must directly cause the "why" at level N+1
- Chain must progress: symptom -> contributing factor -> underlying cause -> systemic cause -> root cause
- Root cause must be systemic (process, policy, behavior, structural gap) -- not "the item needs repair"
- Corrective action must address the root cause, not re-list the items
- Confidence reflects data quality:
  - < 0.5: few items, weak pattern
  - 0.5-0.7: moderate evidence, some inference required
  - 0.7-0.9: clear pattern across multiple items
  - > 0.9: strong convergent evidence
- Return ONLY valid JSON. No explanation, no markdown, no preamble."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_whys(
    instruction: str,
    category: str | None = None,
    api_key: str | None = None,
) -> dict:
    """
    Perform a 5 Whys causal chain analysis on registry items for a given category.
    If category is None, uses the highest-severity open category automatically.

    Returns a dict with keys:
      category, problem_statement, causal_chain, root_cause,
      corrective_action, confidence, confidence_rationale, item_count, error
    """
    _empty = {
        "category": category or "", "problem_statement": "", "causal_chain": [],
        "root_cause": "", "corrective_action": "",
        "confidence": 0.0, "confidence_rationale": "", "item_count": 0, "error": None,
    }

    # Resolve category
    resolved = category
    if not resolved:
        resolved = _highest_severity_category()
    if not resolved:
        return {**_empty, "error": "No open items found in registry."}

    resolved = resolved.lower()

    # Load items
    items = _load_category_items(resolved)
    if not items:
        return {**_empty, "category": resolved,
                "error": f"No open items found in '{resolved}' category."}

    # Trim descriptions to avoid token bloat
    items_trimmed = [
        {
            "id":                i["id"],
            "title":             i["title"],
            "description":       (i["description"] or "")[:120],
            "urgency":           i["urgency"],
            "impact":            i["impact"],
            "days_since_update": i["days_since_update"],
            "status":            i["status"],
        }
        for i in items
    ]

    user_msg = (
        f"Category: {resolved}\n"
        f"User instruction: {instruction}\n\n"
        f"Registry items ({len(items_trimmed)} open/in-progress):\n"
        f"{json.dumps(items_trimmed, indent=2)}"
    )

    try:
        model = get_model(api_key=api_key)
        response = model.invoke([
            {"role": "system", "content": WHYS_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ])
        raw = response.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        result = json.loads(raw)

        chain = result.get("causal_chain", [])
        if len(chain) < 5:
            return {**_empty, "category": resolved, "item_count": len(items),
                    "error": f"Incomplete causal chain ({len(chain)} levels). Try again."}

        return {
            "category":             result.get("category", resolved),
            "problem_statement":    result.get("problem_statement", ""),
            "causal_chain":         chain,
            "root_cause":           result.get("root_cause", ""),
            "corrective_action":    result.get("corrective_action", ""),
            "confidence":           float(result.get("confidence", 0.0)),
            "confidence_rationale": result.get("confidence_rationale", ""),
            "item_count":           len(items),
            "error":                None,
        }

    except json.JSONDecodeError as e:
        return {**_empty, "category": resolved, "item_count": len(items),
                "error": f"5 Whys parse error: {e}"}
    except Exception as e:
        return {**_empty, "category": resolved, "item_count": len(items),
                "error": f"5 Whys failed: {e}"}