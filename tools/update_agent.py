"""
tools/update_agent.py  -  Natural language registry update agent.

Takes a free-text instruction and the current item state, uses Groq/Llama
to interpret the intent, and returns a validated update dict ready for
registry_tools.update_item().

Single-shot LLM call — not a graph node.
"""

import json
from tools.llm_tools import get_model
from tools.registry_tools import update_item

SYSTEM_PROMPT = """You are a home registry update assistant. You will receive:
1. The current state of a registry item
2. A natural language instruction from the user

Your job is to interpret the instruction and return ONLY a JSON object with the fields to update.

Allowed fields and their types:
- title: string
- description: string
- urgency: float between 0.0 and 1.0
- impact: float between 0.0 and 1.0
- status: string — MUST be one of exactly: "open", "in_progress", "closed"
- days_since_update: integer (0 or greater)

Status mapping rules (apply these exactly):
- "mark as resolved", "close", "done", "fixed", "complete", "completed" -> {"status": "closed"}
- "mark as in progress", "in progress", "start", "started", "working on it", "underway", "begun", "active" -> {"status": "in_progress"}
- "reopen", "mark as open", "back to open" -> {"status": "open"}
- "reset the clock", "just updated", "addressed today", "updated today" -> {"days_since_update": 0}
- urgency/impact as percentage (e.g. "70%") -> divide by 100 (0.70)
- Clamp urgency and impact to [0.0, 1.0]

Other rules:
- Only include fields the instruction explicitly asks to change
- If nothing can be reliably inferred, return {}
- Return ONLY valid JSON. No explanation, no markdown, no preamble."""


def interpret_update(item: dict, instruction: str, api_key: str | None = None) -> dict:
    """
    Use LLM to interpret a natural language update instruction for an item.
    Returns a dict of field->value updates (may be empty if nothing inferred).
    """
    model = get_model(api_key=api_key)

    user_msg = f"""Current item:
{json.dumps(item, indent=2)}

User instruction: {instruction}"""

    try:
        response = model.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        parsed = json.loads(raw)

        # Validate and sanitize
        allowed = {"title", "description", "urgency", "impact", "status", "days_since_update"}
        safe = {}
        for k, v in parsed.items():
            if k not in allowed:
                continue
            if k in ("urgency", "impact"):
                v = round(max(0.0, min(1.0, float(v))), 2)
            elif k == "days_since_update":
                v = max(0, int(v))
            elif k == "status" and v not in ("open", "closed"):
                continue
            safe[k] = v
        return safe

    except Exception as e:
        return {"_error": str(e)}


def apply_update(item_id: str, item: dict, instruction: str, api_key: str | None = None) -> tuple[dict | None, dict]:
    """
    Interpret the instruction and apply the update to the registry.

    Returns:
        (updated_item, changes) where:
          - updated_item is the new item state (or None on failure)
          - changes is the dict of fields that were changed
    """
    changes = interpret_update(item, instruction, api_key=api_key)

    if not changes or "_error" in changes:
        return None, changes

    updated = update_item(item_id, changes)
    return updated, changes