"""
tools/update_agent.py  -  Natural language registry command agent.

Handles add, update, and close operations via a single NL instruction.
Single-shot LLM call — not a graph node.
"""

import json
import re
from tools.llm_tools import get_model
from tools.registry_tools import update_item, add_item, close_item, get_registry

# ---------------------------------------------------------------------------
# Hybrid intent router  —  heuristic first, LLM fallback for ambiguous input
# ---------------------------------------------------------------------------

import re as _re

# Strong run-trigger keywords — if matched and no item ID present, route to graph
_RUN_KEYWORDS = re.compile(
    r'\b(review|audit|inspect|check|assess|what needs|morning|weekly|daily|briefing|'
    r'status|run|scan|immediate|urgent|critical|attention|survey|overview|report)\b',
    re.IGNORECASE,
)

# Item ID pattern — strong signal for registry op
_ITEM_ID_PAT = re.compile(r'\b([A-Z]{2,6}-\d{3})\b', re.IGNORECASE)

# Registry action keywords — strong signal even without an ID (for add)
_REGISTRY_KEYWORDS = re.compile(
    r'\b(add|create|log|track|new item|close|remove|mark as|update|edit|change|raise|lower|'
    r'set urgency|set impact|reset the clock|in progress|resolved|fixed)\b',
    re.IGNORECASE,
)

# Chart keywords — user wants a visualization
_CHART_KEYWORDS = re.compile(
    r'\b(chart|plot|graph|visualize|visualise|histogram|scatter|bar chart|pie chart|'
    r'line chart|heatmap|show me a chart|show me a graph|trend|distribution)\b',
    re.IGNORECASE,
)

# RCA keywords — user wants a cross-item root cause analysis
_RCA_KEYWORDS = re.compile(
    r'\b(root cause|rca|root-cause|analyze patterns|analyse patterns|systemic|'
    r'what\'s driving|whats driving|common factors|pattern analysis|'
    r'underlying cause|why are|what\'s causing|whats causing|cross.item)\b',
    re.IGNORECASE,
)

# Category extraction for scoped RCA
_CATEGORY_MAP = {
    "hvac":        "hvac",
    "plumbing":    "plumbing",
    "plumb":       "plumbing",
    "electrical":  "electrical",
    "electric":    "electrical",
    "appliance":   "appliance",
    "appliances":  "appliance",
    "general":     "general",
}
_CATEGORY_PAT = re.compile(
    r'\b(hvac|plumbing|plumb|electrical|electric|appliance|appliances|general)\b',
    re.IGNORECASE,
)


def extract_rca_category(instruction: str) -> str | None:
    """Return canonical category name if instruction scopes the RCA, else None."""
    m = _CATEGORY_PAT.search(instruction)
    if m:
        return _CATEGORY_MAP.get(m.group(1).lower())
    return None

AMBIGUOUS_INTENT_PROMPT = """You are a home management assistant command router.

Given a user input, return ONLY a JSON object with:
- "intent": one of "run", "add", "update", "close", "chart", or "rca"

Definitions:
- "run": trigger an agent analysis run (e.g. "weekly review", "check hvac", "what needs attention")
- "add": add a new item to the registry
- "update": change fields on an existing registry item
- "close": close/remove a registry item
- "chart": generate a chart or visualization from registry or run history data
- "rca": perform a cross-item root cause analysis (e.g. "root cause", "what's driving these issues", "systemic analysis")

Return ONLY valid JSON. No explanation, no markdown, no preamble."""


def classify_input(instruction: str, api_key: str | None = None) -> str:
    """
    Hybrid classifier. Returns one of: "run", "add", "update", "close".

    Heuristic pass first — LLM only called when ambiguous.
    """
    has_item_id      = bool(_ITEM_ID_PAT.search(instruction))
    has_run_keyword  = bool(_RUN_KEYWORDS.search(instruction))
    has_reg_keyword  = bool(_REGISTRY_KEYWORDS.search(instruction))
    has_chart_kw     = bool(_CHART_KEYWORDS.search(instruction))
    has_rca_kw       = bool(_RCA_KEYWORDS.search(instruction))

    # Unambiguous: RCA keyword → rca intent (highest priority, before chart/run)
    if has_rca_kw and not has_item_id and not has_reg_keyword:
        return "rca"

    # Unambiguous: chart keyword → chart intent (before run/query to avoid misroute)
    if has_chart_kw and not has_item_id and not has_reg_keyword:
        return "chart"

    # Unambiguous: item ID present → registry op
    if has_item_id and not has_run_keyword:
        return "registry"

    # Unambiguous: run keyword, no registry signals
    if has_run_keyword and not has_item_id and not has_reg_keyword:
        return "run"

    # Unambiguous: registry keyword, no run/query signals
    if has_reg_keyword and not has_run_keyword:
        return "registry"

    # Ambiguous — fall back to LLM
    try:
        model = get_model(api_key=api_key)
        response = model.invoke([
            {"role": "system", "content": AMBIGUOUS_INTENT_PROMPT},
            {"role": "user",   "content": instruction},
        ])
        raw = response.content.strip().strip("```json").strip("```").strip()
        parsed = json.loads(raw)
        intent = parsed.get("intent", "run")
        if intent in ("chart", "rca"):
            return intent
        return "run" if intent == "run" else "registry"
    except Exception:
        return "run"



# ---------------------------------------------------------------------------
# Intent routing
# ---------------------------------------------------------------------------

INTENT_PROMPT = """You are a home registry command router.

Given a natural language instruction, return ONLY a JSON object with:
- "intent": one of "add", "update", or "close"
- "item_id": the item ID if mentioned (e.g. "APP-001"), or null

Rules:
- "add", "create", "new item", "log", "track" -> intent: "add"
- "close", "remove", "mark as done", "resolve", "fixed", "completed" -> intent: "close"
- Everything else (edit, update, change, mark as in progress, raise urgency, etc.) -> intent: "update"
- Extract item IDs matching pattern like APP-001, HVAC-002, PLB-003, etc.
- Return ONLY valid JSON. No explanation, no markdown, no preamble."""


def route_intent(instruction: str, api_key: str | None = None) -> dict:
    """Return {"intent": "add"|"update"|"close", "item_id": str|None}"""
    model = get_model(api_key=api_key)
    try:
        response = model.invoke([
            {"role": "system", "content": INTENT_PROMPT},
            {"role": "user",   "content": instruction},
        ])
        raw = response.content.strip().strip("```json").strip("```").strip()
        return json.loads(raw)
    except Exception:
        # Fallback: regex sniff for ID, default to update
        id_match = re.search(r'\b([A-Z]{2,6}-\d{3})\b', instruction, re.IGNORECASE)
        return {"intent": "update", "item_id": id_match.group(1).upper() if id_match else None}


# ---------------------------------------------------------------------------
# Update path
# ---------------------------------------------------------------------------

UPDATE_PROMPT = """You are a home registry update assistant. You will receive:
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
    model = get_model(api_key=api_key)
    user_msg = f"Current item:\n{json.dumps(item, indent=2)}\n\nUser instruction: {instruction}"
    try:
        response = model.invoke([
            {"role": "system", "content": UPDATE_PROMPT},
            {"role": "user",   "content": user_msg},
        ])
        raw = response.content.strip().strip("```json").strip("```").strip()
        parsed = json.loads(raw)

        allowed = {"title", "description", "urgency", "impact", "status", "days_since_update"}
        safe = {}
        for k, v in parsed.items():
            if k not in allowed:
                continue
            if k in ("urgency", "impact"):
                v = round(max(0.0, min(1.0, float(v))), 2)
            elif k == "days_since_update":
                v = max(0, int(v))
            elif k == "status" and v not in ("open", "in_progress", "closed"):
                continue
            safe[k] = v
        return safe
    except Exception as e:
        return {"_error": str(e)}


def apply_update(item_id: str, item: dict, instruction: str, api_key: str | None = None) -> tuple[dict | None, dict]:
    changes = interpret_update(item, instruction, api_key=api_key)
    if not changes or "_error" in changes:
        return None, changes
    updated = update_item(item_id, changes)
    return updated, changes


# ---------------------------------------------------------------------------
# Add path
# ---------------------------------------------------------------------------

ADD_PROMPT = """You are a home registry assistant. Extract item details from a natural language instruction.

Return ONLY a JSON object with:
- "title": string (required, short description)
- "description": string (additional context, can be empty string)
- "category": one of "hvac", "plumbing", "electrical", "appliance", "general"
- "urgency": float 0.0-1.0 (default 0.5 if not specified)
- "impact": float 0.0-1.0 (default 0.5 if not specified)

Rules:
- Infer category from context (e.g. "leaky faucet" -> plumbing, "breaker" -> electrical)
- urgency/impact as percentage (e.g. "70%") -> divide by 100
- Clamp urgency and impact to [0.0, 1.0]
- Return ONLY valid JSON. No explanation, no markdown, no preamble."""


def interpret_add(instruction: str, api_key: str | None = None) -> dict:
    model = get_model(api_key=api_key)
    try:
        response = model.invoke([
            {"role": "system", "content": ADD_PROMPT},
            {"role": "user",   "content": instruction},
        ])
        raw = response.content.strip().strip("```json").strip("```").strip()
        parsed = json.loads(raw)

        # Validate
        valid_cats = {"hvac", "plumbing", "electrical", "appliance", "general"}
        if not parsed.get("title", "").strip():
            return {"_error": "Could not extract a title from that instruction."}
        parsed["category"] = parsed.get("category", "general").lower()
        if parsed["category"] not in valid_cats:
            parsed["category"] = "general"
        parsed["urgency"] = round(max(0.0, min(1.0, float(parsed.get("urgency", 0.5)))), 2)
        parsed["impact"]  = round(max(0.0, min(1.0, float(parsed.get("impact",  0.5)))), 2)
        parsed["description"] = parsed.get("description", "")
        return parsed
    except Exception as e:
        return {"_error": str(e)}


def execute_add(instruction: str, api_key: str | None = None) -> tuple[dict | None, dict]:
    """Interpret instruction and add a new item. Returns (new_item, fields) or (None, {_error})."""
    fields = interpret_add(instruction, api_key=api_key)
    if "_error" in fields:
        return None, fields
    new_item = add_item(
        category=fields["category"],
        title=fields["title"],
        description=fields["description"],
        urgency=fields["urgency"],
        impact=fields["impact"],
    )
    return new_item, fields


# ---------------------------------------------------------------------------
# Unified command entrypoint
# ---------------------------------------------------------------------------

def execute_command(instruction: str, api_key: str | None = None) -> dict:
    """
    Route a free-text instruction to add / update / close and execute it.

    Returns a result dict:
      {
        "intent":   "add"|"update"|"close",
        "item_id":  str | None,
        "item":     dict | None,   # resulting item state
        "changes":  dict,          # fields changed / added
        "error":    str | None,
      }
    """
    routed = route_intent(instruction, api_key=api_key)
    intent  = routed.get("intent", "update")
    item_id = routed.get("item_id")

    # --- ADD ---
    if intent == "add":
        new_item, fields = execute_add(instruction, api_key=api_key)
        if new_item is None:
            return {"intent": "add", "item_id": None, "item": None, "changes": {}, "error": fields.get("_error")}
        return {"intent": "add", "item_id": new_item["id"], "item": new_item, "changes": fields, "error": None}

    # --- CLOSE / UPDATE — need to resolve item_id ---
    if not item_id:
        return {"intent": intent, "item_id": None, "item": None, "changes": {},
                "error": "No item ID found in instruction. Try: 'close APP-001' or 'mark APP-001 as in progress'."}

    registry = {i["id"]: i for i in get_registry()}
    # Case-insensitive lookup
    item_id_upper = item_id.upper()
    item = registry.get(item_id_upper)
    if not item:
        return {"intent": intent, "item_id": item_id_upper, "item": None, "changes": {},
                "error": f"Item {item_id_upper} not found in registry."}

    # --- CLOSE ---
    if intent == "close":
        success = close_item(item_id_upper)
        if not success:
            return {"intent": "close", "item_id": item_id_upper, "item": None, "changes": {},
                    "error": f"Could not close {item_id_upper}."}
        return {"intent": "close", "item_id": item_id_upper, "item": item, "changes": {"status": "closed"}, "error": None}

    # --- UPDATE ---
    updated, changes = apply_update(item_id_upper, item, instruction, api_key=api_key)
    if "_error" in changes:
        return {"intent": "update", "item_id": item_id_upper, "item": None, "changes": {}, "error": changes["_error"]}
    if not changes:
        return {"intent": "update", "item_id": item_id_upper, "item": item, "changes": {},
                "error": "Could not infer any changes. Be more specific or include the item ID (e.g. 'raise urgency on APP-001 to 0.9')."}
    return {"intent": "update", "item_id": item_id_upper, "item": updated, "changes": changes, "error": None}