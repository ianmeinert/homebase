"""
llm_tools.py  -  LLM-backed recommendation tools for HOMEBASE specialist subagents.

Each domain function batches all assigned items into a single LLM call and
returns a list of structured Recommendation dicts. JSON output is enforced
via system prompt  -  no response parsing ambiguity.

Phase 1 (POC): rule-based pattern matching in subagent_tools.py
Phase 2 (this): Gemini generates contextual recommendations per item
"""

import json
import os
from typing import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Recommendation(TypedDict):
    item_id: str
    action: str
    estimated_effort: str
    estimated_cost: str
    priority_note: str
    agent: str


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def get_model() -> ChatGoogleGenerativeAI:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        max_output_tokens=1024,
        temperature=0,
    )


# ---------------------------------------------------------------------------
# Shared call logic
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a specialist home maintenance advisor. You will receive a list of home maintenance items assigned to you. For each item, produce a structured recommendation.

Respond ONLY with a valid JSON array  -  no preamble, no markdown, no explanation. Each element must be an object with exactly these keys:
- item_id: string (copy from input)
- action: string (specific, actionable next step  -  1-2 sentences)
- estimated_effort: string (e.g. "15 min DIY", "1-2 hr contractor")
- estimated_cost: string (e.g. "$15-40", "$150-250 contractor")
- priority_note: string (1 sentence on consequence of deferral)
- agent: string (copy from input)

Be specific to the item. Reference the urgency, impact, and days_since_update in your reasoning but do not include them verbatim in the output fields."""


def _call_llm(domain: str, agent_name: str, items: list[dict]) -> list[Recommendation]:
    """
    Make a single LLM call for all items assigned to this subagent.
    Returns a list of Recommendation dicts.
    Falls back to a safe default if the call fails or JSON is malformed.
    """
    if not items:
        return []

    model = get_model()

    items_payload = [
        {
            "item_id": item["id"],
            "title": item["title"],
            "description": item["description"],
            "urgency": item["urgency"],
            "impact": item["impact"],
            "days_since_update": item["days_since_update"],
            "quadrant": item["quadrant"],
            "agent": agent_name,
        }
        for item in items
    ]

    user_msg = f"Domain: {domain}\n\nItems:\n{json.dumps(items_payload, indent=2)}"

    try:
        # Gemini does not support system role -- fold into user message
        combined = f"{SYSTEM_PROMPT}\n\n{user_msg}"
        response = model.invoke([
            {"role": "user", "content": combined},
        ])

        raw = response.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)

        # Validate and normalize
        results = []
        for rec in parsed:
            results.append(Recommendation(
                item_id=rec.get("item_id", ""),
                action=rec.get("action", "See a specialist."),
                estimated_effort=rec.get("estimated_effort", "Unknown"),
                estimated_cost=rec.get("estimated_cost", "Unknown"),
                priority_note=rec.get("priority_note", ""),
                agent=agent_name,
            ))
        return results

    except Exception as e:
        # Graceful fallback  -  never crash the graph on LLM failure
        return [
            Recommendation(
                item_id=item["id"],
                action=f"Inspect and address: {item['title']}",
                estimated_effort="Assess on site",
                estimated_cost="TBD",
                priority_note=f"LLM recommendation unavailable ({type(e).__name__}). Manual review required.",
                agent=agent_name,
            )
            for item in items
        ]


# ---------------------------------------------------------------------------
# Domain functions  -  one per subagent
# ---------------------------------------------------------------------------

def hvac_recommend_llm(items: list[dict]) -> list[Recommendation]:
    return _call_llm(
        domain="HVAC systems (heating, ventilation, air conditioning, filters, ductwork)",
        agent_name="HVACAgent",
        items=items,
    )


def plumbing_recommend_llm(items: list[dict]) -> list[Recommendation]:
    return _call_llm(
        domain="Plumbing systems (pipes, drains, water heater, fixtures, hose bibs)",
        agent_name="PlumbingAgent",
        items=items,
    )


def electrical_recommend_llm(items: list[dict]) -> list[Recommendation]:
    return _call_llm(
        domain="Electrical systems (outlets, panels, fixtures, GFCI, wiring)",
        agent_name="ElectricalAgent",
        items=items,
    )


def appliance_recommend_llm(items: list[dict]) -> list[Recommendation]:
    return _call_llm(
        domain="Home appliances (dishwasher, dryer, refrigerator, washer, oven)",
        agent_name="ApplianceAgent",
        items=items,
    )


def general_recommend_llm(items: list[dict]) -> list[Recommendation]:
    return _call_llm(
        domain="General home maintenance (exterior, gutters, paint, caulk, safety devices, landscaping)",
        agent_name="GeneralAgent",
        items=items,
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

CATEGORY_ROUTER_LLM = {
    "hvac":       hvac_recommend_llm,
    "plumbing":   plumbing_recommend_llm,
    "electrical": electrical_recommend_llm,
    "appliance":  appliance_recommend_llm,
    "general":    general_recommend_llm,
}


def route_to_subagent_llm(category: str, items: list[dict]) -> list[Recommendation]:
    """Route a batch of items to the correct domain LLM function."""
    fn = CATEGORY_ROUTER_LLM.get(category, general_recommend_llm)
    return fn(items)