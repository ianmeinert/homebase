"""
subagents.py  -  Specialist subagent nodes for HOMEBASE.
Each subagent batches its assigned items into a single LLM call
and returns structured recommendations back to the synthesizer.
"""

from graph.state import HombaseState
from tools.llm_tools import route_to_subagent_llm


# ---------------------------------------------------------------------------
# Shared processing logic
# ---------------------------------------------------------------------------

def _process_items_llm(category: str, agent_name: str, state: HombaseState) -> dict:
    """
    Filter delegated items for this category, call the LLM in one batch,
    and return updated state fields.
    """
    items = [
        i for i in state.get("delegated_items", [])
        if i["category"] == category
    ]

    messages = [f"[{agent_name}] Activated  -  {len(items)} item(s) assigned"]

    if not items:
        messages.append(f"[{agent_name}] No items for this domain  -  exiting")
        return {"messages": messages}

    item_ids = [i["id"] for i in items]
    messages.append(f"[{agent_name}] Calling Gemini  -  batch: {item_ids}")

    recommendations = route_to_subagent_llm(category, items)

    results = []
    for item in items:
        rec = next((r for r in recommendations if r["item_id"] == item["id"]), None)
        if rec is None:
            rec = {
                "item_id": item["id"],
                "action": f"Review {item['title']}  -  no recommendation returned.",
                "estimated_effort": "TBD",
                "estimated_cost": "TBD",
                "priority_note": "Manual review required.",
                "agent": agent_name,
            }
        messages.append(f"[{agent_name}] Recommendation ready for [{item['id']}]")
        results.append({"item": item, "recommendation": rec})

    messages.append(f"[{agent_name}] Complete  -  {len(results)} recommendations returned")

    existing = state.get("subagent_results", [])
    return {
        "subagent_results": existing + results,
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# Subagent nodes
# ---------------------------------------------------------------------------

def hvac_agent_node(state: HombaseState) -> dict:
    return _process_items_llm("hvac", "HVACAgent", state)

def plumbing_agent_node(state: HombaseState) -> dict:
    return _process_items_llm("plumbing", "PlumbingAgent", state)

def electrical_agent_node(state: HombaseState) -> dict:
    return _process_items_llm("electrical", "ElectricalAgent", state)

def appliance_agent_node(state: HombaseState) -> dict:
    return _process_items_llm("appliance", "ApplianceAgent", state)

def general_agent_node(state: HombaseState) -> dict:
    return _process_items_llm("general", "GeneralAgent", state)