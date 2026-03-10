"""
orchestrator.py  -  Orchestrator agent node for HOMEBASE.
LLM-backed synthesis via Gemini.
"""

import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from graph.state import HombaseState
from tools.registry_tools import get_registry, classify_registry


SYNTHESIS_SYSTEM_PROMPT = (
    "You are a home management advisor producing a final action report. "
    "You will receive a list of approved maintenance recommendations from specialist agents. "
    "Write a concise, prioritized action plan in plain language.\n\n"
    "Structure your response as:\n"
    "1. A 2-3 sentence executive summary of the home's current state\n"
    "2. A prioritized action list  -  each item on its own line starting with a priority number, "
    "item ID in brackets, and the action\n"
    "3. A brief closing note on any patterns or systemic issues you observe\n\n"
    "Be direct and practical. Homeowner tone, not corporate. No markdown headers."
)


def _llm_synthesize(active_results: list, trigger: str, hitl_notes: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        max_output_tokens=1024,
        temperature=0,
    )
    payload = {
        "trigger": trigger,
        "human_notes": hitl_notes or "None",
        "approved_items": [
            {
                "id": r["item"]["id"],
                "title": r["item"]["title"],
                "quadrant": r["item"]["quadrant"],
                "urgency": r["item"]["urgency"],
                "impact": r["item"]["impact"],
                "days_since_update": r["item"]["days_since_update"],
                "action": r["recommendation"]["action"],
                "effort": r["recommendation"]["estimated_effort"],
                "cost": r["recommendation"]["estimated_cost"],
                "agent": r["recommendation"]["agent"],
            }
            for r in active_results
        ],
    }
    try:
        # Gemini does not support system role -- fold into user message
        combined = f"{SYNTHESIS_SYSTEM_PROMPT}\n\n{json.dumps(payload, indent=2)}"
        response = model.invoke([
            {"role": "user", "content": combined},
        ])
        return response.content.strip()
    except Exception as e:
        return f"Synthesis unavailable ({type(e).__name__}): {str(e)[:300]}"




QUADRANT_LABELS = {
    "HU/HI": "Immediate Action Required",
    "HU/LI": "Schedule Soon",
    "LU/HI": "Contingency - Plan Ahead",
    "LU/LI": "Defer / Accept",
}

QUADRANT_DESCRIPTIONS = {
    "HU/HI": "High urgency AND high impact. Address immediately.",
    "HU/LI": "High urgency, lower impact. Schedule within the week.",
    "LU/HI": "Low urgency but high consequence if ignored. Build a contingency plan.",
    "LU/LI": "Low priority on both dimensions. Defer until bandwidth allows.",
}


def format_item(item: dict) -> str:
    stale_flag = " [STALE]" if item["days_since_update"] >= 14 else ""
    return (
        f"  [{item['id']}] {item['title']}{stale_flag}\n"
        f"    Urgency: {item['urgency']} | Impact: {item['impact']} | "
        f"Days Since Update: {item['days_since_update']}\n"
        f"    {item['description']}"
    )


def build_report(buckets: dict, trigger: str) -> str:
    total = len(buckets["all"])
    stale_count = len(buckets["stale_items"])

    lines = [
        "=" * 60,
        "  HOMEBASE  -  Home Task Registry Analysis",
        f"  Trigger: {trigger}",
        "=" * 60,
        f"\nREGISTRY SUMMARY: {total} open items | {stale_count} stale (14+ days)\n",
    ]

    for bucket_key in ["hu_hi", "hu_li", "lu_hi", "lu_li"]:
        quadrant = bucket_key.upper().replace("_", "/")
        label = QUADRANT_LABELS[quadrant]
        desc = QUADRANT_DESCRIPTIONS[quadrant]
        items = buckets[bucket_key]

        lines.append(f"\n{label} ({quadrant})  -  {len(items)} items")
        lines.append(f"  {desc}")

        if items:
            for item in sorted(items, key=lambda x: -(x["urgency"] + x["impact"])):
                lines.append(format_item(item))
        else:
            lines.append("  No items in this quadrant.")

    if buckets["stale_items"]:
        lines.append("\n\nSTALE ITEMS (14+ days without update):")
        seen = set()
        for item in buckets["stale_items"]:
            if item["id"] not in seen:
                lines.append(
                    f"  [{item['id']}] {item['title']}  -  "
                    f"{item['days_since_update']} days | Quadrant: {item['quadrant']}"
                )
                seen.add(item["id"])

    lines.append("\n" + "=" * 60)
    lines.append("  [Phase 3: HU/HI items will trigger HITL checkpoints]")
    lines.append("=" * 60)

    return "\n".join(lines)


def build_synthesis_report(subagent_results: list[dict], trigger: str) -> str:
    """Build the final synthesized report after subagents have returned recommendations."""
    if not subagent_results:
        return "No subagent recommendations to synthesize."

    # Group by quadrant then category
    hu_hi_results = [r for r in subagent_results if r["item"]["quadrant"] == "HU/HI"]
    lu_hi_results = [r for r in subagent_results if r["item"]["quadrant"] == "LU/HI"]

    lines = [
        "=" * 60,
        "  HOMEBASE  -  Specialist Subagent Recommendations",
        f"  Trigger: {trigger}",
        "=" * 60,
    ]

    if hu_hi_results:
        lines.append(f"\nIMMEDIATE ACTION - HU/HI ({len(hu_hi_results)} items)\n")
        for r in sorted(hu_hi_results, key=lambda x: -(x["item"]["urgency"] + x["item"]["impact"])):
            item = r["item"]
            rec = r["recommendation"]
            stale = " [STALE]" if item["days_since_update"] >= 14 else ""
            lines.append(f"  [{item['id']}] {item['title']}{stale}")
            lines.append(f"    Agent     : {rec['agent']}")
            lines.append(f"    Action    : {rec['action']}")
            lines.append(f"    Effort    : {rec['estimated_effort']}")
            lines.append(f"    Cost est. : {rec['estimated_cost']}")
            lines.append(f"    Note      : {rec['priority_note']}")
            lines.append("")

    if lu_hi_results:
        lines.append(f"\nCONTINGENCY PLANNING - LU/HI ({len(lu_hi_results)} items)\n")
        for r in sorted(lu_hi_results, key=lambda x: -x["item"]["impact"]):
            item = r["item"]
            rec = r["recommendation"]
            stale = " [STALE]" if item["days_since_update"] >= 14 else ""
            lines.append(f"  [{item['id']}] {item['title']}{stale}")
            lines.append(f"    Agent     : {rec['agent']}")
            lines.append(f"    Action    : {rec['action']}")
            lines.append(f"    Effort    : {rec['estimated_effort']}")
            lines.append(f"    Cost est. : {rec['estimated_cost']}")
            lines.append(f"    Note      : {rec['priority_note']}")
            lines.append("")

    lines.append("=" * 60)
    lines.append("  [Phase 3: HU/HI items will trigger HITL checkpoints before action]")
    lines.append("=" * 60)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node: orchestrator  -  Phase 1 + Phase 2 delegation setup
# ---------------------------------------------------------------------------

def orchestrator_node(state: HombaseState) -> dict:
    """
    LangGraph node  -  Orchestrator Agent.
    Loads registry, classifies items, populates state buckets,
    and prepares delegated_items for subagent routing.
    """
    messages = [f"[Orchestrator] Trigger received: '{state['trigger']}'"]

    messages.append("[Orchestrator] Calling tool: get_registry()")
    raw_registry = get_registry()
    messages.append(f"[Orchestrator] Registry loaded: {len(raw_registry)} items")

    messages.append("[Orchestrator] Calling tool: classify_registry()")
    buckets = classify_registry(raw_registry)
    messages.append(
        f"[Orchestrator] Classification complete  -  "
        f"HU/HI: {len(buckets['hu_hi'])} | "
        f"HU/LI: {len(buckets['hu_li'])} | "
        f"LU/HI: {len(buckets['lu_hi'])} | "
        f"LU/LI: {len(buckets['lu_li'])} | "
        f"Stale: {len(buckets['stale_items'])}"
    )

    # Phase 2: delegate HU/HI and LU/HI items to specialist subagents
    delegated = buckets["hu_hi"] + buckets["lu_hi"]
    messages.append(
        f"[Orchestrator] Delegating {len(delegated)} items to specialist subagents "
        f"(HU/HI: {len(buckets['hu_hi'])}, LU/HI: {len(buckets['lu_hi'])})"
    )

    # Initial classification report (pre-subagent)
    report = build_report(buckets, state["trigger"])

    return {
        "raw_registry": raw_registry,
        "classified_items": buckets["all"],
        "hu_hi": buckets["hu_hi"],
        "hu_li": buckets["hu_li"],
        "lu_hi": buckets["lu_hi"],
        "lu_li": buckets["lu_li"],
        "stale_items": buckets["stale_items"],
        "delegated_items": delegated,
        "subagent_results": [],
        "summary_report": report,
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# Node: synthesizer  -  collects subagent results and builds final report
# ---------------------------------------------------------------------------

def synthesizer_node(state: HombaseState) -> dict:
    """
    LangGraph node  -  Synthesizer (Phase 3).
    Incorporates HITL approval decisions before assembling final report.
    Deferred items are excluded from the action plan.
    """
    results = state.get("subagent_results", [])
    deferred = set(state.get("deferred_items", []))
    hitl_approved = state.get("hitl_approved", False)
    hitl_notes = state.get("hitl_notes", "")

    messages = [
        f"[Synthesizer] Resuming after HITL checkpoint",
        f"[Synthesizer] Approval: {hitl_approved} | Deferred items: {list(deferred) or 'none'}",
        f"[Synthesizer] Processing {len(results)} subagent recommendations",
    ]

    # Filter out deferred items
    active_results = [r for r in results if r["item"]["id"] not in deferred]
    skipped_results = [r for r in results if r["item"]["id"] in deferred]

    if skipped_results:
        messages.append(
            f"[Synthesizer] Excluding {len(skipped_results)} deferred item(s): "
            f"{[r['item']['id'] for r in skipped_results]}"
        )

    messages.append("[Synthesizer] Calling Gemini for synthesis narrative...")
    import time; time.sleep(5)  # brief pause to avoid rate limit after subagent batch
    narrative = _llm_synthesize(active_results, state["trigger"], hitl_notes)

    # Build structured report with LLM narrative + HITL decision block
    hitl_lines = [
        "",
        "-" * 60,
        "  HITL DECISION SUMMARY",
        "-" * 60,
        f"  Approved  : {'Yes' if hitl_approved else 'No'}",
    ]
    if deferred:
        hitl_lines.append(f"  Deferred  : {', '.join(sorted(deferred))}")
    if hitl_notes:
        hitl_lines.append(f"  Notes     : {hitl_notes}")
    if not deferred and not hitl_notes:
        hitl_lines.append("  All HU/HI items approved for action.")
    hitl_lines.append("-" * 60)

    report = narrative + "\n" + "\n".join(hitl_lines)

    messages.append("[Synthesizer] Final report assembled.")

    return {
        "summary_report": report,
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# Node: hitl_briefing  -  presents HU/HI findings before interrupt fires
# ---------------------------------------------------------------------------

def hitl_briefing_node(state: HombaseState) -> dict:
    """
    LangGraph node  -  HITL Briefing.
    Formats the HU/HI action plan for human review.
    Execution will pause AFTER this node via interrupt_before on synthesizer.
    The human reads this output, then resumes with approval/deferral input.
    """
    messages = ["[HITL] Preparing action briefing for human review"]

    hu_hi_results = [
        r for r in state.get("subagent_results", [])
        if r["item"]["quadrant"] == "HU/HI"
    ]

    lines = [
        "\n" + "=" * 60,
        "  HOMEBASE - ACTION REQUIRED",
        "  Human-in-the-Loop Checkpoint",
        "=" * 60,
        f"\n  {len(hu_hi_results)} item(s) classified HU/HI require your review.",
        "  The agent will pause here until you approve or defer each item.\n",
    ]

    for i, r in enumerate(sorted(hu_hi_results, key=lambda x: -(x["item"]["urgency"] + x["item"]["impact"])), 1):
        item = r["item"]
        rec = r["recommendation"]
        stale = " [STALE]" if item["days_since_update"] >= 14 else ""
        lines.append(f"  [{i}] [{item['id']}] {item['title']}{stale}")
        lines.append(f"      Urgency: {item['urgency']} | Impact: {item['impact']}")
        lines.append(f"      Action : {rec['action']}")
        lines.append(f"      Effort : {rec['estimated_effort']}  |  Cost: {rec['estimated_cost']}")
        lines.append(f"      Note   : {rec['priority_note']}")
        lines.append("")

    lines.append("  Enter your decision when prompted.")
    lines.append("=" * 60)

    briefing = "\n".join(lines)
    messages.append("[HITL] Briefing prepared  -  awaiting human input")

    return {
        "summary_report": briefing,
        "messages": messages,
    }