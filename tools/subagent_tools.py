"""
subagent_tools.py  -  Mocked tool implementations for specialist subagents.
Each domain tool returns a structured recommendation for a given task item.
Phase 2: All logic is rule-based / mocked.
Phase 3+: These would call real APIs, scheduling systems, vendor lookups, etc.
"""

from typing import TypedDict


class Recommendation(TypedDict):
    item_id: str
    action: str
    estimated_effort: str   # e.g. "15 min DIY", "2-4 hr contractor"
    estimated_cost: str     # e.g. "$10-20", "$150-300"
    priority_note: str
    agent: str


# ---------------------------------------------------------------------------
# HVAC Agent Tools
# ---------------------------------------------------------------------------

def hvac_recommend(item: dict) -> Recommendation:
    """Generate a recommendation for an HVAC-category item."""
    title = item["title"].lower()
    urgency = item["urgency"]
    impact = item["impact"]

    if "filter" in title:
        return Recommendation(
            item_id=item["id"],
            action="Replace air filter  -  check unit size (1\" vs 4\" media filter) before purchasing.",
            estimated_effort="15 min DIY",
            estimated_cost="$15-40",
            priority_note="Running with a clogged filter increases energy cost and risks compressor damage.",
            agent="HVACAgent",
        )
    elif "tune" in title or "ac" in title or "furnace" in title:
        return Recommendation(
            item_id=item["id"],
            action="Schedule annual HVAC tune-up with licensed HVAC technician before peak season.",
            estimated_effort="1-2 hr (contractor)",
            estimated_cost="$80-150",
            priority_note="Pre-season scheduling avoids peak demand wait times and catches issues early.",
            agent="HVACAgent",
        )
    else:
        return Recommendation(
            item_id=item["id"],
            action=f"HVAC inspection recommended. Urgency: {urgency}, Impact: {impact}.",
            estimated_effort="1-3 hr (contractor)",
            estimated_cost="$100-200",
            priority_note="Review with licensed HVAC technician.",
            agent="HVACAgent",
        )


# ---------------------------------------------------------------------------
# Plumbing Agent Tools
# ---------------------------------------------------------------------------

def plumbing_recommend(item: dict) -> Recommendation:
    """Generate a recommendation for a plumbing-category item."""
    title = item["title"].lower()

    if "drain" in title:
        return Recommendation(
            item_id=item["id"],
            action="Try Zip-It drain snake first. If unresolved in 30 min, escalate to plumber.",
            estimated_effort="30 min DIY or 1 hr contractor",
            estimated_cost="$5-10 DIY / $100-150 contractor",
            priority_note="Slow drains often indicate partial blockage  -  ignoring leads to full blockage or overflow.",
            agent="PlumbingAgent",
        )
    elif "water heater" in title or "anode" in title:
        return Recommendation(
            item_id=item["id"],
            action="Schedule plumber to inspect and replace anode rod. Units over 6 years need annual inspection.",
            estimated_effort="1-2 hr (contractor)",
            estimated_cost="$150-250",
            priority_note="Failed anode rod accelerates tank corrosion  -  replacement at 6 years is standard.",
            agent="PlumbingAgent",
        )
    elif "hose" in title or "spigot" in title or "bib" in title:
        return Recommendation(
            item_id=item["id"],
            action="Replace hose bib washer or full bib assembly. DIY-feasible with shutoff valve access.",
            estimated_effort="30-45 min DIY",
            estimated_cost="$10-30 DIY / $80-120 contractor",
            priority_note="Exterior drips risk foundation moisture intrusion in extended wet periods.",
            agent="PlumbingAgent",
        )
    else:
        return Recommendation(
            item_id=item["id"],
            action="Plumbing inspection recommended.",
            estimated_effort="1-2 hr (contractor)",
            estimated_cost="$100-200",
            priority_note="Have a licensed plumber assess.",
            agent="PlumbingAgent",
        )


# ---------------------------------------------------------------------------
# Electrical Agent Tools
# ---------------------------------------------------------------------------

def electrical_recommend(item: dict) -> Recommendation:
    """Generate a recommendation for an electrical-category item."""
    title = item["title"].lower()

    if "gfci" in title or "outlet" in title:
        return Recommendation(
            item_id=item["id"],
            action="Replace GFCI outlet. Test with non-contact voltage tester before work. DIY-feasible.",
            estimated_effort="30-45 min DIY",
            estimated_cost="$15-30 DIY / $80-120 electrician",
            priority_note="Tripping GFCI in garage may indicate moisture ingress or wiring fault  -  inspect carefully.",
            agent="ElectricalAgent",
        )
    elif "light" in title or "motion" in title:
        return Recommendation(
            item_id=item["id"],
            action="Install motion sensor light at back porch. Requires outdoor-rated fixture and existing junction box.",
            estimated_effort="1-2 hr DIY",
            estimated_cost="$30-80 DIY / $120-180 electrician",
            priority_note="Verify junction box exists before purchasing fixture.",
            agent="ElectricalAgent",
        )
    else:
        return Recommendation(
            item_id=item["id"],
            action="Electrical inspection recommended. Do not defer if flickering or tripping is present.",
            estimated_effort="1-2 hr (electrician)",
            estimated_cost="$100-200",
            priority_note="Always use a licensed electrician for panel or wiring work.",
            agent="ElectricalAgent",
        )


# ---------------------------------------------------------------------------
# Appliance Agent Tools
# ---------------------------------------------------------------------------

def appliance_recommend(item: dict) -> Recommendation:
    """Generate a recommendation for an appliance-category item."""
    title = item["title"].lower()

    if "dishwasher" in title or "drain" in title:
        return Recommendation(
            item_id=item["id"],
            action="Check drain hose for kinks, clean filter basket, inspect drain pump. DIY-feasible.",
            estimated_effort="45 min - 1.5 hr DIY",
            estimated_cost="$0-30 DIY / $150-250 appliance tech",
            priority_note="Standing water breeds mold  -  don't defer beyond this week.",
            agent="ApplianceAgent",
        )
    elif "dryer" in title or "vent" in title:
        return Recommendation(
            item_id=item["id"],
            action="Clean dryer vent duct from exterior cap to dryer connection. Use vent brush kit.",
            estimated_effort="1-2 hr DIY",
            estimated_cost="$20-40 DIY / $100-150 contractor",
            priority_note="Lint buildup is a leading cause of residential fires  -  18 months is significantly overdue.",
            agent="ApplianceAgent",
        )
    elif "ice" in title or "refrigerator" in title:
        return Recommendation(
            item_id=item["id"],
            action="Check water line shutoff valve and ice maker on/off arm. Reset ice maker module if needed.",
            estimated_effort="30 min DIY",
            estimated_cost="$0 DIY / $100-150 appliance tech if part replacement needed",
            priority_note="Low consequence  -  defer if DIY steps don't resolve within 30 min.",
            agent="ApplianceAgent",
        )
    else:
        return Recommendation(
            item_id=item["id"],
            action="Appliance inspection recommended.",
            estimated_effort="1-2 hr (appliance tech)",
            estimated_cost="$100-200",
            priority_note="Contact manufacturer or appliance repair service.",
            agent="ApplianceAgent",
        )


# ---------------------------------------------------------------------------
# General Maintenance Agent Tools
# ---------------------------------------------------------------------------

def general_recommend(item: dict) -> Recommendation:
    """Generate a recommendation for a general maintenance item."""
    title = item["title"].lower()

    if "gutter" in title:
        return Recommendation(
            item_id=item["id"],
            action="Clear gutters and flush downspouts with garden hose. Check for sagging sections.",
            estimated_effort="1-2 hr DIY",
            estimated_cost="$0 DIY / $100-200 contractor",
            priority_note="Clogged gutters overflow against fascia and foundation  -  priority before next rain.",
            agent="GeneralAgent",
        )
    elif "fence" in title or "gate" in title or "latch" in title:
        return Recommendation(
            item_id=item["id"],
            action="Replace latch hardware. Most gate latches are standard and available at hardware stores.",
            estimated_effort="20-30 min DIY",
            estimated_cost="$10-25",
            priority_note="Open gate is a security and liability issue, especially with children or pets.",
            agent="GeneralAgent",
        )
    elif "caulk" in title or "shower" in title or "grout" in title:
        return Recommendation(
            item_id=item["id"],
            action="Remove old caulk with oscillating tool or razor, clean, apply 100% silicone caulk.",
            estimated_effort="1-2 hr DIY",
            estimated_cost="$15-30",
            priority_note="Cracked caulk allows moisture behind tile  -  leads to mold and substrate damage.",
            agent="GeneralAgent",
        )
    elif "smoke" in title or "detector" in title or "co" in title:
        return Recommendation(
            item_id=item["id"],
            action="Test all smoke and CO detectors. Replace batteries. Replace units older than 10 years.",
            estimated_effort="15-20 min DIY",
            estimated_cost="$0-50 (batteries / unit replacement)",
            priority_note="Life safety item  -  do this week regardless of other priorities.",
            agent="GeneralAgent",
        )
    elif "paint" in title or "trim" in title:
        return Recommendation(
            item_id=item["id"],
            action="Sand, prime, and repaint affected trim sections. Use exterior-grade paint.",
            estimated_effort="2-4 hr DIY",
            estimated_cost="$20-50",
            priority_note="Low urgency but exposed wood will deteriorate if left unprotected.",
            agent="GeneralAgent",
        )
    else:
        return Recommendation(
            item_id=item["id"],
            action="General maintenance inspection recommended.",
            estimated_effort="1-2 hr DIY",
            estimated_cost="$0-50",
            priority_note="Assess and schedule when bandwidth allows.",
            agent="GeneralAgent",
        )


# ---------------------------------------------------------------------------
# Router  -  maps category to the correct recommend function
# ---------------------------------------------------------------------------

CATEGORY_ROUTER = {
    "hvac":      hvac_recommend,
    "plumbing":  plumbing_recommend,
    "electrical": electrical_recommend,
    "appliance": appliance_recommend,
    "general":   general_recommend,
}


def route_to_subagent(item: dict) -> Recommendation:
    """Route a single item to the appropriate domain recommendation tool."""
    category = item.get("category", "general")
    fn = CATEGORY_ROUTER.get(category, general_recommend)
    return fn(item)