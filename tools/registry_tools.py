"""
tools.py  -  Mocked tool implementations for Phase 1.
These simulate tool calls an agent would make against real integrations in later phases.
"""

import json
import os
from pathlib import Path


REGISTRY_PATH = Path(__file__).parent.parent / "data" / "registry.json"
URGENCY_THRESHOLD = 0.6
IMPACT_THRESHOLD = 0.6
STALE_DAYS_THRESHOLD = 14


def get_registry() -> list[dict]:
    """
    Tool: Load the home task registry.
    Phase 1: reads from local JSON.
    Future: could pull from Home Assistant, Notion, Airtable, etc.
    """
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def classify_item(item: dict) -> dict:
    """
    Tool: Apply quadrant classification to a single registry item.
    Mirrors the RMA quadrant framework  -  Urgency x Impact.
    """
    urgency = item["urgency"]
    impact = item["impact"]

    if urgency >= URGENCY_THRESHOLD and impact >= IMPACT_THRESHOLD:
        quadrant = "HU/HI"
    elif urgency >= URGENCY_THRESHOLD and impact < IMPACT_THRESHOLD:
        quadrant = "HU/LI"
    elif urgency < URGENCY_THRESHOLD and impact >= IMPACT_THRESHOLD:
        quadrant = "LU/HI"
    else:
        quadrant = "LU/LI"

    return {**item, "quadrant": quadrant}


def classify_registry(items: list[dict]) -> dict:
    """
    Tool: Classify all registry items and bucket them by quadrant.
    Also flags stale items regardless of quadrant.
    Returns structured buckets.
    """
    classified = [classify_item(i) for i in items]

    buckets = {
        "hu_hi": [],
        "hu_li": [],
        "lu_hi": [],
        "lu_li": [],
        "stale_items": [],
        "all": classified
    }

    for item in classified:
        bucket = item["quadrant"].lower().replace("/", "_")
        buckets[bucket].append(item)
        if item["days_since_update"] >= STALE_DAYS_THRESHOLD:
            buckets["stale_items"].append(item)

    return buckets


def get_item_detail(item_id: str, registry: list[dict]) -> dict | None:
    """
    Tool: Retrieve full detail for a specific registry item by ID.
    """
    for item in registry:
        if item["id"] == item_id:
            return item
    return None