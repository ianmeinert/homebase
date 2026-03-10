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


# -- Registry CRUD ----------------------------------------------------------------

CATEGORY_PREFIXES = {
    "hvac":       "HVA",
    "plumbing":   "PLM",
    "electrical": "ELC",
    "appliance":  "APP",
    "general":    "GEN",
}


def _next_id(category: str, registry: list[dict]) -> str:
    """Generate next sequential ID for a given category."""
    prefix = CATEGORY_PREFIXES.get(category, "GEN")
    existing = [
        int(i["id"].split("-")[1])
        for i in registry
        if i["id"].startswith(prefix) and i["id"].split("-")[1].isdigit()
    ]
    next_num = max(existing, default=0) + 1
    return f"{prefix}-{next_num:03d}"


def save_registry(items: list[dict]) -> None:
    """Persist registry to disk."""
    with open(REGISTRY_PATH, "w") as f:
        json.dump(items, f, indent=4)


def add_item(
    category: str,
    title: str,
    description: str,
    urgency: float,
    impact: float,
) -> dict:
    """
    Add a new item to the registry.
    Returns the new item with auto-generated ID.
    """
    registry = get_registry()
    new_item = {
        "id": _next_id(category, registry),
        "category": category,
        "title": title,
        "description": description,
        "urgency": round(urgency, 2),
        "impact": round(impact, 2),
        "days_since_update": 0,
        "status": "open",
    }
    registry.append(new_item)
    save_registry(registry)
    return new_item


def update_item(item_id: str, updates: dict) -> dict | None:
    """
    Update fields on an existing registry item.
    Returns updated item or None if not found.
    """
    registry = get_registry()
    for i, item in enumerate(registry):
        if item["id"] == item_id:
            # Only allow safe fields to be updated
            allowed = {"title", "description", "urgency", "impact", "status", "days_since_update"}
            for k, v in updates.items():
                if k in allowed:
                    registry[i][k] = v
            save_registry(registry)
            return registry[i]
    return None


def close_item(item_id: str) -> bool:
    """
    Mark an item as closed and remove it from the active registry.
    Returns True if found and closed, False otherwise.
    """
    registry = get_registry()
    updated = [i for i in registry if i["id"] != item_id]
    if len(updated) == len(registry):
        return False
    save_registry(updated)
    return True