"""
registry_tools.py  -  Registry CRUD backed by SQLite.

Public interface is identical to the JSON-backed version.
All callers (agents, app.py, tests) require no changes.
"""

from tools.db import get_conn, row_to_dict

URGENCY_THRESHOLD = 0.6
IMPACT_THRESHOLD = 0.6
STALE_DAYS_THRESHOLD = 14

CATEGORY_PREFIXES = {
    "hvac":       "HVA",
    "plumbing":   "PLM",
    "electrical": "ELC",
    "appliance":  "APP",
    "general":    "GEN",
}


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get_registry() -> list[dict]:
    """Return all open registry items as a list of dicts."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM registry WHERE status = 'open' ORDER BY id"
    ).fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]


def get_item_detail(item_id: str, registry: list[dict]) -> dict | None:
    """Retrieve full detail for a specific registry item by ID."""
    for item in registry:
        if item["id"] == item_id:
            return item
    return None


# ---------------------------------------------------------------------------
# Classification (pure logic - no DB I/O)
# ---------------------------------------------------------------------------

def classify_item(item: dict) -> dict:
    """Apply quadrant classification to a single registry item."""
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
    """Classify all registry items and bucket them by quadrant."""
    classified = [classify_item(i) for i in items]

    buckets = {
        "hu_hi": [],
        "hu_li": [],
        "lu_hi": [],
        "lu_li": [],
        "stale_items": [],
        "all": classified,
    }

    for item in classified:
        bucket = item["quadrant"].lower().replace("/", "_")
        buckets[bucket].append(item)
        if item["days_since_update"] >= STALE_DAYS_THRESHOLD:
            buckets["stale_items"].append(item)

    return buckets


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def _next_id(category: str, conn) -> str:
    """Generate next sequential ID for a given category (all statuses)."""
    prefix = CATEGORY_PREFIXES.get(category, "GEN")
    rows = conn.execute(
        "SELECT id FROM registry WHERE id LIKE ?", (f"{prefix}-%",)
    ).fetchall()
    existing = []
    for row in rows:
        parts = row["id"].split("-")
        if len(parts) == 2 and parts[1].isdigit():
            existing.append(int(parts[1]))
    next_num = max(existing, default=0) + 1
    return f"{prefix}-{next_num:03d}"


def save_registry(items: list[dict]) -> None:
    """
    Bulk-replace the open registry with the provided list.
    Kept for backwards compatibility. Prefer targeted CRUD functions.
    """
    conn = get_conn()
    conn.execute("DELETE FROM registry WHERE status = 'open'")
    conn.executemany(
        """
        INSERT OR REPLACE INTO registry
            (id, category, title, description, urgency, impact, days_since_update, status)
        VALUES
            (:id, :category, :title, :description, :urgency, :impact, :days_since_update, :status)
        """,
        items,
    )
    conn.commit()
    conn.close()


def add_item(
    category: str,
    title: str,
    description: str,
    urgency: float,
    impact: float,
) -> dict:
    """Add a new item to the registry. Returns the new item dict."""
    conn = get_conn()
    new_id = _next_id(category, conn)
    new_item = {
        "id":                new_id,
        "category":          category,
        "title":             title,
        "description":       description,
        "urgency":           round(urgency, 2),
        "impact":            round(impact, 2),
        "days_since_update": 0,
        "status":            "open",
    }
    conn.execute(
        """
        INSERT INTO registry
            (id, category, title, description, urgency, impact, days_since_update, status)
        VALUES
            (:id, :category, :title, :description, :urgency, :impact, :days_since_update, :status)
        """,
        new_item,
    )
    conn.commit()
    conn.close()
    return new_item


def update_item(item_id: str, updates: dict) -> dict | None:
    """Update allowed fields on an existing item. Returns updated item or None."""
    allowed = {"title", "description", "urgency", "impact", "status", "days_since_update"}
    safe = {k: v for k, v in updates.items() if k in allowed}
    if not safe:
        return None

    conn = get_conn()
    set_clause = ", ".join(f"{k} = :{k}" for k in safe)
    safe["_id"] = item_id
    conn.execute(
        f"UPDATE registry SET {set_clause} WHERE id = :_id",
        safe,
    )
    conn.commit()
    row = conn.execute("SELECT * FROM registry WHERE id = ?", (item_id,)).fetchone()
    conn.close()
    return row_to_dict(row) if row else None


def close_item(item_id: str) -> bool:
    """Remove an item from the registry. Returns True if found."""
    conn = get_conn()
    cursor = conn.execute("DELETE FROM registry WHERE id = ?", (item_id,))
    conn.commit()
    conn.close()
    return cursor.rowcount > 0