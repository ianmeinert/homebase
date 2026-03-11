"""
scripts/seed_run_history.py  —  Insert 100 synthetic run history records into homebase.db.

Generates realistic data with:
  - Varied triggers and category filters
  - Quadrant distributions that shift over time (degradation arc then recovery)
  - Stale counts that trend upward mid-period then resolve
  - HITL approvals and deferrals with notes
  - Timestamped across the past 90 days

Usage:
    uv run python scripts/seed_run_history.py
    uv run python scripts/seed_run_history.py --clear   # wipe existing history first
"""

import argparse
import json
import random
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.db import get_conn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRIGGERS = [
    "weekly home review",
    "morning briefing",
    "what needs immediate attention",
    "full home assessment",
    "fire and safety inspection",
    "plumbing systems audit",
    "electrical systems inspection",
    "hvac seasonal maintenance check",
    "appliance status review",
    "exterior and grounds walkthrough",
]

CATEGORY_FILTERS = [
    None, None, None,           # most runs are full-registry
    '["hvac"]',
    '["plumbing"]',
    '["electrical"]',
    '["appliance"]',
    '["hvac", "electrical"]',
    '["plumbing", "appliance"]',
]

ITEM_IDS = [
    "HV-001", "HV-002", "HV-003",
    "PLB-001", "PLB-002", "PLB-003",
    "EL-001", "EL-002",
    "APP-001", "APP-002", "APP-003",
    "GEN-001", "GEN-002", "GEN-003", "GEN-004",
]

HITL_NOTES_POOL = [
    "Approved all recommendations. Scheduling HVAC contractor next week.",
    "Deferred plumbing items — waiting on insurance assessment.",
    "Approved electrical inspection. Deferred low-priority items.",
    "All HU/HI items approved. Owner will handle GEN items personally.",
    "Deferred APP-002 — appliance under warranty, awaiting service call.",
    "Full approval. Priority given to safety-related electrical items.",
    "Partial deferral — budget constraints this month.",
    "Approved. HVAC filter replacement completed same day.",
    "Deferred outdoor items due to weather. Indoor items approved.",
    "All approved. Contractor booked for EL-001.",
    "",
    "",  # some runs have no notes
]

REPORT_TEMPLATES = [
    "Analysis identified {huhi} critical items requiring immediate attention across {cats}. "
    "HVAC and electrical categories show elevated urgency patterns. "
    "Recommend prioritizing safety-related items before end of week.",

    "Weekly review complete. {total} items assessed, {stale} flagged as stale. "
    "Plumbing and appliance items trending toward higher urgency. "
    "HITL approved {huhi} high-priority recommendations.",

    "Systemic review identified deferred maintenance pattern in {cats}. "
    "{huhi} items escalated to immediate action. "
    "Confidence in recommendations: high based on trend data.",

    "Morning briefing: {total} active items, {stale} stale. "
    "No new critical escalations since last run. "
    "Recommend continued monitoring of HV-001 and EL-002.",

    "Full assessment complete. Quadrant distribution shows {huhi} HU/HI, "
    "stable LU/LI baseline. Category {cats} flagged for follow-up inspection.",
]

# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------

def _quadrant_summary(phase: float) -> dict:
    """
    Generate realistic quadrant distribution based on phase (0.0 = start, 1.0 = end).
    Arc: items degrade mid-period (phase ~0.4-0.6) then partially recover.
    """
    # Degradation arc: HU/HI peaks around phase 0.5
    if phase < 0.3:
        huhi = random.randint(1, 3)
        luli = random.randint(4, 7)
    elif phase < 0.6:
        huhi = random.randint(3, 6)
        luli = random.randint(2, 5)
    else:
        huhi = random.randint(1, 4)
        luli = random.randint(3, 7)

    huli = random.randint(1, 3)
    luhi = random.randint(1, 3)

    return {"HU/HI": huhi, "HU/LI": huli, "LU/HI": luhi, "LU/LI": luli}


def _stale_count(phase: float) -> int:
    """Stale count trends up mid-period (neglect arc) then resolves."""
    if phase < 0.25:
        return random.randint(0, 2)
    elif phase < 0.55:
        return random.randint(2, 7)
    elif phase < 0.75:
        return random.randint(1, 5)
    else:
        return random.randint(0, 3)


def _deferred_items(qs: dict, hitl_approved: int) -> list:
    """Generate deferred item list — more deferrals when HU/HI is high."""
    if not hitl_approved:
        return []
    huhi_count = qs.get("HU/HI", 0)
    n_defer = random.randint(0, min(3, huhi_count))
    return random.sample(ITEM_IDS, min(n_defer, len(ITEM_IDS)))


def _make_report(qs: dict, stale: int, cat_filter) -> str:
    cats = json.loads(cat_filter) if cat_filter else ["hvac", "electrical", "plumbing"]
    cat_str = ", ".join(cats[:2])
    total = sum(qs.values())
    template = random.choice(REPORT_TEMPLATES)
    return template.format(
        huhi=qs.get("HU/HI", 0),
        total=total,
        stale=stale,
        cats=cat_str,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def seed(n: int = 100, clear: bool = False):
    conn = get_conn()

    if clear:
        conn.execute("DELETE FROM run_history")
        conn.commit()
        print("Cleared existing run history.")

    existing = conn.execute("SELECT COUNT(*) FROM run_history").fetchone()[0]
    print(f"Existing run history records: {existing}")

    now = datetime.now()
    start = now - timedelta(days=90)
    interval = timedelta(days=90) / n

    records = []
    for i in range(n):
        phase     = i / n
        ts        = start + interval * i + timedelta(hours=random.randint(7, 20), minutes=random.randint(0, 59))
        run_id    = f"seed-{uuid.uuid4().hex[:8]}"
        trigger   = random.choice(TRIGGERS)
        cat_filter = random.choice(CATEGORY_FILTERS)
        qs        = _quadrant_summary(phase)
        stale     = _stale_count(phase)
        item_count = sum(qs.values())
        hitl_approved = 1 if random.random() > 0.1 else 0  # 90% approved
        notes     = random.choice(HITL_NOTES_POOL)
        deferred  = _deferred_items(qs, hitl_approved)
        report    = _make_report(qs, stale, cat_filter)

        records.append((
            run_id,
            ts.strftime("%Y-%m-%dT%H:%M"),
            trigger,
            cat_filter,
            item_count,
            json.dumps(qs),
            stale,
            hitl_approved,
            notes,
            json.dumps(deferred),
            report,
        ))

    conn.executemany(
        """INSERT OR IGNORE INTO run_history
           (run_id, timestamp, trigger, category_filter, item_count,
            quadrant_summary, stale_count, hitl_approved, hitl_notes,
            deferred_items, summary_report)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        records,
    )
    conn.commit()

    final = conn.execute("SELECT COUNT(*) FROM run_history").fetchone()[0]
    print(f"Inserted {len(records)} synthetic records. Total now: {final}")
    print("Done. Run the app and trigger an RCA to see improved confidence.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed synthetic run history into homebase.db")
    parser.add_argument("--clear", action="store_true", help="Delete existing run history before seeding")
    parser.add_argument("--count", type=int, default=100, help="Number of records to insert (default: 100)")
    args = parser.parse_args()
    seed(n=args.count, clear=args.clear)