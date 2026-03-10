"""
tools/history_tools.py  -  Run history persistence.

Each completed run is appended to data/run_history.json.
Records include trigger, classification summary, HITL decisions,
and the full synthesized report for audit trail purposes.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

HISTORY_PATH = Path(__file__).parent.parent / "data" / "run_history.json"


def _load_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    with open(HISTORY_PATH) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_history(records: list[dict]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(records, f, indent=4)


def save_run(
    trigger: str,
    classified_items: list[dict],
    deferred_items: list[str],
    hitl_approved: bool,
    hitl_notes: str,
    summary_report: str,
    category_filter: list[str] | None = None,
) -> str:
    """
    Persist a completed run to history.
    Returns the run_id of the saved record.
    """
    from collections import Counter

    q_counts = Counter(i.get("quadrant", "unknown") for i in classified_items)
    stale_count = sum(1 for i in classified_items if i.get("days_since_update", 0) >= 14)

    record = {
        "run_id":          str(uuid.uuid4()),
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
        "trigger":         trigger,
        "category_filter": category_filter,
        "item_count":      len(classified_items),
        "quadrant_summary": {
            "HU/HI": q_counts.get("HU/HI", 0),
            "HU/LI": q_counts.get("HU/LI", 0),
            "LU/HI": q_counts.get("LU/HI", 0),
            "LU/LI": q_counts.get("LU/LI", 0),
        },
        "stale_count":    stale_count,
        "hitl_approved":  hitl_approved,
        "hitl_notes":     hitl_notes,
        "deferred_items": deferred_items,
        "summary_report": summary_report,
    }

    history = _load_history()
    history.append(record)
    _save_history(history)
    return record["run_id"]


def get_history(limit: int | None = None) -> list[dict]:
    """
    Return run history newest-first.
    Optionally limit to the most recent N records.
    """
    history = _load_history()
    history = sorted(history, key=lambda r: r["timestamp"], reverse=True)
    if limit:
        return history[:limit]
    return history


def delete_run(run_id: str) -> bool:
    """Delete a single run record by run_id. Returns True if found."""
    history = _load_history()
    updated = [r for r in history if r["run_id"] != run_id]
    if len(updated) == len(history):
        return False
    _save_history(updated)
    return True


def clear_history() -> int:
    """Delete all run history. Returns count of deleted records."""
    history = _load_history()
    count = len(history)
    _save_history([])
    return count