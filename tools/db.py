"""
tools/db.py  -  SQLite connection manager for HOMEBASE.

Single source of truth for:
  - DB path resolution
  - Schema creation / migration
  - Connection factory

Both registry_tools and history_tools import get_conn() from here.
The DB is created and seeded on first access if it does not exist.
"""

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "homebase.db"
SEED_PATH = Path(__file__).parent.parent / "data" / "registry.json"

DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS registry (
    id                TEXT PRIMARY KEY,
    category          TEXT NOT NULL,
    title             TEXT NOT NULL,
    description       TEXT NOT NULL DEFAULT '',
    urgency           REAL NOT NULL DEFAULT 0.5,
    impact            REAL NOT NULL DEFAULT 0.5,
    updated_at        TEXT NOT NULL DEFAULT (datetime('now')),
    status            TEXT NOT NULL DEFAULT 'open'
);

CREATE TABLE IF NOT EXISTS run_history (
    run_id            TEXT PRIMARY KEY,
    timestamp         TEXT NOT NULL,
    trigger           TEXT NOT NULL,
    category_filter   TEXT,          -- JSON array or NULL
    item_count        INTEGER NOT NULL DEFAULT 0,
    quadrant_summary  TEXT NOT NULL DEFAULT '{}',  -- JSON object
    stale_count       INTEGER NOT NULL DEFAULT 0,
    hitl_approved     INTEGER NOT NULL DEFAULT 0,  -- 0/1 boolean
    hitl_notes        TEXT NOT NULL DEFAULT '',
    deferred_items    TEXT NOT NULL DEFAULT '[]',  -- JSON array
    summary_report    TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_run_history_timestamp ON run_history(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_registry_category ON registry(category);
CREATE INDEX IF NOT EXISTS idx_registry_status ON registry(status);
"""


def get_conn() -> sqlite3.Connection:
    """
    Return a sqlite3 connection with row_factory set to Row.
    Creates and seeds the DB on first call if it does not exist.
    """
    first_run = not DB_PATH.exists()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(DDL)
    conn.commit()

    # Migration: add updated_at if upgrading from older schema without it
    cols = [r[1] for r in conn.execute("PRAGMA table_info(registry)").fetchall()]
    if "updated_at" not in cols:
        conn.execute("ALTER TABLE registry ADD COLUMN updated_at TEXT NOT NULL DEFAULT (datetime('now'))")
        # Back-fill from days_since_update if column exists
        if "days_since_update" in cols:
            conn.execute(
                "UPDATE registry SET updated_at = datetime('now', '-' || days_since_update || ' days')"
            )
        conn.commit()

    if first_run and SEED_PATH.exists():
        _seed_registry(conn)

    return conn


def _seed_registry(conn: sqlite3.Connection) -> None:
    """Populate registry table from registry.json seed file."""
    with open(SEED_PATH) as f:
        items = json.load(f)

    # Convert days_since_update from seed file to an actual updated_at timestamp
    for item in items:
        days = item.pop("days_since_update", 0)
        item["updated_at"] = f"datetime('now', '-{days} days')"

    # Use executescript-style to evaluate datetime() — insert one by one
    for item in items:
        conn.execute(
            f"""INSERT OR IGNORE INTO registry
                (id, category, title, description, urgency, impact, updated_at, status)
            VALUES
                (:id, :category, :title, :description, :urgency, :impact, {item.pop('updated_at')}, :status)""",
            item,
        )
    conn.commit()


def row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a sqlite3.Row to a plain dict."""
    return dict(row)