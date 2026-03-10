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
    days_since_update INTEGER NOT NULL DEFAULT 0,
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

    if first_run and SEED_PATH.exists():
        _seed_registry(conn)

    return conn


def _seed_registry(conn: sqlite3.Connection) -> None:
    """Populate registry table from registry.json seed file."""
    with open(SEED_PATH) as f:
        items = json.load(f)

    conn.executemany(
        """
        INSERT OR IGNORE INTO registry
            (id, category, title, description, urgency, impact, days_since_update, status)
        VALUES
            (:id, :category, :title, :description, :urgency, :impact, :days_since_update, :status)
        """,
        items,
    )
    conn.commit()


def row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a sqlite3.Row to a plain dict."""
    return dict(row)