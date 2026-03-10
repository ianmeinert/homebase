"""
conftest.py -- Pytest configuration for homebase test suite.

Patches:
  - LLM model constructor (no API key needed)
  - db.get_conn (in-memory SQLite, seeded fresh per test)
"""

import json
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

SEED_PATH = Path(__file__).parent.parent / "data" / "registry.json"


class MockLLMResponse:
    """Minimal response object matching LangChain message interface."""
    def __init__(self, content: str):
        self.content = content


class MockModel:
    """Stub model that returns deterministic JSON without hitting any API."""

    def invoke(self, messages):
        return MockLLMResponse(
            '[{"item_id": "TEST-001", "action": "Inspect and address.", '
            '"estimated_effort": "1 hr", "estimated_cost": "$50-100", '
            '"priority_note": "Address promptly.", "confidence": 0.85, "agent": "TestAgent"}]'
        )


def _make_in_memory_conn():
    """Create a fresh in-memory SQLite DB with schema + seed data."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    conn.executescript("""
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
            category_filter   TEXT,
            item_count        INTEGER NOT NULL DEFAULT 0,
            quadrant_summary  TEXT NOT NULL DEFAULT '{}',
            stale_count       INTEGER NOT NULL DEFAULT 0,
            hitl_approved     INTEGER NOT NULL DEFAULT 0,
            hitl_notes        TEXT NOT NULL DEFAULT '',
            deferred_items    TEXT NOT NULL DEFAULT '[]',
            summary_report    TEXT NOT NULL DEFAULT ''
        );
    """)

    if SEED_PATH.exists():
        items = json.loads(SEED_PATH.read_text())
        conn.executemany(
            """INSERT OR IGNORE INTO registry
               (id, category, title, description, urgency, impact, days_since_update, status)
               VALUES (:id, :category, :title, :description, :urgency, :impact, :days_since_update, :status)""",
            items,
        )
    conn.commit()
    return conn


@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    """
    Auto-used fixture -- patches:
      1. get_model() in llm_tools  (no Groq API key needed)
      2. _llm_synthesize in orchestrator
      3. db.get_conn  (in-memory SQLite, isolated per test)
    """
    mock_model = MockModel()
    monkeypatch.setattr("tools.llm_tools.get_model", lambda **kw: mock_model)

    def mock_synthesize(active_results, trigger, hitl_notes, **kwargs):
        ids = [r["item"]["id"] for r in active_results]
        return f"Test synthesis for '{trigger}'. Items: {', '.join(ids)}" if ids else "No active items."

    monkeypatch.setattr("agents.orchestrator._llm_synthesize", mock_synthesize)

    # Patch get_conn everywhere it's imported
    def make_conn():
        return _make_in_memory_conn()

    monkeypatch.setattr("tools.db.get_conn", make_conn)
    monkeypatch.setattr("tools.registry_tools.get_conn", make_conn)
    monkeypatch.setattr("tools.history_tools.get_conn", make_conn)