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
            updated_at        TEXT NOT NULL DEFAULT (datetime('now')),
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
        for item in items:
            days = item.pop("days_since_update", 0)
            conn.execute(
                f"""INSERT OR IGNORE INTO registry
                    (id, category, title, description, urgency, impact, updated_at, status)
                VALUES
                    (:id, :category, :title, :description, :urgency, :impact, datetime('now', '-{days} days'), :status)""",
                item,
            )
    conn.commit()
    return conn


@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    """
    Auto-used fixture -- patches:
      1. get_model() in llm_tools  (no Groq API key needed)
      2. _llm_synthesize in orchestrator
      3. db.get_conn  (single shared in-memory SQLite per test)
    """
    from unittest.mock import MagicMock

    _base_conn = _make_in_memory_conn()

    class _NoCloseConn:
        """Proxy that delegates everything to the real conn but ignores close()."""
        def __getattr__(self, name):
            return getattr(_base_conn, name)
        def close(self):
            pass  # no-op — keep shared conn alive
        def execute(self, *a, **kw):
            return _base_conn.execute(*a, **kw)
        def executemany(self, *a, **kw):
            return _base_conn.executemany(*a, **kw)
        def executescript(self, *a, **kw):
            return _base_conn.executescript(*a, **kw)
        def commit(self):
            return _base_conn.commit()
        def rollback(self):
            return _base_conn.rollback()
        def cursor(self):
            return _base_conn.cursor()
        @property
        def row_factory(self):
            return _base_conn.row_factory
        @row_factory.setter
        def row_factory(self, v):
            _base_conn.row_factory = v

    def make_conn():
        return _NoCloseConn()

    mock_model = MockModel()
    mock_model.invoke = MagicMock(side_effect=mock_model.invoke)

    monkeypatch.setattr("tools.llm_tools.get_model", lambda **kw: mock_model)

    def mock_synthesize(active_results, trigger, hitl_notes, **kwargs):
        ids = [r["item"]["id"] for r in active_results]
        narrative = f"Test synthesis for '{trigger}'. Items: {', '.join(ids)}" if ids else "No active items."
        return narrative, "Llama 3.3 70B"

    monkeypatch.setattr("agents.orchestrator._llm_synthesize", mock_synthesize)

    monkeypatch.setattr("tools.db.get_conn", make_conn)
    monkeypatch.setattr("tools.registry_tools.get_conn", make_conn)
    monkeypatch.setattr("tools.history_tools.get_conn", make_conn)
    import tools.chart_agent
    monkeypatch.setattr(tools.chart_agent, "get_conn", make_conn)
    import tools.rca_agent
    monkeypatch.setattr(tools.rca_agent, "get_conn", make_conn)
    monkeypatch.setattr(tools.rca_agent, "get_model", lambda api_key=None: mock_model)
    import tools.whys_agent
    monkeypatch.setattr(tools.whys_agent, "get_conn", make_conn)
    monkeypatch.setattr(tools.whys_agent, "get_model", lambda api_key=None: mock_model)

    yield mock_model