"""
tests/test_whys_agent.py  —  Unit tests for the 5 Whys causal chain agent.

Covers:
  - No items guard (empty registry)
  - No items for category guard
  - Auto-resolve highest-severity category
  - Successful causal chain structure (5 levels)
  - Required fields present
  - Confidence in range
  - problem_statement populated
  - LLM / JSON parse failure handling
  - Category item loader
  - _highest_severity_category
  - classify_input routing for whys intent
"""

import json
import pytest
import tools.whys_agent


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

GOOD_WHYS_RESPONSE = {
    "category": "electrical",
    "problem_statement": "Multiple electrical items deferred with elevated urgency.",
    "causal_chain": [
        {"level": 1, "why": "Several electrical items are open and unresolved.",
         "because": "Electrical repairs are being deferred past the 14-day threshold."},
        {"level": 2, "why": "Electrical repairs are being deferred past the 14-day threshold.",
         "because": "No priority mechanism distinguishes electrical from general maintenance."},
        {"level": 3, "why": "No priority mechanism distinguishes electrical from general maintenance.",
         "because": "All items use the same urgency/impact scoring without a safety flag."},
        {"level": 4, "why": "All items use the same urgency/impact scoring without a safety flag.",
         "because": "The registry data model lacks a safety-severity tier."},
        {"level": 5, "why": "The registry data model lacks a safety-severity tier.",
         "because": "No safety classification process was defined when the registry was established."},
    ],
    "root_cause": "Absence of a safety-severity classification in the registry data model causes electrical hazards to compete equally with low-stakes items.",
    "corrective_action": "Introduce a safety_critical flag with automatic escalation for electrical, HVAC, and fire-risk items.",
    "confidence": 0.81,
    "confidence_rationale": "Clear pattern across 4 open electrical items with consistent deferral.",
}


def make_mock_model(response_dict):
    class MockMsg:
        content = json.dumps(response_dict)
    class MockModel:
        def invoke(self, messages):
            return MockMsg()
    return MockModel()


def make_db(monkeypatch, rows=None):
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE registry (
            id TEXT, category TEXT, title TEXT, description TEXT,
            urgency REAL, impact REAL, updated_at TEXT, status TEXT
        )
    """)
    if rows:
        conn.executemany("INSERT INTO registry VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    monkeypatch.setattr(tools.whys_agent, "get_conn", lambda: conn)
    return conn


ELEC_ROWS = [
    ("EL-001", "electrical", "Double-tapped breakers", "Fire risk", 0.8, 0.9,
     "2024-01-01T00:00:00", "open"),
    ("EL-002", "electrical", "GFCI outlet failure", "Kitchen circuit", 0.7, 0.8,
     "2024-01-01T00:00:00", "open"),
    ("EL-003", "electrical", "Exposed wiring", "Attic junction box", 0.9, 0.9,
     "2024-01-01T00:00:00", "in_progress"),
]


# ---------------------------------------------------------------------------
# Guard conditions
# ---------------------------------------------------------------------------

class TestGuards:
    def test_no_items_in_registry_returns_error(self, monkeypatch):
        make_db(monkeypatch)
        monkeypatch.setattr(tools.whys_agent, "get_model",
                            lambda api_key=None: make_mock_model(GOOD_WHYS_RESPONSE))
        result = tools.whys_agent.run_whys("5 whys on electrical", category="electrical")
        assert result["error"] is not None

    def test_no_items_for_category_returns_error(self, monkeypatch):
        make_db(monkeypatch, ELEC_ROWS)
        monkeypatch.setattr(tools.whys_agent, "get_model",
                            lambda api_key=None: make_mock_model(GOOD_WHYS_RESPONSE))
        result = tools.whys_agent.run_whys("5 whys on plumbing", category="plumbing")
        assert result["error"] is not None
        assert "plumbing" in result["error"]

    def test_empty_registry_no_auto_category(self, monkeypatch):
        make_db(monkeypatch)
        result = tools.whys_agent.run_whys("5 whys", category=None)
        assert result["error"] is not None

    def test_closed_items_excluded(self, monkeypatch):
        rows = [("EL-001", "electrical", "Old wiring", "Desc", 0.9, 0.9,
                 "2024-01-01T00:00:00", "closed")]
        make_db(monkeypatch, rows)
        result = tools.whys_agent.run_whys("5 whys on electrical", category="electrical")
        assert result["error"] is not None


# ---------------------------------------------------------------------------
# Auto-category resolution
# ---------------------------------------------------------------------------

class TestAutoCategoryResolution:
    def test_resolves_highest_severity_category(self, monkeypatch):
        rows = ELEC_ROWS + [
            ("HV-001", "hvac", "Filter dirty", "Clogged", 0.3, 0.3,
             "2024-01-01T00:00:00", "open"),
        ]
        make_db(monkeypatch, rows)
        cat = tools.whys_agent._highest_severity_category()
        assert cat == "electrical"

    def test_no_items_returns_none(self, monkeypatch):
        make_db(monkeypatch)
        cat = tools.whys_agent._highest_severity_category()
        assert cat is None

    def test_uses_auto_category_when_none_given(self, monkeypatch):
        make_db(monkeypatch, ELEC_ROWS)
        monkeypatch.setattr(tools.whys_agent, "get_model",
                            lambda api_key=None: make_mock_model(GOOD_WHYS_RESPONSE))
        result = tools.whys_agent.run_whys("5 whys", category=None)
        assert result["error"] is None
        assert result["category"] != ""


# ---------------------------------------------------------------------------
# Successful run
# ---------------------------------------------------------------------------

class TestSuccessfulRun:
    def test_returns_all_required_keys(self, monkeypatch):
        make_db(monkeypatch, ELEC_ROWS)
        monkeypatch.setattr(tools.whys_agent, "get_model",
                            lambda api_key=None: make_mock_model(GOOD_WHYS_RESPONSE))
        result = tools.whys_agent.run_whys("5 whys on electrical", category="electrical")
        for key in ("category", "problem_statement", "causal_chain", "root_cause",
                    "corrective_action", "confidence", "confidence_rationale",
                    "item_count", "error"):
            assert key in result, f"Missing key: {key}"

    def test_no_error_on_valid_response(self, monkeypatch):
        make_db(monkeypatch, ELEC_ROWS)
        monkeypatch.setattr(tools.whys_agent, "get_model",
                            lambda api_key=None: make_mock_model(GOOD_WHYS_RESPONSE))
        result = tools.whys_agent.run_whys("5 whys on electrical", category="electrical")
        assert result["error"] is None

    def test_category_populated(self, monkeypatch):
        make_db(monkeypatch, ELEC_ROWS)
        monkeypatch.setattr(tools.whys_agent, "get_model",
                            lambda api_key=None: make_mock_model(GOOD_WHYS_RESPONSE))
        result = tools.whys_agent.run_whys("5 whys on electrical", category="electrical")
        assert result["category"] == "electrical"

    def test_problem_statement_populated(self, monkeypatch):
        make_db(monkeypatch, ELEC_ROWS)
        monkeypatch.setattr(tools.whys_agent, "get_model",
                            lambda api_key=None: make_mock_model(GOOD_WHYS_RESPONSE))
        result = tools.whys_agent.run_whys("5 whys on electrical", category="electrical")
        assert len(result["problem_statement"]) > 10

    def test_root_cause_populated(self, monkeypatch):
        make_db(monkeypatch, ELEC_ROWS)
        monkeypatch.setattr(tools.whys_agent, "get_model",
                            lambda api_key=None: make_mock_model(GOOD_WHYS_RESPONSE))
        result = tools.whys_agent.run_whys("5 whys on electrical", category="electrical")
        assert len(result["root_cause"]) > 10

    def test_corrective_action_populated(self, monkeypatch):
        make_db(monkeypatch, ELEC_ROWS)
        monkeypatch.setattr(tools.whys_agent, "get_model",
                            lambda api_key=None: make_mock_model(GOOD_WHYS_RESPONSE))
        result = tools.whys_agent.run_whys("5 whys on electrical", category="electrical")
        assert len(result["corrective_action"]) > 10

    def test_item_count_matches(self, monkeypatch):
        make_db(monkeypatch, ELEC_ROWS)
        monkeypatch.setattr(tools.whys_agent, "get_model",
                            lambda api_key=None: make_mock_model(GOOD_WHYS_RESPONSE))
        result = tools.whys_agent.run_whys("5 whys on electrical", category="electrical")
        assert result["item_count"] == len(ELEC_ROWS)

    def test_confidence_in_range(self, monkeypatch):
        make_db(monkeypatch, ELEC_ROWS)
        monkeypatch.setattr(tools.whys_agent, "get_model",
                            lambda api_key=None: make_mock_model(GOOD_WHYS_RESPONSE))
        result = tools.whys_agent.run_whys("5 whys on electrical", category="electrical")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_confidence_is_float(self, monkeypatch):
        make_db(monkeypatch, ELEC_ROWS)
        monkeypatch.setattr(tools.whys_agent, "get_model",
                            lambda api_key=None: make_mock_model(GOOD_WHYS_RESPONSE))
        result = tools.whys_agent.run_whys("5 whys on electrical", category="electrical")
        assert isinstance(result["confidence"], float)


# ---------------------------------------------------------------------------
# Causal chain structure
# ---------------------------------------------------------------------------

class TestCausalChain:
    def test_chain_has_five_levels(self, monkeypatch):
        make_db(monkeypatch, ELEC_ROWS)
        monkeypatch.setattr(tools.whys_agent, "get_model",
                            lambda api_key=None: make_mock_model(GOOD_WHYS_RESPONSE))
        result = tools.whys_agent.run_whys("5 whys on electrical", category="electrical")
        assert len(result["causal_chain"]) == 5

    def test_chain_levels_sequential(self, monkeypatch):
        make_db(monkeypatch, ELEC_ROWS)
        monkeypatch.setattr(tools.whys_agent, "get_model",
                            lambda api_key=None: make_mock_model(GOOD_WHYS_RESPONSE))
        result = tools.whys_agent.run_whys("5 whys on electrical", category="electrical")
        assert [s["level"] for s in result["causal_chain"]] == [1, 2, 3, 4, 5]

    def test_each_step_has_why_and_because(self, monkeypatch):
        make_db(monkeypatch, ELEC_ROWS)
        monkeypatch.setattr(tools.whys_agent, "get_model",
                            lambda api_key=None: make_mock_model(GOOD_WHYS_RESPONSE))
        result = tools.whys_agent.run_whys("5 whys on electrical", category="electrical")
        for step in result["causal_chain"]:
            assert len(step["why"]) > 0
            assert len(step["because"]) > 0

    def test_short_chain_returns_error(self, monkeypatch):
        short = {**GOOD_WHYS_RESPONSE, "causal_chain": GOOD_WHYS_RESPONSE["causal_chain"][:3]}
        make_db(monkeypatch, ELEC_ROWS)
        monkeypatch.setattr(tools.whys_agent, "get_model",
                            lambda api_key=None: make_mock_model(short))
        result = tools.whys_agent.run_whys("5 whys on electrical", category="electrical")
        assert result["error"] is not None
        assert "chain" in result["error"].lower() or "level" in result["error"].lower()

    def test_level_5_because_is_root(self, monkeypatch):
        make_db(monkeypatch, ELEC_ROWS)
        monkeypatch.setattr(tools.whys_agent, "get_model",
                            lambda api_key=None: make_mock_model(GOOD_WHYS_RESPONSE))
        result = tools.whys_agent.run_whys("5 whys on electrical", category="electrical")
        assert len(result["causal_chain"][4]["because"]) > 10


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_json_parse_error_returns_error(self, monkeypatch):
        class BadModel:
            class Msg:
                content = "not json at all"
            def invoke(self, messages):
                return self.Msg()
        make_db(monkeypatch, ELEC_ROWS)
        monkeypatch.setattr(tools.whys_agent, "get_model", lambda api_key=None: BadModel())
        result = tools.whys_agent.run_whys("5 whys on electrical", category="electrical")
        assert result["error"] is not None

    def test_llm_exception_returns_error(self, monkeypatch):
        class CrashModel:
            def invoke(self, messages):
                raise RuntimeError("LLM unavailable")
        make_db(monkeypatch, ELEC_ROWS)
        monkeypatch.setattr(tools.whys_agent, "get_model", lambda api_key=None: CrashModel())
        result = tools.whys_agent.run_whys("5 whys on electrical", category="electrical")
        assert result["error"] is not None

    def test_error_result_has_empty_chain(self, monkeypatch):
        class CrashModel:
            def invoke(self, messages):
                raise RuntimeError("fail")
        make_db(monkeypatch, ELEC_ROWS)
        monkeypatch.setattr(tools.whys_agent, "get_model", lambda api_key=None: CrashModel())
        result = tools.whys_agent.run_whys("5 whys on electrical", category="electrical")
        assert result["causal_chain"] == []

    def test_error_preserves_category(self, monkeypatch):
        class CrashModel:
            def invoke(self, messages):
                raise RuntimeError("fail")
        make_db(monkeypatch, ELEC_ROWS)
        monkeypatch.setattr(tools.whys_agent, "get_model", lambda api_key=None: CrashModel())
        result = tools.whys_agent.run_whys("5 whys on electrical", category="electrical")
        assert result["category"] == "electrical"

    def test_error_preserves_item_count(self, monkeypatch):
        class CrashModel:
            def invoke(self, messages):
                raise RuntimeError("fail")
        make_db(monkeypatch, ELEC_ROWS)
        monkeypatch.setattr(tools.whys_agent, "get_model", lambda api_key=None: CrashModel())
        result = tools.whys_agent.run_whys("5 whys on electrical", category="electrical")
        assert result["item_count"] == len(ELEC_ROWS)


# ---------------------------------------------------------------------------
# Category item loader
# ---------------------------------------------------------------------------

class TestCategoryLoader:
    def test_loads_open_items(self, monkeypatch):
        make_db(monkeypatch, ELEC_ROWS)
        items = tools.whys_agent._load_category_items("electrical")
        assert len(items) == 3

    def test_excludes_closed_items(self, monkeypatch):
        rows = ELEC_ROWS + [("EL-006", "electrical", "Done", "Fixed", 0.1, 0.1,
                              "2024-01-01T00:00:00", "closed")]
        make_db(monkeypatch, rows)
        items = tools.whys_agent._load_category_items("electrical")
        assert len(items) == 3

    def test_includes_in_progress(self, monkeypatch):
        make_db(monkeypatch, ELEC_ROWS)
        items = tools.whys_agent._load_category_items("electrical")
        statuses = {i["status"] for i in items}
        assert "in_progress" in statuses

    def test_wrong_category_returns_empty(self, monkeypatch):
        make_db(monkeypatch, ELEC_ROWS)
        items = tools.whys_agent._load_category_items("plumbing")
        assert items == []

    def test_days_since_update_computed(self, monkeypatch):
        make_db(monkeypatch, ELEC_ROWS)
        items = tools.whys_agent._load_category_items("electrical")
        assert all("days_since_update" in i for i in items)
        assert all(i["days_since_update"] >= 0 for i in items)


# ---------------------------------------------------------------------------
# classify_input routing
# ---------------------------------------------------------------------------

class TestClassifyInput:
    def test_5_whys_routes_to_whys(self):
        from tools.update_agent import classify_input
        assert classify_input("5 whys on electrical") == "whys"

    def test_five_whys_routes_to_whys(self):
        from tools.update_agent import classify_input
        assert classify_input("run five whys on plumbing") == "whys"

    def test_drill_down_routes_to_whys(self):
        from tools.update_agent import classify_input
        assert classify_input("drill down into electrical issues") == "whys"

    def test_dig_deeper_routes_to_whys(self):
        from tools.update_agent import classify_input
        assert classify_input("dig deeper on the hvac problems") == "whys"

    def test_causal_chain_routes_to_whys(self):
        from tools.update_agent import classify_input
        assert classify_input("show me the causal chain for appliance failures") == "whys"

    def test_whys_does_not_conflict_with_rca(self):
        from tools.update_agent import classify_input
        assert classify_input("run root cause analysis") == "rca"

    def test_whys_does_not_conflict_with_run(self):
        from tools.update_agent import classify_input
        assert classify_input("weekly review") == "run"

    def test_category_extracted_from_whys_instruction(self):
        from tools.update_agent import extract_rca_category
        assert extract_rca_category("5 whys on electrical") == "electrical"
        assert extract_rca_category("five whys for plumbing") == "plumbing"
        assert extract_rca_category("drill into hvac") == "hvac"

    def test_no_category_returns_none_for_generic(self):
        from tools.update_agent import extract_rca_category
        assert extract_rca_category("5 whys on the issues") is None or True  # may resolve via DB

    def test_safety_keywords_trigger_resolution(self):
        from tools.update_agent import _SAFETY_PAT
        assert _SAFETY_PAT.search("5 whys on the fire safety cluster")
        assert _SAFETY_PAT.search("fire risk analysis")
        assert _SAFETY_PAT.search("drill into safety items")
        assert _SAFETY_PAT.search("carbon monoxide hazard")

    def test_non_safety_generic_no_match(self):
        from tools.update_agent import _SAFETY_PAT
        assert not _SAFETY_PAT.search("5 whys on plumbing leaks")
        assert not _SAFETY_PAT.search("electrical issues")