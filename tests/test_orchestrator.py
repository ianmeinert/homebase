"""
test_orchestrator.py  -  Unit tests for the orchestrator agent node.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agents.orchestrator import orchestrator_node, build_report, format_item


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_state():
    return {
        "trigger": "weekly home review",
        "raw_registry": [],
        "classified_items": [],
        "hu_hi": [],
        "hu_li": [],
        "lu_hi": [],
        "lu_li": [],
        "stale_items": [],
        "summary_report": "",
        "messages": [],
    }


@pytest.fixture
def sample_buckets():
    hu_hi_item = {"id": "T-001", "category": "hvac", "title": "Critical HVAC", "description": "HVAC unit failing.", "urgency": 0.8, "impact": 0.8, "days_since_update": 20, "status": "open", "quadrant": "HU/HI"}
    lu_li_item  = {"id": "T-002", "category": "general", "title": "Low priority paint", "description": "Peeling trim paint.", "urgency": 0.2, "impact": 0.2, "days_since_update": 5, "status": "open", "quadrant": "LU/LI"}
    return {
        "hu_hi": [hu_hi_item],
        "hu_li": [],
        "lu_hi": [],
        "lu_li": [lu_li_item],
        "stale_items": [hu_hi_item],
        "all": [hu_hi_item, lu_li_item],
    }


# ---------------------------------------------------------------------------
# orchestrator_node
# ---------------------------------------------------------------------------

class TestOrchestratorNode:
    def test_returns_dict(self, base_state):
        result = orchestrator_node(base_state)
        assert isinstance(result, dict)

    def test_populates_raw_registry(self, base_state):
        result = orchestrator_node(base_state)
        assert isinstance(result["raw_registry"], list)
        assert len(result["raw_registry"]) > 0

    def test_populates_classified_items(self, base_state):
        result = orchestrator_node(base_state)
        assert isinstance(result["classified_items"], list)
        assert len(result["classified_items"]) == len(result["raw_registry"])

    def test_all_classified_items_have_quadrant(self, base_state):
        result = orchestrator_node(base_state)
        for item in result["classified_items"]:
            assert "quadrant" in item
            assert item["quadrant"] in {"HU/HI", "HU/LI", "LU/HI", "LU/LI"}

    def test_buckets_populated(self, base_state):
        result = orchestrator_node(base_state)
        bucket_total = (
            len(result["hu_hi"]) +
            len(result["hu_li"]) +
            len(result["lu_hi"]) +
            len(result["lu_li"])
        )
        assert bucket_total == len(result["classified_items"])

    def test_summary_report_is_string(self, base_state):
        result = orchestrator_node(base_state)
        assert isinstance(result["summary_report"], str)
        assert len(result["summary_report"]) > 0

    def test_messages_populated(self, base_state):
        result = orchestrator_node(base_state)
        assert isinstance(result["messages"], list)
        assert len(result["messages"]) >= 4  # At minimum: trigger, load, classify, report msgs

    def test_trigger_appears_in_messages(self, base_state):
        result = orchestrator_node(base_state)
        trigger = base_state["trigger"]
        assert any(trigger in msg for msg in result["messages"])

    def test_trigger_appears_in_report(self, base_state):
        result = orchestrator_node(base_state)
        assert base_state["trigger"] in result["summary_report"]

    def test_custom_trigger_reflected(self):
        state = {
            "trigger": "morning briefing",
            "raw_registry": [], "classified_items": [], "hu_hi": [], "hu_li": [],
            "lu_hi": [], "lu_li": [], "stale_items": [], "summary_report": "", "messages": [],
        }
        result = orchestrator_node(state)
        assert "morning briefing" in result["summary_report"]


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------

class TestBuildReport:
    def test_returns_string(self, sample_buckets):
        result = build_report(sample_buckets, "test trigger")
        assert isinstance(result, str)

    def test_trigger_in_report(self, sample_buckets):
        result = build_report(sample_buckets, "weekly home review")
        assert "weekly home review" in result

    def test_quadrant_labels_present(self, sample_buckets):
        result = build_report(sample_buckets, "test")
        assert "HU/HI" in result
        assert "HU/LI" in result
        assert "LU/HI" in result
        assert "LU/LI" in result

    def test_item_id_in_report(self, sample_buckets):
        result = build_report(sample_buckets, "test")
        assert "T-001" in result
        assert "T-002" in result

    def test_stale_flag_in_report(self, sample_buckets):
        result = build_report(sample_buckets, "test")
        assert "STALE" in result

    def test_summary_counts_correct(self, sample_buckets):
        result = build_report(sample_buckets, "test")
        assert "2 open items" in result
        assert "1 stale" in result

    def test_empty_buckets_handled(self):
        empty_buckets = {"hu_hi": [], "hu_li": [], "lu_hi": [], "lu_li": [], "stale_items": [], "all": []}
        result = build_report(empty_buckets, "test")
        assert "0 open items" in result
        assert isinstance(result, str)

    def test_phase_placeholders_present(self, sample_buckets):
        result = build_report(sample_buckets, "test")
        assert "Phase 3" in result


# ---------------------------------------------------------------------------
# format_item
# ---------------------------------------------------------------------------

class TestFormatItem:
    def test_includes_item_id(self):
        item = {"id": "X-001", "title": "Test", "urgency": 0.5, "impact": 0.5, "days_since_update": 5, "description": "desc"}
        result = format_item(item)
        assert "X-001" in result

    def test_includes_title(self):
        item = {"id": "X-001", "title": "My Title", "urgency": 0.5, "impact": 0.5, "days_since_update": 5, "description": "desc"}
        result = format_item(item)
        assert "My Title" in result

    def test_stale_flag_shown_at_14_days(self):
        item = {"id": "X", "title": "T", "urgency": 0.5, "impact": 0.5, "days_since_update": 14, "description": "d"}
        result = format_item(item)
        assert "STALE" in result

    def test_stale_flag_not_shown_at_13_days(self):
        item = {"id": "X", "title": "T", "urgency": 0.5, "impact": 0.5, "days_since_update": 13, "description": "d"}
        result = format_item(item)
        assert "STALE" not in result

    def test_urgency_and_impact_shown(self):
        item = {"id": "X", "title": "T", "urgency": 0.75, "impact": 0.85, "days_since_update": 1, "description": "d"}
        result = format_item(item)
        assert "0.75" in result
        assert "0.85" in result