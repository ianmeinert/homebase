"""
test_subagents.py  -  Unit tests for specialist subagent nodes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agents.subagents import (
    hvac_agent_node,
    plumbing_agent_node,
    electrical_agent_node,
    appliance_agent_node,
    general_agent_node,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_item(id, category, title, quadrant="HU/HI", days=10):
    return {
        "id": id, "category": category, "title": title,
        "description": "Test.", "urgency": 0.7, "impact": 0.7,
        "days_since_update": days, "status": "open", "quadrant": quadrant,
    }


def base_state(delegated_items=None):
    return {
        "trigger": "test",
        "raw_registry": [], "classified_items": [],
        "hu_hi": [], "hu_li": [], "lu_hi": [], "lu_li": [], "stale_items": [],
        "delegated_items": delegated_items or [],
        "subagent_results": [],
        "summary_report": "",
        "messages": [],
    }


# ---------------------------------------------------------------------------
# Each subagent processes only its own category
# ---------------------------------------------------------------------------

class TestAgentCategoryFiltering:
    def test_hvac_agent_only_processes_hvac(self):
        items = [
            make_item("H1", "hvac", "Replace filter"),
            make_item("P1", "plumbing", "Fix drain"),   # should be ignored
        ]
        state = base_state(items)
        result = hvac_agent_node(state)
        assert len(result["subagent_results"]) == 1
        assert result["subagent_results"][0]["item"]["id"] == "H1"

    def test_plumbing_agent_only_processes_plumbing(self):
        items = [
            make_item("H1", "hvac", "Replace filter"),
            make_item("P1", "plumbing", "Fix drain"),
        ]
        state = base_state(items)
        result = plumbing_agent_node(state)
        assert len(result["subagent_results"]) == 1
        assert result["subagent_results"][0]["item"]["id"] == "P1"

    def test_electrical_agent_only_processes_electrical(self):
        items = [
            make_item("E1", "electrical", "Replace GFCI outlet"),
            make_item("G1", "general", "Clean gutters"),
        ]
        state = base_state(items)
        result = electrical_agent_node(state)
        assert len(result["subagent_results"]) == 1
        assert result["subagent_results"][0]["item"]["id"] == "E1"

    def test_appliance_agent_only_processes_appliances(self):
        items = [
            make_item("A1", "appliance", "Dryer vent cleaning overdue"),
            make_item("E1", "electrical", "GFCI outlet"),
        ]
        state = base_state(items)
        result = appliance_agent_node(state)
        assert len(result["subagent_results"]) == 1
        assert result["subagent_results"][0]["item"]["id"] == "A1"

    def test_general_agent_only_processes_general(self):
        items = [
            make_item("G1", "general", "Clean gutters"),
            make_item("H1", "hvac", "HVAC filter"),
        ]
        state = base_state(items)
        result = general_agent_node(state)
        assert len(result["subagent_results"]) == 1
        assert result["subagent_results"][0]["item"]["id"] == "G1"


# ---------------------------------------------------------------------------
# Empty delegated_items  -  agents exit gracefully
# ---------------------------------------------------------------------------

class TestAgentEmptyDelegation:
    def test_hvac_agent_handles_no_items(self):
        result = hvac_agent_node(base_state([]))
        assert result.get("subagent_results", []) == [] or "subagent_results" not in result
        assert len(result["messages"]) > 0

    def test_plumbing_agent_handles_no_items(self):
        result = plumbing_agent_node(base_state([]))
        assert result.get("subagent_results", []) == [] or "subagent_results" not in result

    def test_electrical_agent_handles_no_items(self):
        result = electrical_agent_node(base_state([]))
        assert result.get("subagent_results", []) == [] or "subagent_results" not in result

    def test_appliance_agent_handles_no_items(self):
        result = appliance_agent_node(base_state([]))
        assert result.get("subagent_results", []) == [] or "subagent_results" not in result

    def test_general_agent_handles_no_items(self):
        result = general_agent_node(base_state([]))
        assert result.get("subagent_results", []) == [] or "subagent_results" not in result


# ---------------------------------------------------------------------------
# Result structure validation
# ---------------------------------------------------------------------------

class TestAgentResultStructure:
    def test_result_contains_item_and_recommendation(self):
        items = [make_item("H1", "hvac", "Replace HVAC air filter")]
        result = hvac_agent_node(base_state(items))
        assert len(result["subagent_results"]) == 1
        entry = result["subagent_results"][0]
        assert "item" in entry
        assert "recommendation" in entry

    def test_recommendation_has_required_fields(self):
        items = [make_item("H1", "hvac", "Replace HVAC air filter")]
        result = hvac_agent_node(base_state(items))
        rec = result["subagent_results"][0]["recommendation"]
        required = {"item_id", "action", "estimated_effort", "estimated_cost", "priority_note", "agent"}
        assert required.issubset(rec.keys())

    def test_messages_are_produced(self):
        items = [make_item("H1", "hvac", "Replace HVAC air filter")]
        result = hvac_agent_node(base_state(items))
        assert len(result["messages"]) >= 3  # activated + processing + complete

    def test_multiple_items_all_processed(self):
        items = [
            make_item("G1", "general", "Clean gutters"),
            make_item("G2", "general", "Test smoke detectors"),
            make_item("G3", "general", "Caulk shower"),
        ]
        result = general_agent_node(base_state(items))
        assert len(result["subagent_results"]) == 3

    def test_existing_results_preserved(self):
        """Subagent should append to existing results, not overwrite."""
        existing = [{"item": {"id": "PREV"}, "recommendation": {"agent": "OtherAgent"}}]
        items = [make_item("H1", "hvac", "Replace HVAC air filter")]
        state = base_state(items)
        state["subagent_results"] = existing
        result = hvac_agent_node(state)
        ids = [r["item"]["id"] for r in result["subagent_results"]]
        assert "PREV" in ids
        assert "H1" in ids