"""
test_graph.py  -  Integration tests for the LangGraph graph definition.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from graph.graph import graph, build_graph
from graph.state import HombaseState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def initial_state():
    return {
        "trigger": "test run",
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


# ---------------------------------------------------------------------------
# Graph structure
# ---------------------------------------------------------------------------

class TestGraphStructure:
    def test_graph_compiles(self):
        g = build_graph()
        assert g is not None

    def test_graph_has_orchestrator_node(self):
        g = build_graph()
        assert "orchestrator" in g.nodes

    def test_module_level_graph_exists(self):
        assert graph is not None


# ---------------------------------------------------------------------------
# Graph execution  -  integration
# ---------------------------------------------------------------------------

class TestGraphExecution:
    def test_graph_invokes_without_error(self, initial_state):
        result = graph.invoke(initial_state)
        assert result is not None

    def test_result_contains_summary_report(self, initial_state):
        result = graph.invoke(initial_state)
        assert "summary_report" in result
        assert len(result["summary_report"]) > 0

    def test_result_contains_messages(self, initial_state):
        result = graph.invoke(initial_state)
        assert "messages" in result
        assert len(result["messages"]) > 0

    def test_result_contains_classified_items(self, initial_state):
        result = graph.invoke(initial_state)
        assert "classified_items" in result
        assert len(result["classified_items"]) > 0

    def test_all_state_keys_present_in_result(self, initial_state):
        result = graph.invoke(initial_state)
        for key in initial_state.keys():
            assert key in result, f"Missing key in result: {key}"

    def test_trigger_preserved_in_result(self, initial_state):
        result = graph.invoke(initial_state)
        assert result["trigger"] == "test run"

    def test_different_triggers_produce_different_reports(self, initial_state):
        state_a = {**initial_state, "trigger": "morning briefing"}
        state_b = {**initial_state, "trigger": "weekly home review"}
        result_a = graph.invoke(state_a)
        result_b = graph.invoke(state_b)
        assert result_a["summary_report"] != result_b["summary_report"]

    def test_bucket_keys_all_present(self, initial_state):
        result = graph.invoke(initial_state)
        for key in ["hu_hi", "hu_li", "lu_hi", "lu_li", "stale_items"]:
            assert key in result

    def test_bucket_counts_sum_to_total(self, initial_state):
        result = graph.invoke(initial_state)
        total = len(result["classified_items"])
        bucket_sum = (
            len(result["hu_hi"]) +
            len(result["hu_li"]) +
            len(result["lu_hi"]) +
            len(result["lu_li"])
        )
        assert bucket_sum == total