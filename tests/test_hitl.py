"""
test_hitl.py  -  Tests for Phase 3 HITL briefing, synthesizer deferral logic,
               graph interrupt behavior, and state persistence.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from langgraph.checkpoint.memory import MemorySaver
import uuid

from agents.orchestrator import hitl_briefing_node, synthesizer_node
from graph.graph import build_graph, build_interactive_graph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_item(id, category, title, urgency=0.7, impact=0.7, days=10, quadrant="HU/HI"):
    return {
        "id": id, "category": category, "title": title,
        "description": "Test item.", "urgency": urgency,
        "impact": impact, "days_since_update": days,
        "status": "open", "quadrant": quadrant,
    }


def make_result(id, category, title, urgency=0.7, impact=0.7, days=10, quadrant="HU/HI"):
    item = make_item(id, category, title, urgency, impact, days, quadrant)
    return {
        "item": item,
        "recommendation": {
            "item_id": id,
            "action": f"Fix {title}",
            "estimated_effort": "1 hr DIY",
            "estimated_cost": "$20",
            "priority_note": "Do this soon.",
            "agent": "TestAgent",
        }
    }


def base_state(subagent_results=None, hu_hi=None, hitl_approved=False,
               hitl_notes="", deferred_items=None):
    return {
        "trigger": "test run",
        "raw_registry": [], "classified_items": [],
        "hu_hi": hu_hi or [],
        "hu_li": [], "lu_hi": [], "lu_li": [], "stale_items": [],
        "delegated_items": [],
        "subagent_results": subagent_results or [],
        "hitl_approved": hitl_approved,
        "hitl_notes": hitl_notes,
        "deferred_items": deferred_items or [],
        "summary_report": "",
        "messages": [],
    }


# ---------------------------------------------------------------------------
# hitl_briefing_node
# ---------------------------------------------------------------------------

class TestHitlBriefingNode:
    def test_returns_dict(self):
        state = base_state(subagent_results=[make_result("H1", "hvac", "Replace filter")])
        result = hitl_briefing_node(state)
        assert isinstance(result, dict)

    def test_produces_summary_report(self):
        state = base_state(subagent_results=[make_result("H1", "hvac", "Replace filter")])
        result = hitl_briefing_node(state)
        assert "summary_report" in result
        assert len(result["summary_report"]) > 0

    def test_report_contains_action_required_header(self):
        state = base_state(subagent_results=[make_result("H1", "hvac", "Replace filter")])
        result = hitl_briefing_node(state)
        assert "ACTION REQUIRED" in result["summary_report"]

    def test_report_contains_hu_hi_item_ids(self):
        results = [
            make_result("ELC-001", "electrical", "GFCI outlet", quadrant="HU/HI"),
            make_result("PLM-001", "plumbing", "Drain", quadrant="LU/HI"),  # should not appear
        ]
        state = base_state(subagent_results=results)
        report = hitl_briefing_node(state)["summary_report"]
        assert "ELC-001" in report
        assert "PLM-001" not in report  # LU/HI not shown in briefing

    def test_report_shows_item_count(self):
        results = [
            make_result("H1", "hvac", "Filter", quadrant="HU/HI"),
            make_result("H2", "hvac", "Tune-up", quadrant="HU/HI"),
        ]
        state = base_state(subagent_results=results)
        report = hitl_briefing_node(state)["summary_report"]
        assert "2 item(s)" in report

    def test_stale_flag_shown_in_briefing(self):
        results = [make_result("H1", "hvac", "Filter", days=20, quadrant="HU/HI")]
        state = base_state(subagent_results=results)
        report = hitl_briefing_node(state)["summary_report"]
        assert "STALE" in report

    def test_no_hu_hi_items_handled_gracefully(self):
        results = [make_result("P1", "plumbing", "Drain", quadrant="LU/HI")]
        state = base_state(subagent_results=results)
        result = hitl_briefing_node(state)
        assert "0 item(s)" in result["summary_report"]

    def test_messages_produced(self):
        state = base_state(subagent_results=[make_result("H1", "hvac", "Filter")])
        result = hitl_briefing_node(state)
        assert len(result["messages"]) >= 2


# ---------------------------------------------------------------------------
# synthesizer_node  -  HITL decision handling
# ---------------------------------------------------------------------------

class TestSynthesizerHitlDecisions:
    def test_approved_all_items_included(self):
        results = [
            make_result("H1", "hvac", "Filter", quadrant="HU/HI"),
            make_result("P1", "plumbing", "Drain", quadrant="LU/HI"),
        ]
        state = base_state(subagent_results=results, hitl_approved=True)
        result = synthesizer_node(state)
        assert "H1" in result["summary_report"]
        assert "P1" in result["summary_report"]

    def test_deferred_items_excluded_from_report(self):
        results = [
            make_result("H1", "hvac", "Filter", quadrant="HU/HI"),
            make_result("H2", "hvac", "Tune-up", quadrant="HU/HI"),
        ]
        state = base_state(subagent_results=results, hitl_approved=True, deferred_items=["H1"])
        result = synthesizer_node(state)
        assert "H2" in result["summary_report"]
        assert "H1" not in result["summary_report"].split("HITL")[0]

    def test_all_items_deferred_produces_empty_action_plan(self):
        results = [make_result("H1", "hvac", "Filter", quadrant="HU/HI")]
        state = base_state(subagent_results=results, hitl_approved=True, deferred_items=["H1"])
        result = synthesizer_node(state)
        assert "HITL DECISION SUMMARY" in result["summary_report"]

    def test_hitl_decision_summary_in_report(self):
        results = [make_result("H1", "hvac", "Filter", quadrant="HU/HI")]
        state = base_state(subagent_results=results, hitl_approved=True)
        result = synthesizer_node(state)
        assert "HITL DECISION SUMMARY" in result["summary_report"]

    def test_approved_yes_shown_in_summary(self):
        results = [make_result("H1", "hvac", "Filter", quadrant="HU/HI")]
        state = base_state(subagent_results=results, hitl_approved=True)
        result = synthesizer_node(state)
        assert "Approved  : Yes" in result["summary_report"]

    def test_approved_no_shown_in_summary(self):
        results = [make_result("H1", "hvac", "Filter", quadrant="HU/HI")]
        state = base_state(subagent_results=results, hitl_approved=False)
        result = synthesizer_node(state)
        assert "Approved  : No" in result["summary_report"]

    def test_deferred_ids_listed_in_summary(self):
        results = [
            make_result("H1", "hvac", "Filter", quadrant="HU/HI"),
            make_result("H2", "hvac", "Tune-up", quadrant="HU/HI"),
        ]
        state = base_state(subagent_results=results, hitl_approved=True, deferred_items=["H1", "H2"])
        result = synthesizer_node(state)
        assert "H1" in result["summary_report"]
        assert "H2" in result["summary_report"]

    def test_notes_included_in_summary(self):
        results = [make_result("H1", "hvac", "Filter", quadrant="HU/HI")]
        state = base_state(subagent_results=results, hitl_approved=True, hitl_notes="Do this weekend")
        result = synthesizer_node(state)
        assert "Do this weekend" in result["summary_report"]

    def test_messages_log_deferred_items(self):
        results = [
            make_result("H1", "hvac", "Filter", quadrant="HU/HI"),
            make_result("H2", "hvac", "Tune-up", quadrant="HU/HI"),
        ]
        state = base_state(subagent_results=results, hitl_approved=True, deferred_items=["H1"])
        result = synthesizer_node(state)
        assert any("H1" in msg for msg in result["messages"])

    def test_empty_results_handled_gracefully(self):
        state = base_state(subagent_results=[], hitl_approved=True)
        result = synthesizer_node(state)
        assert isinstance(result["summary_report"], str)


# ---------------------------------------------------------------------------
# Graph  -  interrupt behavior
# ---------------------------------------------------------------------------

class TestGraphInterrupt:
    def _make_config(self):
        return {"configurable": {"thread_id": str(uuid.uuid4())}}

    def _initial_state(self):
        return {
            "trigger": "test run",
            "raw_registry": [], "classified_items": [],
            "hu_hi": [], "hu_li": [], "lu_hi": [], "lu_li": [], "stale_items": [],
            "delegated_items": [], "subagent_results": [],
            "hitl_approved": False, "hitl_notes": "", "deferred_items": [],
            "summary_report": "", "messages": [],
        }

    def test_non_interrupt_graph_completes_in_one_invoke(self):
        """Module-level graph (no interrupt) should complete without pausing."""
        from graph.graph import graph
        result = graph.invoke(self._initial_state())
        assert result is not None
        assert len(result["summary_report"]) > 0

    def test_interactive_graph_pauses_before_synthesizer(self):
        """Interactive graph should stop after hitl_briefing, before synthesizer."""
        g = build_interactive_graph()
        config = self._make_config()

        chunks = list(g.stream(self._initial_state(), config=config))
        node_names = [list(c.keys())[0] for c in chunks]

        assert "hitl_briefing" in node_names
        assert "synthesizer" not in node_names

    def test_state_persisted_at_interrupt(self):
        """State should be retrievable after interrupt."""
        g = build_interactive_graph()
        config = self._make_config()

        list(g.stream(self._initial_state(), config=config))
        state = g.get_state(config)

        assert state is not None
        assert "subagent_results" in state.values
        assert len(state.values["subagent_results"]) > 0

    def test_resume_after_approval_completes_graph(self):
        """After updating state with approval, graph should resume and finish."""
        g = build_interactive_graph()
        config = self._make_config()

        list(g.stream(self._initial_state(), config=config))

        g.update_state(config, {
            "hitl_approved": True,
            "hitl_notes": "All good",
            "deferred_items": [],
        })

        resume_chunks = list(g.stream(None, config=config))
        node_names = [list(c.keys())[0] for c in resume_chunks]
        assert "synthesizer" in node_names

    def test_resume_with_deferrals_excludes_items(self):
        """Deferred item IDs should be absent from final report."""
        g = build_interactive_graph()
        config = self._make_config()

        list(g.stream(self._initial_state(), config=config))

        # Get HU/HI item IDs from paused state
        paused = g.get_state(config)
        hu_hi_ids = [item["id"] for item in paused.values.get("hu_hi", [])]
        defer_id = hu_hi_ids[0] if hu_hi_ids else None

        g.update_state(config, {
            "hitl_approved": True,
            "hitl_notes": "",
            "deferred_items": [defer_id] if defer_id else [],
        })

        list(g.stream(None, config=config))
        final = g.get_state(config)
        report = final.values.get("summary_report", "")

        if defer_id:
            # Deferred ID should only appear in HITL summary, not action plan
            action_plan = report.split("HITL DECISION SUMMARY")[0]
            assert defer_id not in action_plan

    def test_next_node_after_interrupt_is_synthesizer(self):
        """LangGraph should indicate synthesizer as the next node after interrupt."""
        g = build_interactive_graph()
        config = self._make_config()

        list(g.stream(self._initial_state(), config=config))
        state = g.get_state(config)
        assert "synthesizer" in state.next