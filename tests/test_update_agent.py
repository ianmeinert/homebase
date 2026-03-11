"""
tests/test_update_agent.py  -  Tests for the natural language update agent.

All LLM calls mocked via conftest autouse fixture.
"""

import json
import pytest
from unittest.mock import patch, MagicMock


SAMPLE_ITEM = {
    "id": "PLB-001",
    "category": "plumbing",
    "title": "Fix slow drain in master bath",
    "description": "Slow drain for 3 weeks, getting worse.",
    "urgency": 0.7,
    "impact": 0.5,
    "days_since_update": 21,
    "status": "open",
}


def _mock_llm_response(payload: dict):
    """Return a mock LLM response with the given JSON payload."""
    mock = MagicMock()
    mock.content = json.dumps(payload)
    return mock


# ---------------------------------------------------------------------------
# interpret_update
# ---------------------------------------------------------------------------

class TestInterpretUpdate:

    def test_close_instruction(self, monkeypatch):
        from tools.update_agent import interpret_update
        monkeypatch.setattr("tools.update_agent.get_model", lambda **kw: MagicMock(
            invoke=lambda msgs: _mock_llm_response({"status": "closed"})
        ))
        result = interpret_update(SAMPLE_ITEM, "mark as resolved")
        assert result == {"status": "closed"}

    def test_urgency_update(self, monkeypatch):
        from tools.update_agent import interpret_update
        monkeypatch.setattr("tools.update_agent.get_model", lambda **kw: MagicMock(
            invoke=lambda msgs: _mock_llm_response({"urgency": 0.9})
        ))
        result = interpret_update(SAMPLE_ITEM, "raise urgency to 0.9")
        assert result == {"urgency": 0.9}

    def test_reset_clock(self, monkeypatch):
        from tools.update_agent import interpret_update
        monkeypatch.setattr("tools.update_agent.get_model", lambda **kw: MagicMock(
            invoke=lambda msgs: _mock_llm_response({"days_since_update": 0})
        ))
        result = interpret_update(SAMPLE_ITEM, "reset the clock")
        assert result == {"days_since_update": 0}

    def test_multiple_fields(self, monkeypatch):
        from tools.update_agent import interpret_update
        monkeypatch.setattr("tools.update_agent.get_model", lambda **kw: MagicMock(
            invoke=lambda msgs: _mock_llm_response({
                "urgency": 0.95, "description": "Getting much worse."
            })
        ))
        result = interpret_update(SAMPLE_ITEM, "bump urgency to 0.95 and update description")
        assert result["urgency"] == 0.95
        assert result["description"] == "Getting much worse."

    def test_urgency_clamped_high(self, monkeypatch):
        from tools.update_agent import interpret_update
        monkeypatch.setattr("tools.update_agent.get_model", lambda **kw: MagicMock(
            invoke=lambda msgs: _mock_llm_response({"urgency": 1.5})
        ))
        result = interpret_update(SAMPLE_ITEM, "max urgency")
        assert result["urgency"] == 1.0

    def test_urgency_clamped_low(self, monkeypatch):
        from tools.update_agent import interpret_update
        monkeypatch.setattr("tools.update_agent.get_model", lambda **kw: MagicMock(
            invoke=lambda msgs: _mock_llm_response({"urgency": -0.2})
        ))
        result = interpret_update(SAMPLE_ITEM, "very low urgency")
        assert result["urgency"] == 0.0

    def test_invalid_status_rejected(self, monkeypatch):
        from tools.update_agent import interpret_update
        monkeypatch.setattr("tools.update_agent.get_model", lambda **kw: MagicMock(
            invoke=lambda msgs: _mock_llm_response({"status": "pending"})
        ))
        result = interpret_update(SAMPLE_ITEM, "mark as pending")
        assert "status" not in result

    def test_disallowed_field_rejected(self, monkeypatch):
        from tools.update_agent import interpret_update
        monkeypatch.setattr("tools.update_agent.get_model", lambda **kw: MagicMock(
            invoke=lambda msgs: _mock_llm_response({"id": "HACKED", "urgency": 0.5})
        ))
        result = interpret_update(SAMPLE_ITEM, "change id")
        assert "id" not in result
        assert result.get("urgency") == 0.5

    def test_empty_response(self, monkeypatch):
        from tools.update_agent import interpret_update
        monkeypatch.setattr("tools.update_agent.get_model", lambda **kw: MagicMock(
            invoke=lambda msgs: _mock_llm_response({})
        ))
        result = interpret_update(SAMPLE_ITEM, "do nothing")
        assert result == {}

    def test_llm_exception_returns_error(self, monkeypatch):
        from tools.update_agent import interpret_update
        def bad_model(**kw):
            m = MagicMock()
            m.invoke.side_effect = RuntimeError("API down")
            return m
        monkeypatch.setattr("tools.update_agent.get_model", bad_model)
        result = interpret_update(SAMPLE_ITEM, "update something")
        assert "_error" in result

    def test_malformed_json_returns_error(self, monkeypatch):
        from tools.update_agent import interpret_update
        mock = MagicMock()
        mock.content = "not json at all"
        monkeypatch.setattr("tools.update_agent.get_model", lambda **kw: MagicMock(
            invoke=lambda msgs: mock
        ))
        result = interpret_update(SAMPLE_ITEM, "update something")
        assert "_error" in result

    def test_markdown_fence_stripped(self, monkeypatch):
        from tools.update_agent import interpret_update
        mock = MagicMock()
        mock.content = "```json\n{\"urgency\": 0.8}\n```"
        monkeypatch.setattr("tools.update_agent.get_model", lambda **kw: MagicMock(
            invoke=lambda msgs: mock
        ))
        result = interpret_update(SAMPLE_ITEM, "raise urgency")
        assert result == {"urgency": 0.8}

    def test_days_since_update_clamped(self, monkeypatch):
        from tools.update_agent import interpret_update
        monkeypatch.setattr("tools.update_agent.get_model", lambda **kw: MagicMock(
            invoke=lambda msgs: _mock_llm_response({"days_since_update": -5})
        ))
        result = interpret_update(SAMPLE_ITEM, "reset clock")
        assert result["days_since_update"] == 0


# ---------------------------------------------------------------------------
# apply_update
# ---------------------------------------------------------------------------

class TestApplyUpdate:

    def test_successful_apply(self, monkeypatch):
        from tools.update_agent import apply_update
        monkeypatch.setattr("tools.update_agent.get_model", lambda **kw: MagicMock(
            invoke=lambda msgs: _mock_llm_response({"urgency": 0.9})
        ))
        updated, changes = apply_update("PLB-001", SAMPLE_ITEM, "raise urgency to 0.9")
        assert changes == {"urgency": 0.9}
        assert updated is not None
        assert updated["urgency"] == 0.9

    def test_empty_changes_returns_none(self, monkeypatch):
        from tools.update_agent import apply_update
        monkeypatch.setattr("tools.update_agent.get_model", lambda **kw: MagicMock(
            invoke=lambda msgs: _mock_llm_response({})
        ))
        updated, changes = apply_update("PLM-001", SAMPLE_ITEM, "do nothing")
        assert updated is None
        assert changes == {}

    def test_error_returns_none(self, monkeypatch):
        from tools.update_agent import apply_update
        def bad_model(**kw):
            m = MagicMock()
            m.invoke.side_effect = Exception("fail")
            return m
        monkeypatch.setattr("tools.update_agent.get_model", bad_model)
        updated, changes = apply_update("PLM-001", SAMPLE_ITEM, "anything")
        assert updated is None
        assert "_error" in changes