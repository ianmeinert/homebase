"""
conftest.py -- Pytest configuration for homebase test suite.

Patches the LLM model constructor globally so tests never require
a real API key. All LLM-dependent tests use this mock automatically.
"""

import pytest
from unittest.mock import MagicMock, patch


class MockLLMResponse:
    """Minimal response object matching LangChain message interface."""
    def __init__(self, content: str):
        self.content = content


class MockModel:
    """Stub model that returns deterministic JSON without hitting any API."""

    def invoke(self, messages):
        # Return a minimal valid JSON recommendation array
        # Tests that care about specific content use their own patches
        return MockLLMResponse(
            '[{"item_id": "TEST-001", "action": "Inspect and address.", '
            '"estimated_effort": "1 hr", "estimated_cost": "$50-100", '
            '"priority_note": "Address promptly.", "agent": "TestAgent"}]'
        )


@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    """
    Auto-used fixture -- patches get_model() in llm_tools and
    _llm_synthesize in orchestrator for every test in the suite.
    Prevents ValidationError from ChatGroq requiring
    an API key at instantiation time.
    """
    mock_model = MockModel()

    # Patch model constructor in llm_tools
    monkeypatch.setattr("tools.llm_tools.get_model", lambda: mock_model)

    # Patch synthesis LLM call in orchestrator
    def mock_synthesize(active_results, trigger, hitl_notes):
        ids = [r["item"]["id"] for r in active_results]
        return f"Test synthesis for '{trigger}'. Items: {', '.join(ids)}" if ids else "No active items."

    monkeypatch.setattr("agents.orchestrator._llm_synthesize", mock_synthesize)