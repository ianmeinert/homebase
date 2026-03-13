"""
test_llm_providers.py  -  Tests for multi-provider LLM abstraction.

Coverage:
- active_provider() detection logic (env var, arg, neither)
- get_synthesizer_model() returns correct type per provider
- get_subagent_model() always returns ChatGroq
- provider_meta() returns correct label/color per provider
- Key resolution helpers (env var vs arg priority)
"""

import os
import pytest
from unittest.mock import patch

from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

from tools.llm_providers import (
    active_provider,
    get_synthesizer_model,
    get_subagent_model,
    is_claude_active,
    provider_meta,
    CLAUDE_MODEL,
    GROQ_MODEL,
    PROVIDER_META,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """Ensure ANTHROPIC_API_KEY and GROQ_API_KEY are not set between tests."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# active_provider()
# ---------------------------------------------------------------------------

class TestActiveProvider:
    def test_returns_groq_with_no_key(self):
        assert active_provider() == "groq"

    def test_returns_groq_with_empty_string(self):
        assert active_provider(anthropic_key="") == "groq"

    def test_returns_groq_with_whitespace_only(self):
        assert active_provider(anthropic_key="   ") == "groq"

    def test_returns_claude_with_direct_key(self):
        assert active_provider(anthropic_key="sk-ant-test123") == "claude"

    def test_returns_claude_with_env_var(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env456")
        assert active_provider() == "claude"

    def test_direct_key_takes_precedence_over_no_env(self):
        # Explicit arg wins when env is absent
        assert active_provider(anthropic_key="sk-ant-arg") == "claude"

    def test_env_var_used_when_no_arg(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
        assert active_provider(anthropic_key=None) == "claude"


# ---------------------------------------------------------------------------
# is_claude_active()
# ---------------------------------------------------------------------------

class TestIsClaudeActive:
    def test_false_without_key(self):
        assert is_claude_active() is False

    def test_true_with_key(self):
        assert is_claude_active(anthropic_key="sk-ant-x") is True

    def test_true_with_env_var(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
        assert is_claude_active() is True


# ---------------------------------------------------------------------------
# get_synthesizer_model()
# ---------------------------------------------------------------------------

class TestGetSynthesizerModel:
    def test_returns_groq_without_anthropic_key(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        model = get_synthesizer_model()
        assert isinstance(model, ChatGroq)

    def test_returns_groq_with_empty_anthropic_key(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        model = get_synthesizer_model(anthropic_api_key="")
        assert isinstance(model, ChatGroq)

    def test_returns_anthropic_with_direct_key(self):
        model = get_synthesizer_model(anthropic_api_key="sk-ant-test")
        assert isinstance(model, ChatAnthropic)

    def test_returns_anthropic_with_env_var(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
        model = get_synthesizer_model()
        assert isinstance(model, ChatAnthropic)

    def test_anthropic_model_name_is_correct(self):
        model = get_synthesizer_model(anthropic_api_key="sk-ant-test")
        assert model.model == CLAUDE_MODEL

    def test_groq_model_name_is_correct(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        model = get_synthesizer_model()
        assert model.model_name == GROQ_MODEL

    def test_groq_passed_key_used(self):
        model = get_synthesizer_model(groq_api_key="gsk_direct")
        assert isinstance(model, ChatGroq)

    def test_anthropic_key_overrides_groq(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_env")
        model = get_synthesizer_model(
            groq_api_key="gsk_direct",
            anthropic_api_key="sk-ant-wins",
        )
        assert isinstance(model, ChatAnthropic)


# ---------------------------------------------------------------------------
# get_subagent_model()
# ---------------------------------------------------------------------------

class TestGetSubagentModel:
    def test_always_returns_groq(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        model = get_subagent_model()
        assert isinstance(model, ChatGroq)

    def test_always_groq_even_with_anthropic_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        model = get_subagent_model()
        assert isinstance(model, ChatGroq)

    def test_groq_model_name(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        model = get_subagent_model()
        assert model.model_name == GROQ_MODEL

    def test_direct_key_used(self):
        model = get_subagent_model(groq_api_key="gsk_direct")
        assert isinstance(model, ChatGroq)


# ---------------------------------------------------------------------------
# provider_meta()
# ---------------------------------------------------------------------------

class TestProviderMeta:
    def test_groq_meta_without_key(self):
        meta = provider_meta()
        assert meta["vendor"] == "Groq"
        assert meta["model"] == GROQ_MODEL
        assert meta["color"] == "#56d364"

    def test_claude_meta_with_key(self):
        meta = provider_meta(anthropic_key="sk-ant-test")
        assert meta["vendor"] == "Anthropic"
        assert meta["model"] == CLAUDE_MODEL
        assert meta["color"] == "#d2a8ff"

    def test_groq_label(self):
        meta = provider_meta()
        assert "Llama" in meta["label"]

    def test_claude_label(self):
        meta = provider_meta(anthropic_key="sk-ant-x")
        assert "Claude" in meta["label"]

    def test_provider_meta_constant_keys(self):
        for key in ("label", "model", "vendor", "color"):
            assert key in PROVIDER_META["groq"]
            assert key in PROVIDER_META["claude"]


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

class TestModelConstants:
    def test_groq_model_is_llama(self):
        assert "llama" in GROQ_MODEL.lower()

    def test_claude_model_is_sonnet(self):
        assert "claude" in CLAUDE_MODEL.lower()
        assert "sonnet" in CLAUDE_MODEL.lower()