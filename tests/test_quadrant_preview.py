"""
tests/test_quadrant_preview.py  —  Unit tests for Predictive Quadrant Preview.

Covers:
  - Short / empty input guards (< 8 chars)
  - Length boundary (exactly 7 chars rejected, 8+ accepted)
  - Happy path: all four valid quadrants returned correctly
  - Confidence clamped to [0.0, 1.0]
  - Confidence type coercion (string, missing key)
  - Invalid quadrant value caught and returned as error
  - Markdown fence stripping (```json ... ```)
  - Malformed / non-JSON LLM response handled gracefully
  - LLM exception handled gracefully (never raises)
  - Rationale field passed through correctly
  - Empty rationale handled
  - Model instantiated with provided api_key
  - Model falls back to env var when api_key is None
  - _VALID_QUADRANTS constant is complete and correct
  - TypedDict structure (all four keys present)
  - System prompt contains all four quadrant definitions
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch, call

import tools.quadrant_preview as qp_module
from tools.quadrant_preview import predict_quadrant, _VALID_QUADRANTS, _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_llm_response(quadrant: str, confidence: float, rationale: str) -> MagicMock:
    """Build a mock LangChain response with well-formed JSON content."""
    mock = MagicMock()
    mock.content = json.dumps({
        "quadrant": quadrant,
        "confidence": confidence,
        "rationale": rationale,
    })
    return mock


def make_raw_response(content: str) -> MagicMock:
    """Build a mock response with arbitrary raw content."""
    mock = MagicMock()
    mock.content = content
    return mock


def mock_groq(response: MagicMock):
    """Context manager: patches ChatGroq so .invoke() returns the given response."""
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = response
    return patch("tools.quadrant_preview.ChatGroq", return_value=mock_instance)


# ---------------------------------------------------------------------------
# Constants and structure
# ---------------------------------------------------------------------------

class TestConstants:

    def test_valid_quadrants_complete(self):
        assert _VALID_QUADRANTS == {"HU/HI", "HU/LI", "LU/HI", "LU/LI"}

    def test_valid_quadrants_count(self):
        assert len(_VALID_QUADRANTS) == 4

    def test_system_prompt_contains_all_quadrants(self):
        for q in ("HU/HI", "HU/LI", "LU/HI", "LU/LI"):
            assert q in _SYSTEM_PROMPT, f"System prompt missing quadrant definition: {q}"

    def test_system_prompt_requires_json_only(self):
        # Prompt must instruct the model to return only JSON
        assert "JSON" in _SYSTEM_PROMPT or "json" in _SYSTEM_PROMPT

    def test_typedict_keys(self):
        keys = set(qp_module.QuadrantPreview.__annotations__.keys())
        assert keys == {"quadrant", "confidence", "rationale", "error"}


# ---------------------------------------------------------------------------
# Input guards
# ---------------------------------------------------------------------------

class TestInputGuards:

    def test_empty_string_rejected(self):
        result = predict_quadrant("")
        assert result["error"] is not None
        assert result["quadrant"] == ""
        assert result["confidence"] == 0.0

    def test_whitespace_only_rejected(self):
        result = predict_quadrant("   ")
        assert result["error"] is not None

    def test_six_chars_rejected(self):
        result = predict_quadrant("abcdef")
        assert result["error"] is not None

    def test_seven_chars_rejected(self):
        # Boundary: len < 8 means 7 chars must fail
        result = predict_quadrant("1234567")
        assert result["error"] is not None

    def test_eight_chars_accepted(self):
        # len >= 8 should pass the guard (may still fail on LLM call without key)
        result = predict_quadrant("12345678")
        # Error may occur from missing API key, but not the short-input guard
        if result["error"]:
            assert "Description too short" not in result["error"]

    def test_short_input_returns_empty_quadrant(self):
        result = predict_quadrant("hi")
        assert result["quadrant"] == ""

    def test_short_input_returns_zero_confidence(self):
        result = predict_quadrant("hi")
        assert result["confidence"] == 0.0

    def test_short_input_returns_empty_rationale(self):
        result = predict_quadrant("hi")
        assert result["rationale"] == ""


# ---------------------------------------------------------------------------
# Happy paths — all four quadrants
# ---------------------------------------------------------------------------

class TestHappyPaths:

    def test_hu_hi_returned(self):
        with mock_groq(make_llm_response("HU/HI", 0.91, "Active safety risk in winter.")):
            result = predict_quadrant(
                "Furnace making grinding noise and it is January", api_key="fake"
            )
        assert result["quadrant"] == "HU/HI"
        assert result["error"] is None

    def test_hu_li_returned(self):
        with mock_groq(make_llm_response("HU/LI", 0.74, "Annoying but no structural risk.")):
            result = predict_quadrant(
                "Front door handle broke and won't latch properly", api_key="fake"
            )
        assert result["quadrant"] == "HU/LI"
        assert result["error"] is None

    def test_lu_hi_returned(self):
        with mock_groq(make_llm_response("LU/HI", 0.68, "Long-term consequence if ignored.")):
            result = predict_quadrant(
                "Roof shingles starting to curl but no active leak", api_key="fake"
            )
        assert result["quadrant"] == "LU/HI"
        assert result["error"] is None

    def test_lu_li_returned(self):
        with mock_groq(make_llm_response("LU/LI", 0.82, "Cosmetic issue only.")):
            result = predict_quadrant(
                "Touch up paint needed on the back fence", api_key="fake"
            )
        assert result["quadrant"] == "LU/LI"
        assert result["error"] is None

    def test_confidence_passed_through(self):
        with mock_groq(make_llm_response("HU/HI", 0.91, "Active risk.")):
            result = predict_quadrant(
                "No hot water and pipes may be frozen outside", api_key="fake"
            )
        assert result["confidence"] == 0.91

    def test_rationale_passed_through(self):
        with mock_groq(make_llm_response("LU/LI", 0.78, "Minor cosmetic gap only.")):
            result = predict_quadrant(
                "Small caulk gap near the bathroom window sill", api_key="fake"
            )
        assert result["rationale"] == "Minor cosmetic gap only."

    def test_error_is_none_on_success(self):
        with mock_groq(make_llm_response("LU/HI", 0.65, "Plan and schedule.")):
            result = predict_quadrant(
                "HVAC unit is aging and efficiency is dropping", api_key="fake"
            )
        assert result["error"] is None


# ---------------------------------------------------------------------------
# Confidence normalization
# ---------------------------------------------------------------------------

class TestConfidenceNormalization:

    def test_confidence_above_one_clamped_to_one(self):
        with mock_groq(make_llm_response("HU/HI", 1.5, "Over-confident.")):
            result = predict_quadrant(
                "Electrical panel sparking and breaker keeps tripping", api_key="fake"
            )
        assert result["confidence"] == 1.0

    def test_confidence_below_zero_clamped_to_zero(self):
        with mock_groq(make_llm_response("LU/LI", -0.3, "Under-confident.")):
            result = predict_quadrant(
                "Minor scuff on baseboard near the hallway door", api_key="fake"
            )
        assert result["confidence"] == 0.0

    def test_confidence_exactly_zero_preserved(self):
        with mock_groq(make_llm_response("LU/LI", 0.0, "No confidence.")):
            result = predict_quadrant(
                "Something very minor and routine around the house", api_key="fake"
            )
        assert result["confidence"] == 0.0

    def test_confidence_exactly_one_preserved(self):
        with mock_groq(make_llm_response("HU/HI", 1.0, "Maximum confidence.")):
            result = predict_quadrant(
                "Gas leak smell inside the house right now urgent", api_key="fake"
            )
        assert result["confidence"] == 1.0

    def test_confidence_string_coerced_to_float(self):
        mock_resp = make_raw_response(json.dumps({
            "quadrant": "HU/HI",
            "confidence": "0.85",
            "rationale": "String confidence coerced.",
        }))
        with mock_groq(mock_resp):
            result = predict_quadrant(
                "Water is actively dripping from the ceiling above", api_key="fake"
            )
        assert result["confidence"] == pytest.approx(0.85)
        assert result["error"] is None

    def test_confidence_missing_key_defaults_to_half(self):
        mock_resp = make_raw_response(json.dumps({
            "quadrant": "LU/HI",
            "rationale": "No confidence key present.",
        }))
        with mock_groq(mock_resp):
            result = predict_quadrant(
                "Foundation has a small hairline crack in the basement", api_key="fake"
            )
        assert result["confidence"] == pytest.approx(0.5)
        assert result["error"] is None

    def test_confidence_null_defaults_to_half(self):
        mock_resp = make_raw_response(json.dumps({
            "quadrant": "LU/LI",
            "confidence": None,
            "rationale": "Null confidence.",
        }))
        with mock_groq(mock_resp):
            result = predict_quadrant(
                "Light switch cover plate is slightly cracked here", api_key="fake"
            )
        assert result["confidence"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Error cases — invalid LLM output
# ---------------------------------------------------------------------------

class TestInvalidLLMOutput:

    def test_invalid_quadrant_returns_error(self):
        with mock_groq(make_llm_response("MEDIUM", 0.5, "Bad output.")):
            result = predict_quadrant(
                "Something that is somewhat urgent and moderately important", api_key="fake"
            )
        assert result["error"] is not None
        assert result["quadrant"] == ""

    def test_empty_quadrant_string_returns_error(self):
        with mock_groq(make_llm_response("", 0.5, "Empty quadrant.")):
            result = predict_quadrant(
                "Something that needs attention at some point soon", api_key="fake"
            )
        assert result["error"] is not None

    def test_lowercase_quadrant_returns_error(self):
        # Quadrant must be uppercase; lowercase should fail validation
        with mock_groq(make_llm_response("hu/hi", 0.9, "Lowercase variant.")):
            result = predict_quadrant(
                "Broken furnace in the dead of winter emergency situation", api_key="fake"
            )
        assert result["error"] is not None

    def test_malformed_json_returns_error(self):
        with mock_groq(make_raw_response("not json at all here")):
            result = predict_quadrant(
                "Some description that is long enough to pass guard", api_key="fake"
            )
        assert result["error"] is not None
        assert result["quadrant"] == ""

    def test_partial_json_returns_error(self):
        with mock_groq(make_raw_response('{"quadrant": "HU/HI"')):  # unclosed brace
            result = predict_quadrant(
                "Partial JSON response from the language model here", api_key="fake"
            )
        assert result["error"] is not None

    def test_json_array_instead_of_object_returns_error(self):
        with mock_groq(make_raw_response('[{"quadrant": "HU/HI", "confidence": 0.9}]')):
            result = predict_quadrant(
                "Array response instead of object from the model", api_key="fake"
            )
        # JSON array has no .get() — will raise AttributeError caught as error
        assert result["error"] is not None

    def test_plain_text_response_returns_error(self):
        with mock_groq(make_raw_response("This issue is high urgency and high impact.")):
            result = predict_quadrant(
                "Flooding in the basement from a burst pipe right now", api_key="fake"
            )
        assert result["error"] is not None


# ---------------------------------------------------------------------------
# Markdown fence stripping
# ---------------------------------------------------------------------------

class TestMarkdownFenceStripping:

    def test_json_fence_stripped(self):
        payload = json.dumps({"quadrant": "LU/HI", "confidence": 0.72, "rationale": "Fence stripped."})
        with mock_groq(make_raw_response(f"```json\n{payload}\n```")):
            result = predict_quadrant(
                "Roof is aging visibly but not leaking yet this year", api_key="fake"
            )
        assert result["quadrant"] == "LU/HI"
        assert result["error"] is None

    def test_plain_fence_stripped(self):
        payload = json.dumps({"quadrant": "HU/LI", "confidence": 0.65, "rationale": "Plain fence."})
        with mock_groq(make_raw_response(f"```\n{payload}\n```")):
            result = predict_quadrant(
                "Garage door opener remote stopped working this week", api_key="fake"
            )
        assert result["quadrant"] == "HU/LI"
        assert result["error"] is None

    def test_no_fence_still_works(self):
        payload = json.dumps({"quadrant": "LU/LI", "confidence": 0.88, "rationale": "No fence needed."})
        with mock_groq(make_raw_response(payload)):
            result = predict_quadrant(
                "Minor scuff marks on the baseboard paint in hallway", api_key="fake"
            )
        assert result["quadrant"] == "LU/LI"
        assert result["error"] is None


# ---------------------------------------------------------------------------
# LLM exception handling
# ---------------------------------------------------------------------------

class TestLLMExceptionHandling:

    def test_network_exception_returns_error(self):
        mock_instance = MagicMock()
        mock_instance.invoke.side_effect = Exception("Network timeout")
        with patch("tools.quadrant_preview.ChatGroq", return_value=mock_instance):
            result = predict_quadrant(
                "Roof is aging and shingles are starting to curl up", api_key="fake"
            )
        assert result["error"] is not None
        assert "Network timeout" in result["error"]
        assert result["quadrant"] == ""
        assert result["confidence"] == 0.0

    def test_auth_exception_returns_error(self):
        mock_instance = MagicMock()
        mock_instance.invoke.side_effect = Exception("401 Unauthorized")
        with patch("tools.quadrant_preview.ChatGroq", return_value=mock_instance):
            result = predict_quadrant(
                "Water heater making popping sounds when heating up", api_key="bad_key"
            )
        assert result["error"] is not None
        assert result["quadrant"] == ""

    def test_exception_never_raises(self):
        """predict_quadrant must never propagate exceptions to the caller."""
        mock_instance = MagicMock()
        mock_instance.invoke.side_effect = RuntimeError("Catastrophic failure")
        with patch("tools.quadrant_preview.ChatGroq", return_value=mock_instance):
            try:
                result = predict_quadrant(
                    "Something that causes a catastrophic model failure here", api_key="fake"
                )
            except Exception as e:
                pytest.fail(f"predict_quadrant raised an exception: {e}")
        assert result["error"] is not None

    def test_error_field_includes_exception_type(self):
        mock_instance = MagicMock()
        mock_instance.invoke.side_effect = ValueError("Bad value in response")
        with patch("tools.quadrant_preview.ChatGroq", return_value=mock_instance):
            result = predict_quadrant(
                "Something that produces a bad value from the model", api_key="fake"
            )
        assert "ValueError" in result["error"]


# ---------------------------------------------------------------------------
# API key handling
# ---------------------------------------------------------------------------

class TestAPIKeyHandling:

    def test_api_key_passed_to_chatgroq(self):
        with patch("tools.quadrant_preview.ChatGroq") as MockGroq:
            instance = MockGroq.return_value
            instance.invoke.return_value = make_llm_response("HU/HI", 0.9, "Test.")
            predict_quadrant(
                "Furnace is completely dead and it is freezing outside", api_key="test_key_123"
            )
        call_kwargs = MockGroq.call_args
        assert call_kwargs is not None
        # api_key should be passed as a kwarg
        assert call_kwargs.kwargs.get("api_key") == "test_key_123"

    def test_none_api_key_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "env_key_456")
        with patch("tools.quadrant_preview.ChatGroq") as MockGroq:
            instance = MockGroq.return_value
            instance.invoke.return_value = make_llm_response("LU/LI", 0.5, "Env key used.")
            predict_quadrant(
                "Touch up paint on the fence boards at the back", api_key=None
            )
        call_kwargs = MockGroq.call_args
        assert call_kwargs.kwargs.get("api_key") == "env_key_456"


# ---------------------------------------------------------------------------
# Dedup / idempotency (pure logic — no LLM call needed)
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_leading_trailing_whitespace_stripped(self):
        """Whitespace around a valid description should not cause a guard rejection."""
        with patch("tools.quadrant_preview.ChatGroq") as MockGroq:
            instance = MockGroq.return_value
            instance.invoke.return_value = make_llm_response("HU/HI", 0.88, "Stripped.")
            result = predict_quadrant(
                "   Furnace stopped working and it is very cold outside   ",
                api_key="fake"
            )
        # Should not return the short-input error
        if result["error"]:
            assert "Description too short" not in result["error"]

    def test_all_four_quadrants_valid(self):
        """Each quadrant value in _VALID_QUADRANTS should pass the guard check."""
        for q in _VALID_QUADRANTS:
            with mock_groq(make_llm_response(q, 0.75, f"Testing {q}.")):
                result = predict_quadrant(
                    f"This is a test description for quadrant {q} classification check",
                    api_key="fake"
                )
            assert result["quadrant"] == q, f"Failed for quadrant: {q}"
            assert result["error"] is None

    def test_empty_rationale_handled(self):
        mock_resp = make_raw_response(json.dumps({
            "quadrant": "LU/LI",
            "confidence": 0.6,
            "rationale": "",
        }))
        with mock_groq(mock_resp):
            result = predict_quadrant(
                "Small cosmetic scratch on the baseboard near hallway", api_key="fake"
            )
        assert result["rationale"] == ""
        assert result["error"] is None

    def test_missing_rationale_key_handled(self):
        mock_resp = make_raw_response(json.dumps({
            "quadrant": "LU/HI",
            "confidence": 0.7,
        }))
        with mock_groq(mock_resp):
            result = predict_quadrant(
                "Foundation crack detected in the basement wall corner", api_key="fake"
            )
        assert result["rationale"] == ""
        assert result["error"] is None