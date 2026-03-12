"""
tests/test_completeness_agent.py  —  Unit tests for Completeness Scorer + Prompt Agent.

Covers:
  - Short / empty input guards
  - Category normalization (canonical, fuzzy, unknown fallback)
  - CATEGORY_RUBRICS structure (all five categories, required keys)
  - Happy path: score, missing_fields, questions returned correctly
  - Score clamped to [0.0, 1.0]
  - Score type coercion (string, missing key, null)
  - Missing fields and questions returned as lists of strings
  - Empty questions list when description is complete
  - Markdown fence stripping
  - Malformed / non-JSON LLM response handled gracefully
  - LLM exception handled gracefully (never raises)
  - Error field populated on failure
  - API key passed to ChatGroq
  - None api_key falls back to env var
  - System prompt contains category name and field names
  - _infer_category_from_description keyword routing
  - All five categories covered by keyword inference
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch

import tools.completeness_agent as ca_module
from tools.completeness_agent import (
    score_completeness,
    CompletenessResult,
    CATEGORY_RUBRICS,
    _normalize_category,
    _build_system_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_llm_response(score: float, missing_fields: list, questions: list) -> MagicMock:
    mock = MagicMock()
    mock.content = json.dumps({
        "score":          score,
        "missing_fields": missing_fields,
        "questions":      questions,
    })
    return mock


def make_raw_response(content: str) -> MagicMock:
    mock = MagicMock()
    mock.content = content
    return mock


def mock_groq(response: MagicMock):
    instance = MagicMock()
    instance.invoke.return_value = response
    return patch("tools.completeness_agent.ChatGroq", return_value=instance)


# ---------------------------------------------------------------------------
# Rubric structure
# ---------------------------------------------------------------------------

class TestRubricStructure:

    def test_all_five_categories_present(self):
        assert set(CATEGORY_RUBRICS.keys()) == {"hvac", "plumbing", "electrical", "appliance", "general"}

    def test_each_category_has_description(self):
        for cat, rubric in CATEGORY_RUBRICS.items():
            assert "description" in rubric, f"Missing 'description' in {cat}"
            assert rubric["description"].strip()

    def test_each_category_has_fields(self):
        for cat, rubric in CATEGORY_RUBRICS.items():
            assert "fields" in rubric, f"Missing 'fields' in {cat}"
            assert len(rubric["fields"]) >= 4, f"Too few fields in {cat}"

    def test_each_field_has_name_and_description(self):
        for cat, rubric in CATEGORY_RUBRICS.items():
            for field in rubric["fields"]:
                assert "name" in field,        f"Field missing 'name' in {cat}"
                assert "description" in field, f"Field missing 'description' in {cat}"

    def test_typedict_keys(self):
        keys = set(CompletenessResult.__annotations__.keys())
        assert keys == {"category", "score", "missing_fields", "questions", "error"}


# ---------------------------------------------------------------------------
# Category normalization
# ---------------------------------------------------------------------------

class TestCategoryNormalization:

    def test_canonical_hvac(self):
        assert _normalize_category("hvac") == "hvac"

    def test_canonical_plumbing(self):
        assert _normalize_category("plumbing") == "plumbing"

    def test_canonical_electrical(self):
        assert _normalize_category("electrical") == "electrical"

    def test_canonical_appliance(self):
        assert _normalize_category("appliance") == "appliance"

    def test_canonical_general(self):
        assert _normalize_category("general") == "general"

    def test_uppercase_normalized(self):
        assert _normalize_category("HVAC") == "hvac"

    def test_mixed_case_normalized(self):
        assert _normalize_category("Plumbing") == "plumbing"

    def test_fuzzy_partial_match(self):
        assert _normalize_category("hvac systems") == "hvac"

    def test_unknown_falls_back_to_general(self):
        assert _normalize_category("roofing") == "general"

    def test_empty_falls_back_to_general(self):
        assert _normalize_category("") == "general"

    def test_whitespace_stripped(self):
        assert _normalize_category("  electrical  ") == "electrical"


# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------

class TestSystemPromptBuilding:

    def test_prompt_contains_category_name(self):
        for cat in CATEGORY_RUBRICS:
            prompt = _build_system_prompt(cat)
            assert cat in prompt, f"Category '{cat}' missing from its own prompt"

    def test_prompt_contains_all_field_names(self):
        for cat, rubric in CATEGORY_RUBRICS.items():
            prompt = _build_system_prompt(cat)
            for field in rubric["fields"]:
                assert field["name"] in prompt, \
                    f"Field '{field['name']}' missing from {cat} prompt"

    def test_prompt_requires_json(self):
        for cat in CATEGORY_RUBRICS:
            prompt = _build_system_prompt(cat)
            assert "JSON" in prompt or "json" in prompt


# ---------------------------------------------------------------------------
# Input guards
# ---------------------------------------------------------------------------

class TestInputGuards:

    def test_empty_string_rejected(self):
        result = score_completeness("", "hvac")
        assert result["error"] is not None
        assert result["score"] == 0.0

    def test_whitespace_only_rejected(self):
        result = score_completeness("   ", "plumbing")
        assert result["error"] is not None

    def test_seven_chars_rejected(self):
        result = score_completeness("1234567", "electrical")
        assert result["error"] is not None

    def test_short_input_returns_empty_lists(self):
        result = score_completeness("short", "general")
        assert result["missing_fields"] == []
        assert result["questions"] == []

    def test_short_input_category_preserved(self):
        result = score_completeness("hi", "hvac")
        assert result["category"] == "hvac"


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

class TestHappyPaths:

    def test_score_returned(self):
        with mock_groq(make_llm_response(0.6, ["duration", "location"], ["How long?", "Which room?"])):
            result = score_completeness(
                "Plumbing issue in the bathroom", "plumbing", api_key="fake"
            )
        assert result["score"] == 0.6
        assert result["error"] is None

    def test_missing_fields_returned(self):
        with mock_groq(make_llm_response(0.4, ["duration", "water_damage", "severity"], ["Q1", "Q2", "Q3"])):
            result = score_completeness(
                "There is a leak somewhere", "plumbing", api_key="fake"
            )
        assert "duration" in result["missing_fields"]
        assert result["error"] is None

    def test_questions_returned(self):
        with mock_groq(make_llm_response(0.6, ["duration"], ["How long has this been occurring?"])):
            result = score_completeness(
                "Furnace not blowing warm air in living room", "hvac", api_key="fake"
            )
        assert len(result["questions"]) == 1
        assert "How long" in result["questions"][0]
        assert result["error"] is None

    def test_complete_description_returns_empty_questions(self):
        with mock_groq(make_llm_response(1.0, [], [])):
            result = score_completeness(
                "Furnace stopped blowing heat two days ago in living room, January, last serviced 6 months ago",
                "hvac", api_key="fake"
            )
        assert result["score"] == 1.0
        assert result["questions"] == []
        assert result["missing_fields"] == []
        assert result["error"] is None

    def test_category_normalized_in_result(self):
        with mock_groq(make_llm_response(0.8, ["age"], ["How old is the appliance?"])):
            result = score_completeness(
                "Dishwasher not draining after cycle completes", "APPLIANCE", api_key="fake"
            )
        assert result["category"] == "appliance"
        assert result["error"] is None

    def test_all_five_categories_run(self):
        for cat in ("hvac", "plumbing", "electrical", "appliance", "general"):
            with mock_groq(make_llm_response(0.6, ["duration"], ["How long?"])):
                result = score_completeness(
                    f"Some issue that is relevant to {cat} systems in the home",
                    cat, api_key="fake"
                )
            assert result["error"] is None, f"Failed for category: {cat}"
            assert result["category"] == cat


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------

class TestScoreNormalization:

    def test_score_above_one_clamped(self):
        with mock_groq(make_llm_response(1.5, [], [])):
            result = score_completeness(
                "Complete description of an issue with all relevant details included",
                "general", api_key="fake"
            )
        assert result["score"] == 1.0

    def test_score_below_zero_clamped(self):
        with mock_groq(make_llm_response(-0.2, ["symptom"], ["What is happening?"])):
            result = score_completeness(
                "Something is wrong with the house somewhere", "general", api_key="fake"
            )
        assert result["score"] == 0.0

    def test_score_string_coerced(self):
        mock_resp = make_raw_response(json.dumps({
            "score": "0.75",
            "missing_fields": ["duration"],
            "questions": ["How long has this been occurring?"],
        }))
        with mock_groq(mock_resp):
            result = score_completeness(
                "Electrical outlet in kitchen stopped working this week",
                "electrical", api_key="fake"
            )
        assert result["score"] == pytest.approx(0.75)
        assert result["error"] is None

    def test_score_missing_key_defaults_to_half(self):
        mock_resp = make_raw_response(json.dumps({
            "missing_fields": ["duration"],
            "questions": ["How long?"],
        }))
        with mock_groq(mock_resp):
            result = score_completeness(
                "Refrigerator making loud noise from back of unit",
                "appliance", api_key="fake"
            )
        assert result["score"] == pytest.approx(0.5)
        assert result["error"] is None

    def test_score_null_defaults_to_half(self):
        mock_resp = make_raw_response(json.dumps({
            "score": None,
            "missing_fields": [],
            "questions": [],
        }))
        with mock_groq(mock_resp):
            result = score_completeness(
                "Caulk gap around the bathtub is cracking and peeling away",
                "general", api_key="fake"
            )
        assert result["score"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Output list handling
# ---------------------------------------------------------------------------

class TestOutputLists:

    def test_missing_fields_is_list(self):
        with mock_groq(make_llm_response(0.4, ["duration", "location"], ["Q1", "Q2"])):
            result = score_completeness(
                "There is a plumbing issue somewhere in the house",
                "plumbing", api_key="fake"
            )
        assert isinstance(result["missing_fields"], list)

    def test_questions_is_list(self):
        with mock_groq(make_llm_response(0.4, ["duration", "location"], ["Q1", "Q2"])):
            result = score_completeness(
                "There is a plumbing issue somewhere in the house",
                "plumbing", api_key="fake"
            )
        assert isinstance(result["questions"], list)

    def test_missing_fields_empty_list_when_complete(self):
        with mock_groq(make_llm_response(1.0, [], [])):
            result = score_completeness(
                "HVAC not heating, furnace, living room, started 2 days ago, winter, last serviced 6 months ago",
                "hvac", api_key="fake"
            )
        assert result["missing_fields"] == []

    def test_list_items_cast_to_string(self):
        mock_resp = make_raw_response(json.dumps({
            "score": 0.4,
            "missing_fields": [1, 2],       # ints, not strings
            "questions": [True, "How long?"],
        }))
        with mock_groq(mock_resp):
            result = score_completeness(
                "Something is wrong with the electrical panel in basement",
                "electrical", api_key="fake"
            )
        for item in result["missing_fields"]:
            assert isinstance(item, str)
        for item in result["questions"]:
            assert isinstance(item, str)

    def test_missing_fields_key_absent_returns_empty_list(self):
        mock_resp = make_raw_response(json.dumps({
            "score": 0.6,
            "questions": ["How long has this been happening?"],
        }))
        with mock_groq(mock_resp):
            result = score_completeness(
                "Washer leaving water on the floor after each cycle run",
                "appliance", api_key="fake"
            )
        assert result["missing_fields"] == []
        assert result["error"] is None


# ---------------------------------------------------------------------------
# Markdown fence stripping
# ---------------------------------------------------------------------------

class TestMarkdownFenceStripping:

    def test_json_fence_stripped(self):
        payload = json.dumps({"score": 0.6, "missing_fields": ["duration"], "questions": ["How long?"]})
        with mock_groq(make_raw_response(f"```json\n{payload}\n```")):
            result = score_completeness(
                "Furnace is not producing heat in the main living area",
                "hvac", api_key="fake"
            )
        assert result["score"] == 0.6
        assert result["error"] is None

    def test_plain_fence_stripped(self):
        payload = json.dumps({"score": 0.8, "missing_fields": ["age"], "questions": ["How old?"]})
        with mock_groq(make_raw_response(f"```\n{payload}\n```")):
            result = score_completeness(
                "Refrigerator not cooling properly, making clicking noise hourly",
                "appliance", api_key="fake"
            )
        assert result["score"] == 0.8
        assert result["error"] is None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_malformed_json_returns_error(self):
        with mock_groq(make_raw_response("not json at all here")):
            result = score_completeness(
                "Some description that is long enough to pass guard",
                "general", api_key="fake"
            )
        assert result["error"] is not None
        assert result["score"] == 0.0

    def test_partial_json_returns_error(self):
        with mock_groq(make_raw_response('{"score": 0.5')):
            result = score_completeness(
                "Electrical issue in the kitchen outlet stopped working",
                "electrical", api_key="fake"
            )
        assert result["error"] is not None

    def test_llm_exception_returns_error(self):
        instance = MagicMock()
        instance.invoke.side_effect = Exception("Network timeout")
        with patch("tools.completeness_agent.ChatGroq", return_value=instance):
            result = score_completeness(
                "Water is dripping from the ceiling in the hallway",
                "plumbing", api_key="fake"
            )
        assert result["error"] is not None
        assert "Network timeout" in result["error"]

    def test_exception_never_raises(self):
        instance = MagicMock()
        instance.invoke.side_effect = RuntimeError("Catastrophic failure")
        with patch("tools.completeness_agent.ChatGroq", return_value=instance):
            try:
                result = score_completeness(
                    "Something that causes a catastrophic failure in the model",
                    "hvac", api_key="fake"
                )
            except Exception as e:
                pytest.fail(f"score_completeness raised an exception: {e}")
        assert result["error"] is not None

    def test_error_includes_exception_type(self):
        instance = MagicMock()
        instance.invoke.side_effect = ValueError("Bad response value")
        with patch("tools.completeness_agent.ChatGroq", return_value=instance):
            result = score_completeness(
                "Plumbing issue with the main water supply line here",
                "plumbing", api_key="fake"
            )
        assert "ValueError" in result["error"]

    def test_error_returns_empty_lists(self):
        instance = MagicMock()
        instance.invoke.side_effect = Exception("Failure")
        with patch("tools.completeness_agent.ChatGroq", return_value=instance):
            result = score_completeness(
                "Some valid description about an issue in the home",
                "general", api_key="fake"
            )
        assert result["missing_fields"] == []
        assert result["questions"] == []


# ---------------------------------------------------------------------------
# API key handling
# ---------------------------------------------------------------------------

class TestAPIKeyHandling:

    def test_api_key_passed_to_chatgroq(self):
        with patch("tools.completeness_agent.ChatGroq") as MockGroq:
            instance = MockGroq.return_value
            instance.invoke.return_value = make_llm_response(0.8, [], [])
            score_completeness(
                "Furnace not heating, has been two days, living room, January",
                "hvac", api_key="test_key_789"
            )
        assert MockGroq.call_args.kwargs.get("api_key") == "test_key_789"

    def test_none_api_key_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "env_key_abc")
        with patch("tools.completeness_agent.ChatGroq") as MockGroq:
            instance = MockGroq.return_value
            instance.invoke.return_value = make_llm_response(0.6, ["duration"], ["How long?"])
            score_completeness(
                "There is a leak under the kitchen sink running slowly",
                "plumbing", api_key=None
            )
        assert MockGroq.call_args.kwargs.get("api_key") == "env_key_abc"


# ---------------------------------------------------------------------------
# _infer_category_from_description (app.py helper — tested via logic parity)
# ---------------------------------------------------------------------------

class TestCategoryInference:
    """
    Tests the keyword-based category inference logic defined inline in app.py.
    Duplicated here as a standalone function for isolated testing.
    """

    @staticmethod
    def _infer(desc: str) -> str:
        d = desc.lower()
        if any(k in d for k in ("washer", "dryer", "dishwasher", "refrigerator", "fridge", "oven", "stove", "microwave", "appliance")):
            return "appliance"
        if any(k in d for k in ("furnace", "hvac", "ac ", "air condition", "vent", "duct", "filter", "thermostat", "cooling", "heating", "heat pump")):
            return "hvac"
        if "heat" in d and not any(k in d for k in ("washer", "dryer", "dishwasher", "refrigerator", "fridge", "oven", "stove", "microwave")):
            return "hvac"
        if any(k in d for k in ("pipe", "drain", "leak", "plumb", "toilet", "faucet", "sink", "water heater", "shower", "hose", "clog", "flood")):
            return "plumbing"
        if any(k in d for k in ("outlet", "circuit", "breaker", "electrical", "wiring", "panel", "gfci", "light", "switch", "spark", "power")):
            return "electrical"
        return "general"

    def test_furnace_maps_to_hvac(self):
        assert self._infer("furnace making grinding noise") == "hvac"

    def test_thermostat_maps_to_hvac(self):
        assert self._infer("thermostat not responding to changes") == "hvac"

    def test_leak_maps_to_plumbing(self):
        assert self._infer("there is a leak under the sink") == "plumbing"

    def test_toilet_maps_to_plumbing(self):
        assert self._infer("toilet keeps running after flush") == "plumbing"

    def test_outlet_maps_to_electrical(self):
        assert self._infer("outlet in kitchen stopped working") == "electrical"

    def test_breaker_maps_to_electrical(self):
        assert self._infer("circuit breaker keeps tripping") == "electrical"

    def test_dryer_maps_to_appliance(self):
        assert self._infer("dryer not heating up after cycle starts") == "appliance"

    def test_refrigerator_maps_to_appliance(self):
        assert self._infer("refrigerator not staying cold enough") == "appliance"

    def test_unknown_maps_to_general(self):
        assert self._infer("something looks off near the back fence") == "general"

    def test_empty_maps_to_general(self):
        assert self._infer("") == "general"