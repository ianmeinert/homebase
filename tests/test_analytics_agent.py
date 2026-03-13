"""
tests/test_analytics_agent.py  --  Unit tests for Spreadsheet Analytics Agent.

Covers:
  load_file()
    - CSV, XLSX, ODS dispatch
    - Unsupported extension raises ValueError
    - Empty bytes handling

  profile_dataframe()
    - Numeric col detection and stats
    - Datetime col detection and date range extraction
    - Categorical col detection and top-5 value counts
    - Text col detection
    - Null percentage calculation
    - Truncation at ROW_LIMIT rows (truncated=True, original_rows set)
    - No truncation under limit (truncated=False)
    - Empty DataFrame guard
    - Mixed type column handling
    - Sample values capped at 5

  analyze_spreadsheet()
    - Valid LLM response → AnalyticsResult populated
    - Findings clamped to 8 max
    - Confidence clamped to [0.0, 1.0]
    - Markdown fence stripping
    - Invalid JSON → error field set, findings=[]
    - Empty profile guard
    - LLM exception handled gracefully
    - trend normalization (unexpected → "unknown")
    - severity normalization (unexpected → "info")
    - API key passed to genai.Client

  correlate_findings()
    - Valid LLM response → correlations merged into result
    - Empty registry → correlations=[], no LLM call
    - Empty findings → correlations=[], no LLM call
    - No-match response → correlations=[]
    - Invalid item_id filtered out
    - Result merging preserves findings/narrative
    - Invalid JSON → correlations=[], error appended
    - LLM exception → correlations=[], error appended
"""

import io
import json
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

import tools.analytics_agent as aa_module
from tools.analytics_agent import (
    load_file,
    profile_dataframe,
    analyze_spreadsheet,
    correlate_findings,
    ROW_LIMIT,
    AnalyticsResult,
    DataProfile,
    MetricFinding,
    CorrelationMatch,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

SAMPLE_REGISTRY = [
    {"id": "HV-001", "category": "hvac",      "title": "HVAC filter replacement",    "status": "open",  "description": ""},
    {"id": "APP-001","category": "appliance",  "title": "Dishwasher not draining",    "status": "open",  "description": ""},
    {"id": "PLB-001","category": "plumbing",   "title": "Water heater inspection",    "status": "open",  "description": ""},
    {"id": "EL-001", "category": "electrical", "title": "GFCI outlet replacement",    "status": "open",  "description": ""},
]

SAMPLE_DF = pd.DataFrame({
    "date":     pd.to_datetime(["2023-01-01", "2023-06-15", "2024-01-10"]),
    "category": ["hvac", "plumbing", "hvac"],
    "cost":     [150.0, 95.0, 420.0],
    "vendor":   ["AirPro", "Roto-Rooter", "AirPro"],
    "notes":    ["Annual inspection", "Drain snake", "Emergency repair"],
})


def _make_analyze_payload(
    findings=None,
    narrative="Test narrative.",
    confidence=0.8,
):
    if findings is None:
        findings = [
            {"metric": "Total spend", "value": "$665", "trend": "increasing",
             "insight": "Costs rose YoY.", "severity": "warning"},
            {"metric": "HVAC dominance", "value": "85%", "trend": "stable",
             "insight": "HVAC is primary cost driver.", "severity": "info"},
            {"metric": "Vendor concentration", "value": "AirPro 67%", "trend": "stable",
             "insight": "Single vendor dependency.", "severity": "critical"},
        ]
    return json.dumps({"findings": findings, "narrative": narrative, "confidence": confidence})


def _make_correlate_payload(correlations=None):
    if correlations is None:
        correlations = [
            {
                "item_id": "HV-001",
                "item_title": "HVAC filter replacement",
                "relevance": "HVAC accounts for majority of logged spend.",
                "proposed_note": "Analytics log shows $570 HVAC spend across 2 calls.",
            }
        ]
    return json.dumps({"correlations": correlations})


def _mock_client(payload: str):
    """Return a mock genai.Client whose generate_content returns payload as .text."""
    mock_response = MagicMock()
    mock_response.text = payload
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    return mock_client


# ---------------------------------------------------------------------------
# load_file() tests
# ---------------------------------------------------------------------------

class TestLoadFile:

    def test_csv_parse(self):
        csv_bytes = b"col1,col2\n1,a\n2,b\n"
        df = load_file(csv_bytes, "test.csv")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["col1", "col2"]
        assert len(df) == 2

    def test_csv_case_insensitive_extension(self):
        csv_bytes = b"x,y\n1,2\n"
        df = load_file(csv_bytes, "data.CSV")
        assert len(df) == 1

    def test_xlsx_parse(self):
        buf = io.BytesIO()
        pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_excel(buf, index=False, engine="openpyxl")
        df = load_file(buf.getvalue(), "data.xlsx")
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2

    def test_ods_parse(self):
        buf = io.BytesIO()
        pd.DataFrame({"m": [10, 20]}).to_excel(buf, index=False, engine="odf")
        df = load_file(buf.getvalue(), "data.ods")
        assert "m" in df.columns

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_file(b"data", "report.pdf")

    def test_unsupported_no_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_file(b"data", "noextension")

    def test_xls_extension_accepted(self):
        # .xls dispatches to openpyxl — should not raise ValueError
        buf = io.BytesIO()
        pd.DataFrame({"z": [1]}).to_excel(buf, index=False, engine="openpyxl")
        df = load_file(buf.getvalue(), "old.xls")
        assert "z" in df.columns


# ---------------------------------------------------------------------------
# profile_dataframe() tests
# ---------------------------------------------------------------------------

class TestProfileDataframe:

    def test_empty_dataframe_returns_zeroed_profile(self):
        profile = profile_dataframe(pd.DataFrame())
        assert profile["row_count"] == 0
        assert profile["col_count"] == 0
        assert profile["truncated"] is False
        assert profile["columns"] == []

    def test_none_input_returns_zeroed_profile(self):
        profile = profile_dataframe(None)
        assert profile["row_count"] == 0

    def test_numeric_col_detected(self):
        df = pd.DataFrame({"cost": [10.0, 20.0, 30.0]})
        profile = profile_dataframe(df)
        assert "cost" in profile["numeric_cols"]

    def test_numeric_stats_computed(self):
        df = pd.DataFrame({"cost": [10.0, 20.0, 30.0]})
        profile = profile_dataframe(df)
        stats = profile["numeric_stats"]["cost"]
        assert stats["min"] == 10.0
        assert stats["max"] == 30.0
        assert abs(stats["mean"] - 20.0) < 0.01
        assert abs(stats["sum"] - 60.0) < 0.01

    def test_datetime_col_detected(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2023-01-01", "2024-06-15"])})
        profile = profile_dataframe(df)
        assert "date" in profile["date_cols"]

    def test_date_range_extracted(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2023-01-01", "2024-12-31"])})
        profile = profile_dataframe(df)
        assert "2023-01-01" in profile["date_range"]
        assert "2024-12-31" in profile["date_range"]

    def test_no_date_cols_date_range_is_none(self):
        df = pd.DataFrame({"val": [1, 2, 3]})
        profile = profile_dataframe(df)
        assert profile["date_range"] is None

    def test_categorical_col_detected(self):
        df = pd.DataFrame({"cat": ["a", "b", "a", "c"]})
        profile = profile_dataframe(df)
        assert "cat" in profile["cat_cols"]

    def test_cat_counts_top_5(self):
        df = pd.DataFrame({"cat": ["a"] * 10 + ["b"] * 5 + ["c"] * 3 + ["d"] * 2 + ["e"] * 1 + ["f"] * 1})
        profile = profile_dataframe(df)
        assert len(profile["cat_counts"]["cat"]) <= 5
        assert "a" in profile["cat_counts"]["cat"]

    def test_null_percentage_calculated(self):
        df = pd.DataFrame({"val": [1.0, None, 3.0, None]})
        profile = profile_dataframe(df)
        col = next(c for c in profile["columns"] if c["name"] == "val")
        assert abs(col["null_pct"] - 0.5) < 0.01

    def test_sample_values_capped_at_5(self):
        df = pd.DataFrame({"cat": [str(i) for i in range(20)]})
        profile = profile_dataframe(df)
        col = next(c for c in profile["columns"] if c["name"] == "cat")
        assert len(col["sample_values"]) <= 5

    def test_no_truncation_under_limit(self):
        df = pd.DataFrame({"x": range(ROW_LIMIT - 1)})
        profile = profile_dataframe(df)
        assert profile["truncated"] is False
        assert profile["original_rows"] == ROW_LIMIT - 1
        assert profile["row_count"] == ROW_LIMIT - 1

    def test_truncation_at_row_limit(self):
        df = pd.DataFrame({"x": range(ROW_LIMIT + 50)})
        profile = profile_dataframe(df)
        assert profile["truncated"] is True
        assert profile["original_rows"] == ROW_LIMIT + 50
        assert profile["row_count"] == ROW_LIMIT

    def test_truncation_exactly_at_limit(self):
        df = pd.DataFrame({"x": range(ROW_LIMIT)})
        profile = profile_dataframe(df)
        assert profile["truncated"] is False
        assert profile["row_count"] == ROW_LIMIT

    def test_column_count_correct(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        profile = profile_dataframe(df)
        assert profile["col_count"] == 3

    def test_row_count_correct(self):
        df = pd.DataFrame({"x": range(42)})
        profile = profile_dataframe(df)
        assert profile["row_count"] == 42

    def test_mixed_df_profiles_all_cols(self):
        profile = profile_dataframe(SAMPLE_DF)
        col_names = [c["name"] for c in profile["columns"]]
        assert "date" in col_names
        assert "cost" in col_names
        assert "category" in col_names
        assert "vendor" in col_names

    def test_string_date_col_parsed(self):
        df = pd.DataFrame({"date": ["2023-01-01", "2023-06-15", "2024-01-10"]})
        profile = profile_dataframe(df)
        # String dates should be auto-converted to datetime if majority parse
        assert profile["date_range"] is not None or "date" in profile["cat_cols"]

    def test_unique_count_correct(self):
        df = pd.DataFrame({"cat": ["a", "a", "b", "c"]})
        profile = profile_dataframe(df)
        col = next(c for c in profile["columns"] if c["name"] == "cat")
        assert col["unique_count"] == 3

    def test_std_zero_for_single_row(self):
        df = pd.DataFrame({"cost": [100.0]})
        profile = profile_dataframe(df)
        assert profile["numeric_stats"]["cost"]["std"] == 0.0


# ---------------------------------------------------------------------------
# analyze_spreadsheet() tests
# ---------------------------------------------------------------------------

class TestAnalyzeSpreadsheet:

    def _profile(self):
        return profile_dataframe(SAMPLE_DF)

    def test_happy_path_returns_findings(self):
        profile = self._profile()
        with patch("tools.analytics_agent.genai.Client", return_value=_mock_client(_make_analyze_payload())):
            result = analyze_spreadsheet(profile, api_key="test-key")
        assert result["error"] is None
        assert len(result["findings"]) == 3
        assert result["narrative"] == "Test narrative."
        assert result["confidence"] == 0.8

    def test_findings_capped_at_8(self):
        many = [
            {"metric": f"M{i}", "value": f"{i}", "trend": "stable",
             "insight": "x", "severity": "info"}
            for i in range(12)
        ]
        payload = json.dumps({"findings": many, "narrative": "n", "confidence": 0.7})
        profile = self._profile()
        with patch("tools.analytics_agent.genai.Client", return_value=_mock_client(payload)):
            result = analyze_spreadsheet(profile, api_key="test-key")
        assert len(result["findings"]) == 8

    def test_confidence_clamped_high(self):
        payload = json.dumps({"findings": [], "narrative": "n", "confidence": 1.5})
        profile = self._profile()
        with patch("tools.analytics_agent.genai.Client", return_value=_mock_client(payload)):
            result = analyze_spreadsheet(profile, api_key="test-key")
        assert result["confidence"] == 1.0

    def test_confidence_clamped_low(self):
        payload = json.dumps({"findings": [], "narrative": "n", "confidence": -0.5})
        profile = self._profile()
        with patch("tools.analytics_agent.genai.Client", return_value=_mock_client(payload)):
            result = analyze_spreadsheet(profile, api_key="test-key")
        assert result["confidence"] == 0.0

    def test_markdown_fence_stripped(self):
        payload = "```json\n" + _make_analyze_payload() + "\n```"
        profile = self._profile()
        with patch("tools.analytics_agent.genai.Client", return_value=_mock_client(payload)):
            result = analyze_spreadsheet(profile, api_key="test-key")
        assert result["error"] is None
        assert len(result["findings"]) > 0

    def test_invalid_json_sets_error(self):
        profile = self._profile()
        with patch("tools.analytics_agent.genai.Client", return_value=_mock_client("not json")):
            result = analyze_spreadsheet(profile, api_key="test-key")
        assert result["error"] is not None
        assert "parse error" in result["error"].lower() or "JSONDecodeError" in result["error"]
        assert result["findings"] == []

    def test_empty_profile_returns_error(self):
        result = analyze_spreadsheet(profile_dataframe(pd.DataFrame()), api_key="test-key")
        assert result["error"] is not None
        assert result["findings"] == []

    def test_none_profile_returns_error(self):
        result = analyze_spreadsheet(None, api_key="test-key")
        assert result["error"] is not None

    def test_llm_exception_captured(self):
        profile = self._profile()
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API down")
        with patch("tools.analytics_agent.genai.Client", return_value=mock_client):
            result = analyze_spreadsheet(profile, api_key="test-key")
        assert result["error"] is not None
        assert "RuntimeError" in result["error"]
        assert result["findings"] == []

    def test_trend_normalization_unknown(self):
        findings = [
            {"metric": "M", "value": "v", "trend": "sideways",
             "insight": "i", "severity": "info"},
            {"metric": "M2", "value": "v2", "trend": "stable",
             "insight": "i2", "severity": "info"},
            {"metric": "M3", "value": "v3", "trend": "increasing",
             "insight": "i3", "severity": "info"},
        ]
        payload = json.dumps({"findings": findings, "narrative": "n", "confidence": 0.7})
        profile = self._profile()
        with patch("tools.analytics_agent.genai.Client", return_value=_mock_client(payload)):
            result = analyze_spreadsheet(profile, api_key="test-key")
        assert result["findings"][0]["trend"] == "unknown"
        assert result["findings"][1]["trend"] == "stable"

    def test_severity_normalization_to_info(self):
        findings = [
            {"metric": "M", "value": "v", "trend": "stable",
             "insight": "i", "severity": "extreme"},
            {"metric": "M2", "value": "v", "trend": "stable",
             "insight": "i", "severity": "critical"},
            {"metric": "M3", "value": "v", "trend": "stable",
             "insight": "i", "severity": "warning"},
        ]
        payload = json.dumps({"findings": findings, "narrative": "n", "confidence": 0.7})
        profile = self._profile()
        with patch("tools.analytics_agent.genai.Client", return_value=_mock_client(payload)):
            result = analyze_spreadsheet(profile, api_key="test-key")
        assert result["findings"][0]["severity"] == "info"
        assert result["findings"][1]["severity"] == "critical"
        assert result["findings"][2]["severity"] == "warning"

    def test_api_key_passed_to_client(self):
        profile = self._profile()
        with patch("tools.analytics_agent.genai.Client") as mock_cls:
            mock_cls.return_value = _mock_client(_make_analyze_payload())
            analyze_spreadsheet(profile, api_key="my-test-key")
        mock_cls.assert_called_once_with(api_key="my-test-key")

    def test_correlations_initialized_empty(self):
        profile = self._profile()
        with patch("tools.analytics_agent.genai.Client", return_value=_mock_client(_make_analyze_payload())):
            result = analyze_spreadsheet(profile, api_key="test-key")
        assert result["correlations"] == []

    def test_confidence_bad_type_defaults(self):
        payload = json.dumps({"findings": [], "narrative": "n", "confidence": "high"})
        profile = self._profile()
        with patch("tools.analytics_agent.genai.Client", return_value=_mock_client(payload)):
            result = analyze_spreadsheet(profile, api_key="test-key")
        assert result["confidence"] == 0.5

    def test_findings_non_list_handled(self):
        payload = json.dumps({"findings": "not a list", "narrative": "n", "confidence": 0.5})
        profile = self._profile()
        with patch("tools.analytics_agent.genai.Client", return_value=_mock_client(payload)):
            result = analyze_spreadsheet(profile, api_key="test-key")
        assert result["findings"] == []

    def test_profile_included_in_result(self):
        profile = self._profile()
        with patch("tools.analytics_agent.genai.Client", return_value=_mock_client(_make_analyze_payload())):
            result = analyze_spreadsheet(profile, api_key="test-key")
        assert result["profile"]["row_count"] == profile["row_count"]


# ---------------------------------------------------------------------------
# correlate_findings() tests
# ---------------------------------------------------------------------------

class TestCorrelateFindings:

    def _base_result(self) -> AnalyticsResult:
        profile = profile_dataframe(SAMPLE_DF)
        return AnalyticsResult(
            profile=profile,
            findings=[
                MetricFinding(metric="HVAC spend", value="$570", trend="increasing",
                              insight="HVAC dominates.", severity="warning"),
                MetricFinding(metric="Total spend", value="$665", trend="stable",
                              insight="Total logged.", severity="info"),
                MetricFinding(metric="Vendor concentration", value="AirPro 67%",
                              trend="stable", insight="Single vendor.", severity="critical"),
            ],
            narrative="HVAC costs are the primary driver.",
            correlations=[],
            confidence=0.8,
            error=None,
        )

    def test_happy_path_correlations_populated(self):
        result = self._base_result()
        with patch("tools.analytics_agent.genai.Client",
                   return_value=_mock_client(_make_correlate_payload())):
            updated = correlate_findings(result, SAMPLE_REGISTRY, api_key="test-key")
        assert len(updated["correlations"]) == 1
        assert updated["correlations"][0]["item_id"] == "HV-001"

    def test_empty_registry_returns_unchanged_no_llm_call(self):
        result = self._base_result()
        with patch("tools.analytics_agent.genai.Client") as mock_cls:
            updated = correlate_findings(result, [], api_key="test-key")
        mock_cls.assert_not_called()
        assert updated["correlations"] == []
        assert updated["findings"] == result["findings"]

    def test_empty_findings_returns_unchanged_no_llm_call(self):
        result = self._base_result()
        result = AnalyticsResult(**{**result, "findings": []})
        with patch("tools.analytics_agent.genai.Client") as mock_cls:
            updated = correlate_findings(result, SAMPLE_REGISTRY, api_key="test-key")
        mock_cls.assert_not_called()
        assert updated["correlations"] == []

    def test_no_match_response_returns_empty_correlations(self):
        result = self._base_result()
        payload = json.dumps({"correlations": []})
        with patch("tools.analytics_agent.genai.Client", return_value=_mock_client(payload)):
            updated = correlate_findings(result, SAMPLE_REGISTRY, api_key="test-key")
        assert updated["correlations"] == []

    def test_invalid_item_id_filtered_out(self):
        result = self._base_result()
        bad_payload = json.dumps({"correlations": [
            {"item_id": "XX-999", "item_title": "Fake", "relevance": "r",
             "proposed_note": "n"},
        ]})
        with patch("tools.analytics_agent.genai.Client", return_value=_mock_client(bad_payload)):
            updated = correlate_findings(result, SAMPLE_REGISTRY, api_key="test-key")
        assert updated["correlations"] == []

    def test_findings_and_narrative_preserved(self):
        result = self._base_result()
        with patch("tools.analytics_agent.genai.Client",
                   return_value=_mock_client(_make_correlate_payload())):
            updated = correlate_findings(result, SAMPLE_REGISTRY, api_key="test-key")
        assert updated["findings"] == result["findings"]
        assert updated["narrative"] == result["narrative"]
        assert updated["confidence"] == result["confidence"]

    def test_invalid_json_sets_error_preserves_result(self):
        result = self._base_result()
        with patch("tools.analytics_agent.genai.Client", return_value=_mock_client("bad json")):
            updated = correlate_findings(result, SAMPLE_REGISTRY, api_key="test-key")
        assert updated["correlations"] == []
        assert updated["error"] is not None
        assert "parse error" in updated["error"].lower() or "JSONDecodeError" in updated["error"]
        assert updated["findings"] == result["findings"]

    def test_llm_exception_captured_preserves_result(self):
        result = self._base_result()
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = ConnectionError("timeout")
        with patch("tools.analytics_agent.genai.Client", return_value=mock_client):
            updated = correlate_findings(result, SAMPLE_REGISTRY, api_key="test-key")
        assert updated["correlations"] == []
        assert updated["error"] is not None
        assert "ConnectionError" in updated["error"]

    def test_existing_error_appended_not_replaced(self):
        result = self._base_result()
        result = AnalyticsResult(**{**result, "error": "prior error"})
        with patch("tools.analytics_agent.genai.Client", return_value=_mock_client("bad json")):
            updated = correlate_findings(result, SAMPLE_REGISTRY, api_key="test-key")
        assert "prior error" in updated["error"]

    def test_multiple_correlations_returned(self):
        result = self._base_result()
        payload = json.dumps({"correlations": [
            {"item_id": "HV-001", "item_title": "HVAC", "relevance": "r1", "proposed_note": "n1"},
            {"item_id": "APP-001", "item_title": "Dishwasher", "relevance": "r2", "proposed_note": "n2"},
        ]})
        with patch("tools.analytics_agent.genai.Client", return_value=_mock_client(payload)):
            updated = correlate_findings(result, SAMPLE_REGISTRY, api_key="test-key")
        assert len(updated["correlations"]) == 2

    def test_non_list_correlations_handled(self):
        result = self._base_result()
        payload = json.dumps({"correlations": "not a list"})
        with patch("tools.analytics_agent.genai.Client", return_value=_mock_client(payload)):
            updated = correlate_findings(result, SAMPLE_REGISTRY, api_key="test-key")
        assert updated["correlations"] == []

    def test_api_key_passed_to_client(self):
        result = self._base_result()
        with patch("tools.analytics_agent.genai.Client") as mock_cls:
            mock_cls.return_value = _mock_client(_make_correlate_payload())
            correlate_findings(result, SAMPLE_REGISTRY, api_key="my-key")
        mock_cls.assert_called_once_with(api_key="my-key")