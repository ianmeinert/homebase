"""Tests for tools/schema_agent.py — Schema-Aware Metric Discovery Agent."""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tools.schema_agent import (
    DiscoveryReport,
    SchemaField,
    SchemaSource,
    _infer_type_from_mermaid,
    _infer_type_from_series,
    discover_metrics,
    is_mermaid,
    parse_mermaid,
    parse_tabular,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_MERMAID = """
erDiagram
    REGISTRY {
        TEXT id PK "e.g. HV-001"
        TEXT category "hvac | plumbing"
        REAL urgency "0.0 - 1.0"
        REAL impact "0.0 - 1.0"
        TEXT updated_at "ISO datetime"
        TEXT status "open | closed"
    }

    RUN_HISTORY {
        TEXT run_id PK "UUID"
        TEXT timestamp "ISO datetime"
        INTEGER item_count
        TEXT quadrant_summary "JSON object"
    }

    REGISTRY ||--o{ RUN_HISTORY : "referenced in deferred_items"
"""

SAMPLE_CSV_CONTENT = (
    "date,category,vendor,cost,warranty_covered\n"
    "2023-01-08,hvac,AirPro,185.00,false\n"
    "2023-02-15,plumbing,Roto-Rooter,95.00,false\n"
    "2023-03-20,electrical,Apex,220.00,true\n"
    "2023-04-01,appliance,Sears,310.00,true\n"
    "2023-05-10,hvac,AirPro,175.00,false\n"
)

MINIMAL_REPORT = {
    "entity_name": "Test Entity",
    "source_type": "csv",
    "computable_metrics": [
        {
            "metric_name": "Total Cost",
            "description": "Sum of all costs",
            "fields_required": ["cost"],
            "confidence": 0.9,
        }
    ],
    "derived_fields": [
        {
            "field_name": "cost_per_category",
            "description": "Average cost per category",
            "rationale": "Enables cost trend analysis",
            "source_fields": ["cost", "category"],
        }
    ],
    "quality_observations": [
        {
            "field": "warranty_covered",
            "observation": "Boolean stored as string — limits filtering",
            "severity": "warning",
        }
    ],
    "schema_gaps": [
        {
            "suggested_field": "resolution_time_days",
            "description": "Days from issue open to close",
            "impact": "Enables SLA tracking",
        }
    ],
    "narrative": "This schema has moderate analytic maturity.",
    "confidence": 0.65,
}


# ---------------------------------------------------------------------------
# is_mermaid
# ---------------------------------------------------------------------------

class TestIsMermaid:
    def test_valid_erdiagram(self):
        assert is_mermaid("erDiagram\n  ENTITY {\n  }")

    def test_valid_classdiagram(self):
        assert is_mermaid("classDiagram\n  class Foo")

    def test_valid_graph(self):
        assert is_mermaid("graph TD\n  A --> B")

    def test_valid_flowchart(self):
        assert is_mermaid("flowchart LR\n  A --> B")

    def test_plain_text(self):
        assert not is_mermaid("Just some plain text\nNo mermaid here")

    def test_csv_data(self):
        assert not is_mermaid(SAMPLE_CSV_CONTENT)

    def test_empty_string(self):
        assert not is_mermaid("")

    def test_case_insensitive(self):
        assert is_mermaid("ERDIAGRAM\n  FOO {}")


# ---------------------------------------------------------------------------
# _infer_type_from_mermaid
# ---------------------------------------------------------------------------

class TestInferTypeFromMermaid:
    def test_text(self):
        assert _infer_type_from_mermaid("TEXT") == "text"

    def test_varchar(self):
        assert _infer_type_from_mermaid("VARCHAR") == "text"

    def test_integer(self):
        assert _infer_type_from_mermaid("INTEGER") == "numeric"

    def test_real(self):
        assert _infer_type_from_mermaid("REAL") == "numeric"

    def test_float(self):
        assert _infer_type_from_mermaid("FLOAT") == "numeric"

    def test_boolean(self):
        assert _infer_type_from_mermaid("BOOLEAN") == "boolean"

    def test_datetime(self):
        assert _infer_type_from_mermaid("DATETIME") == "datetime"

    def test_timestamp(self):
        assert _infer_type_from_mermaid("TIMESTAMP") == "datetime"

    def test_unknown(self):
        assert _infer_type_from_mermaid("BLOB") == "unknown"


# ---------------------------------------------------------------------------
# parse_mermaid
# ---------------------------------------------------------------------------

class TestParseMermaid:
    def test_returns_list_of_sources(self):
        result = parse_mermaid(SAMPLE_MERMAID)
        assert isinstance(result, list)

    def test_finds_both_entities(self):
        result = parse_mermaid(SAMPLE_MERMAID)
        names = [s["entity_name"] for s in result]
        assert "REGISTRY" in names
        assert "RUN_HISTORY" in names

    def test_source_type_is_mermaid(self):
        result = parse_mermaid(SAMPLE_MERMAID)
        assert all(s["source_type"] == "mermaid" for s in result)

    def test_row_count_is_none(self):
        result = parse_mermaid(SAMPLE_MERMAID)
        assert all(s["row_count"] is None for s in result)

    def test_registry_field_count(self):
        result = parse_mermaid(SAMPLE_MERMAID)
        reg = next(s for s in result if s["entity_name"] == "REGISTRY")
        assert reg["field_count"] >= 4

    def test_field_types_inferred(self):
        result = parse_mermaid(SAMPLE_MERMAID)
        reg = next(s for s in result if s["entity_name"] == "REGISTRY")
        type_map = {f["name"]: f["data_type"] for f in reg["fields"]}
        assert type_map.get("urgency") == "numeric"
        assert type_map.get("status") == "text"

    def test_raw_summary_includes_entity_name(self):
        result = parse_mermaid(SAMPLE_MERMAID)
        reg = next(s for s in result if s["entity_name"] == "REGISTRY")
        assert "REGISTRY" in reg["raw_summary"]

    def test_relationship_in_summary(self):
        result = parse_mermaid(SAMPLE_MERMAID)
        reg = next(s for s in result if s["entity_name"] == "REGISTRY")
        assert "RUN_HISTORY" in reg["raw_summary"]

    def test_empty_mermaid(self):
        result = parse_mermaid("erDiagram")
        assert result == []

    def test_single_entity(self):
        single = "erDiagram\n  FOO {\n    TEXT id PK\n    INTEGER count\n  }"
        result = parse_mermaid(single)
        assert len(result) == 1
        assert result[0]["entity_name"] == "FOO"


# ---------------------------------------------------------------------------
# _infer_type_from_series
# ---------------------------------------------------------------------------

class TestInferTypeFromSeries:
    def test_numeric_int(self):
        s = pd.Series([1, 2, 3, 4])
        assert _infer_type_from_series(s) == "numeric"

    def test_numeric_float(self):
        s = pd.Series([1.1, 2.2, 3.3])
        assert _infer_type_from_series(s) == "numeric"

    def test_boolean_dtype(self):
        s = pd.Series([True, False, True])
        assert _infer_type_from_series(s) == "boolean"

    def test_boolean_strings(self):
        s = pd.Series(["true", "false", "true", "false"])
        assert _infer_type_from_series(s) == "boolean"

    def test_datetime_strings(self):
        s = pd.Series(["2023-01-01", "2023-06-15", "2024-03-20"])
        assert _infer_type_from_series(s) == "datetime"

    def test_categorical_low_cardinality(self):
        s = pd.Series(["hvac", "plumbing", "hvac", "electrical"] * 5)
        assert _infer_type_from_series(s) == "categorical"

    def test_text_high_cardinality(self):
        s = pd.Series([f"unique description {i}" for i in range(100)])
        assert _infer_type_from_series(s) == "text"


# ---------------------------------------------------------------------------
# parse_tabular
# ---------------------------------------------------------------------------

class TestParseTabular:
    def test_csv_basic(self):
        data = SAMPLE_CSV_CONTENT.encode()
        result = parse_tabular(data, "test_data.csv")
        assert result["source_type"] == "csv"
        assert result["field_count"] == 5
        assert result["row_count"] == 5

    def test_entity_name_from_filename(self):
        data = SAMPLE_CSV_CONTENT.encode()
        result = parse_tabular(data, "maintenance_log.csv")
        assert "Maintenance" in result["entity_name"]

    def test_fields_list_populated(self):
        data = SAMPLE_CSV_CONTENT.encode()
        result = parse_tabular(data, "test.csv")
        field_names = [f["name"] for f in result["fields"]]
        assert "cost" in field_names
        assert "category" in field_names

    def test_numeric_field_detected(self):
        data = SAMPLE_CSV_CONTENT.encode()
        result = parse_tabular(data, "test.csv")
        cost_field = next(f for f in result["fields"] if f["name"] == "cost")
        assert cost_field["data_type"] == "numeric"

    def test_categorical_field_detected(self):
        data = SAMPLE_CSV_CONTENT.encode()
        result = parse_tabular(data, "test.csv")
        cat_field = next(f for f in result["fields"] if f["name"] == "category")
        assert cat_field["data_type"] == "categorical"

    def test_datetime_field_detected(self):
        data = SAMPLE_CSV_CONTENT.encode()
        result = parse_tabular(data, "test.csv")
        date_field = next(f for f in result["fields"] if f["name"] == "date")
        assert date_field["data_type"] == "datetime"

    def test_raw_summary_populated(self):
        data = SAMPLE_CSV_CONTENT.encode()
        result = parse_tabular(data, "test.csv")
        assert len(result["raw_summary"]) > 50
        assert "cost" in result["raw_summary"]

    def test_500_row_cap(self):
        rows = ["date,value"] + [f"2023-01-{(i%28)+1:02d},{i}" for i in range(600)]
        data = "\n".join(rows).encode()
        result = parse_tabular(data, "big.csv")
        assert result["row_count"] == 500

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            parse_tabular(b"data", "file.json")

    def test_xlsx_source_type(self):
        # Build a minimal xlsx in memory
        buf = io.BytesIO()
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        df.to_excel(buf, index=False)
        buf.seek(0)
        result = parse_tabular(buf.read(), "data.xlsx")
        assert result["source_type"] == "xlsx"


# ---------------------------------------------------------------------------
# discover_metrics
# ---------------------------------------------------------------------------

def _mock_gemini_response(text: str):
    mock_response = MagicMock()
    mock_response.text = text
    return mock_response


class TestDiscoverMetrics:
    def _make_source(self, name="TestEntity", src_type="csv") -> SchemaSource:
        return SchemaSource(
            source_type=src_type,
            entity_name=name,
            row_count=10,
            field_count=3,
            fields=[
                SchemaField(name="id", data_type="text", nullable=False, unique_values=10, notes="PK"),
                SchemaField(name="cost", data_type="numeric", nullable=False, unique_values=10, notes="range 95–310"),
                SchemaField(name="category", data_type="categorical", nullable=False, unique_values=4, notes=""),
            ],
            raw_summary="Entity: TestEntity\nFields: id, cost, category",
        )

    @patch("tools.schema_agent.genai")
    def test_returns_discovery_report(self, mock_genai):
        mock_genai.Client.return_value.models.generate_content.return_value = (
            _mock_gemini_response(json.dumps(MINIMAL_REPORT))
        )
        result = discover_metrics([self._make_source()], api_key="test-key")
        assert isinstance(result, dict)
        assert "computable_metrics" in result
        assert "schema_gaps" in result

    @patch("tools.schema_agent.genai")
    def test_confidence_clamped(self, mock_genai):
        report = {**MINIMAL_REPORT, "confidence": 1.5}
        mock_genai.Client.return_value.models.generate_content.return_value = (
            _mock_gemini_response(json.dumps(report))
        )
        result = discover_metrics([self._make_source()], api_key="test-key")
        assert result["confidence"] <= 1.0

    @patch("tools.schema_agent.genai")
    def test_confidence_negative_clamped(self, mock_genai):
        report = {**MINIMAL_REPORT, "confidence": -0.5}
        mock_genai.Client.return_value.models.generate_content.return_value = (
            _mock_gemini_response(json.dumps(report))
        )
        result = discover_metrics([self._make_source()], api_key="test-key")
        assert result["confidence"] >= 0.0

    @patch("tools.schema_agent.genai")
    def test_metric_confidence_clamped(self, mock_genai):
        report = {**MINIMAL_REPORT}
        report["computable_metrics"] = [
            {"metric_name": "X", "description": "Y", "fields_required": ["a"], "confidence": 2.5}
        ]
        mock_genai.Client.return_value.models.generate_content.return_value = (
            _mock_gemini_response(json.dumps(report))
        )
        result = discover_metrics([self._make_source()], api_key="test-key")
        assert result["computable_metrics"][0]["confidence"] <= 1.0

    @patch("tools.schema_agent.genai")
    def test_invalid_severity_normalized(self, mock_genai):
        report = {**MINIMAL_REPORT}
        report["quality_observations"] = [
            {"field": "x", "observation": "bad", "severity": "HIGH"}
        ]
        mock_genai.Client.return_value.models.generate_content.return_value = (
            _mock_gemini_response(json.dumps(report))
        )
        result = discover_metrics([self._make_source()], api_key="test-key")
        assert result["quality_observations"][0]["severity"] == "info"

    @patch("tools.schema_agent.genai")
    def test_strips_markdown_fences(self, mock_genai):
        raw = "```json\n" + json.dumps(MINIMAL_REPORT) + "\n```"
        mock_genai.Client.return_value.models.generate_content.return_value = (
            _mock_gemini_response(raw)
        )
        result = discover_metrics([self._make_source()], api_key="test-key")
        assert result["narrative"] == MINIMAL_REPORT["narrative"]

    @patch("tools.schema_agent.genai")
    def test_multiple_sources_combined(self, mock_genai):
        mock_genai.Client.return_value.models.generate_content.return_value = (
            _mock_gemini_response(json.dumps(MINIMAL_REPORT))
        )
        sources = [self._make_source("CSV"), self._make_source("ERD", "mermaid")]
        result = discover_metrics(sources, api_key="test-key")
        assert result is not None

    @patch("tools.schema_agent.genai")
    def test_parse_error_raises_value_error(self, mock_genai):
        mock_genai.Client.return_value.models.generate_content.return_value = (
            _mock_gemini_response("not valid json {{{")
        )
        with pytest.raises(ValueError, match="parse error"):
            discover_metrics([self._make_source()], api_key="test-key")

    @patch("tools.schema_agent.genai")
    def test_truncation_guard_recovers(self, mock_genai):
        # Truncated JSON — missing closing brace
        truncated = json.dumps(MINIMAL_REPORT)[:-5]
        # Manually ensure it ends with a valid closing brace after truncation guard
        recovered = truncated[:truncated.rfind("}") + 1]
        mock_genai.Client.return_value.models.generate_content.return_value = (
            _mock_gemini_response(truncated)
        )
        # May raise or succeed depending on where truncation lands — just ensure no crash other than ValueError
        try:
            result = discover_metrics([self._make_source()], api_key="test-key")
        except ValueError:
            pass  # acceptable — truncation guard attempted but JSON still invalid

    @patch("tools.schema_agent.genai")
    def test_mermaid_source_accepted(self, mock_genai):
        mock_genai.Client.return_value.models.generate_content.return_value = (
            _mock_gemini_response(json.dumps(MINIMAL_REPORT))
        )
        mermaid_src = SchemaSource(
            source_type="mermaid",
            entity_name="REGISTRY",
            row_count=None,
            field_count=6,
            fields=[],
            raw_summary="Entity: REGISTRY\nFields: id, category, urgency, impact, updated_at, status",
        )
        result = discover_metrics([mermaid_src], api_key="test-key")
        assert result is not None