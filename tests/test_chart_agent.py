"""
tests/test_chart_agent.py — Tests for AI chart generation agent.
"""
import json
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_model(response_text: str):
    m = MagicMock()
    m.invoke.return_value = MagicMock(content=response_text)
    return m


# ---------------------------------------------------------------------------
# _pick_datasets
# ---------------------------------------------------------------------------

def test_pick_datasets_registry_keywords():
    from tools.chart_agent import _pick_datasets
    data = _pick_datasets("chart urgency by category")
    assert "registry" in data

def test_pick_datasets_history_keywords():
    from tools.chart_agent import _pick_datasets
    data = _pick_datasets("plot run history trends over time")
    assert "run_history" in data

def test_pick_datasets_both_on_ambiguous():
    from tools.chart_agent import _pick_datasets
    data = _pick_datasets("show me everything")
    assert "registry" in data
    assert "run_history" in data


# ---------------------------------------------------------------------------
# _build_from_spec — deterministic figure builder
# ---------------------------------------------------------------------------

def _sample_data():
    return {
        "registry": [
            {"id": "HV-001", "category": "hvac", "title": "Filter", "urgency": 0.8,
             "impact": 0.6, "days_since_update": 5, "status": "open"},
            {"id": "PL-001", "category": "plumbing", "title": "Leak", "urgency": 0.9,
             "impact": 0.9, "days_since_update": 20, "status": "open"},
            {"id": "EL-001", "category": "electrical", "title": "Breaker", "urgency": 0.3,
             "impact": 0.5, "days_since_update": 3, "status": "open"},
        ],
        "run_history": [
            {"run_id": "r1", "date": "2026-01-01", "trigger": "weekly", "item_count": 5,
             "stale_count": 1, "hitl_approved": 1, "category_filter": "all"},
            {"run_id": "r2", "date": "2026-01-08", "trigger": "weekly", "item_count": 7,
             "stale_count": 2, "hitl_approved": 1, "category_filter": "all"},
        ]
    }


def test_build_bar_chart():
    from tools.chart_agent import _build_from_spec
    spec = {"chart_type": "bar", "title": "Urgency by Category",
            "data_source": "registry", "x": "category", "y": "urgency",
            "color": None, "filters": {}, "aggregation": "mean"}
    fig, err = _build_from_spec(spec, _sample_data())
    assert err is None
    assert fig is not None
    assert len(fig.data) > 0


def test_build_scatter_chart():
    from tools.chart_agent import _build_from_spec
    spec = {"chart_type": "scatter", "title": "Impact vs Urgency",
            "data_source": "registry", "x": "urgency", "y": "impact",
            "color": None, "filters": {}, "aggregation": "none"}
    fig, err = _build_from_spec(spec, _sample_data())
    assert err is None
    assert fig is not None


def test_build_pie_chart():
    from tools.chart_agent import _build_from_spec
    spec = {"chart_type": "pie", "title": "Items by Status",
            "data_source": "registry", "x": "status", "y": "count",
            "color": None, "filters": {}, "aggregation": "count"}
    fig, err = _build_from_spec(spec, _sample_data())
    assert err is None
    assert fig is not None


def test_build_line_chart():
    from tools.chart_agent import _build_from_spec
    spec = {"chart_type": "line", "title": "Item Count Over Time",
            "data_source": "run_history", "x": "date", "y": "item_count",
            "color": None, "filters": {}, "aggregation": "none"}
    fig, err = _build_from_spec(spec, _sample_data())
    assert err is None
    assert fig is not None


def test_build_histogram():
    from tools.chart_agent import _build_from_spec
    spec = {"chart_type": "histogram", "title": "Urgency Distribution",
            "data_source": "registry", "x": "urgency", "y": None,
            "color": None, "filters": {}, "aggregation": "none"}
    fig, err = _build_from_spec(spec, _sample_data())
    assert err is None
    assert fig is not None


def test_build_with_color_grouping():
    from tools.chart_agent import _build_from_spec
    spec = {"chart_type": "bar", "title": "Urgency by Category (colored by status)",
            "data_source": "registry", "x": "category", "y": "urgency",
            "color": "status", "filters": {}, "aggregation": "mean"}
    fig, err = _build_from_spec(spec, _sample_data())
    assert err is None
    assert len(fig.data) >= 1


def test_build_with_filter():
    from tools.chart_agent import _build_from_spec
    spec = {"chart_type": "bar", "title": "Open Items Urgency",
            "data_source": "registry", "x": "category", "y": "urgency",
            "color": None, "filters": {"status": "open"}, "aggregation": "mean"}
    fig, err = _build_from_spec(spec, _sample_data())
    assert err is None
    assert fig is not None


def test_build_filter_no_rows():
    from tools.chart_agent import _build_from_spec
    spec = {"chart_type": "bar", "title": "Test",
            "data_source": "registry", "x": "category", "y": "urgency",
            "color": None, "filters": {"status": "nonexistent"}, "aggregation": "none"}
    fig, err = _build_from_spec(spec, _sample_data())
    assert fig is None
    assert "no rows" in err.lower()


def test_build_unknown_chart_type_falls_back_to_bar():
    from tools.chart_agent import _build_from_spec
    spec = {"chart_type": "waterfall", "title": "Test",
            "data_source": "registry", "x": "category", "y": "urgency",
            "color": None, "filters": {}, "aggregation": "none"}
    fig, err = _build_from_spec(spec, _sample_data())
    # should fall back to bar and succeed
    assert err is None
    assert fig is not None


def test_build_empty_datasource():
    from tools.chart_agent import _build_from_spec
    spec = {"chart_type": "bar", "title": "Empty",
            "data_source": "registry", "x": "category", "y": "urgency",
            "color": None, "filters": {}, "aggregation": "none"}
    fig, err = _build_from_spec(spec, {"registry": [], "run_history": []})
    assert fig is None
    assert err is not None


# ---------------------------------------------------------------------------
# _build_complex — LLM returns full figure dict
# ---------------------------------------------------------------------------

def test_build_complex_valid_figure():
    from tools.chart_agent import _build_complex
    fig_dict = {
        "data": [{"type": "bar", "x": ["Jan", "Feb"], "y": [3, 5], "name": "items"}],
        "layout": {"title": {"text": "Test"}, "paper_bgcolor": "#0d1117",
                   "plot_bgcolor": "#161b22", "font": {"color": "#e6edf3"}}
    }
    with patch("tools.chart_agent.get_model") as mock_gm:
        mock_gm.return_value = _mock_model(json.dumps(fig_dict))
        fig, err = _build_complex("complex chart instruction", _sample_data())
    assert err is None
    assert fig is not None


def test_build_complex_sanitizes_bad_trace_type():
    from tools.chart_agent import _build_complex
    fig_dict = {
        "data": [{"type": "candlestick", "x": ["A"], "y": [1]}],
        "layout": {"title": {"text": "Bad type"}}
    }
    with patch("tools.chart_agent.get_model") as mock_gm:
        mock_gm.return_value = _mock_model(json.dumps(fig_dict))
        fig, err = _build_complex("complex chart", _sample_data())
    # Should sanitize to bar and not raise
    assert err is None


def test_build_complex_handles_json_error():
    from tools.chart_agent import _build_complex
    with patch("tools.chart_agent.get_model") as mock_gm:
        mock_gm.return_value = _mock_model("not valid json {{}")
        fig, err = _build_complex("complex chart", _sample_data())
    assert fig is None
    assert err is not None


# ---------------------------------------------------------------------------
# generate_chart — top-level integration (mocked LLM)
# ---------------------------------------------------------------------------

def test_generate_chart_simple_path():
    spec = {"chart_type": "bar", "title": "Urgency by Category",
            "data_source": "registry", "x": "category", "y": "urgency",
            "color": None, "filters": {}, "aggregation": "mean"}
    responses = [
        json.dumps({"complexity": "simple"}),
        json.dumps(spec),
    ]
    call_count = {"n": 0}
    def side_effect(*args, **kwargs):
        r = responses[call_count["n"]]
        call_count["n"] += 1
        return MagicMock(content=r)

    mock_model = MagicMock()
    mock_model.invoke.side_effect = side_effect

    with patch("tools.chart_agent.get_model", return_value=mock_model):
        from tools.chart_agent import generate_chart
        fig, err = generate_chart("chart urgency by category")

    assert err is None
    assert fig is not None


def test_generate_chart_complex_path():
    fig_dict = {
        "data": [{"type": "scatter", "x": [0.1, 0.9], "y": [0.2, 0.8]}],
        "layout": {"title": {"text": "Complex"}}
    }
    responses = [
        json.dumps({"complexity": "complex"}),
        json.dumps(fig_dict),
    ]
    call_count = {"n": 0}
    def side_effect(*args, **kwargs):
        r = responses[call_count["n"]]
        call_count["n"] += 1
        return MagicMock(content=r)

    mock_model = MagicMock()
    mock_model.invoke.side_effect = side_effect

    with patch("tools.chart_agent.get_model", return_value=mock_model):
        from tools.chart_agent import generate_chart
        fig, err = generate_chart("compare stale count vs item count per run")

    assert err is None
    assert fig is not None


def test_generate_chart_llm_failure_returns_error():
    with patch("tools.chart_agent.get_model") as mock_gm:
        mock_gm.return_value.invoke.side_effect = Exception("API down")
        from tools.chart_agent import generate_chart
        fig, err = generate_chart("chart anything")
    assert fig is None
    assert err is not None


# ---------------------------------------------------------------------------
# classify_input — chart intent routing
# ---------------------------------------------------------------------------

def test_classify_chart_keyword():
    from tools.update_agent import classify_input
    assert classify_input("chart urgency by category") == "chart"

def test_classify_plot_keyword():
    from tools.update_agent import classify_input
    assert classify_input("plot run history over time") == "chart"

def test_classify_graph_keyword():
    from tools.update_agent import classify_input
    assert classify_input("graph the stale items by category") == "chart"

def test_classify_histogram_keyword():
    from tools.update_agent import classify_input
    assert classify_input("show me a histogram of urgency scores") == "chart"

def test_classify_visualize_keyword():
    from tools.update_agent import classify_input
    assert classify_input("visualize impact vs urgency scatter") == "chart"

def test_chart_keyword_doesnt_override_registry_op():
    from tools.update_agent import classify_input
    # "add" + no chart keyword — should stay registry
    result = classify_input("add new hvac filter item")
    assert result == "registry"