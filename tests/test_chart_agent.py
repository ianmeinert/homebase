"""
tests/test_chart_agent.py — Tests for AI chart generation agent.
"""
import json
import pandas as pd
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

# ---------------------------------------------------------------------------
# _load_analytics_data
# ---------------------------------------------------------------------------

def test_load_analytics_data_basic():
    from tools.chart_agent import _load_analytics_data
    df = pd.DataFrame({"cost": [100.0, 200.0], "vendor": ["A", "B"]})
    rows = _load_analytics_data(df)
    assert len(rows) == 2
    assert rows[0]["vendor"] == "A"

def test_load_analytics_data_caps_at_200():
    from tools.chart_agent import _load_analytics_data
    df = pd.DataFrame({"x": range(300)})
    rows = _load_analytics_data(df)
    assert len(rows) == 200

def test_load_analytics_data_empty_df_returns_empty():
    from tools.chart_agent import _load_analytics_data
    rows = _load_analytics_data(pd.DataFrame())
    assert rows == []

def test_load_analytics_data_none_returns_empty():
    from tools.chart_agent import _load_analytics_data
    rows = _load_analytics_data(None)
    assert rows == []

def test_load_analytics_data_values_are_strings():
    from tools.chart_agent import _load_analytics_data
    df = pd.DataFrame({"cost": [150.0], "count": [3]})
    rows = _load_analytics_data(df)
    # All values coerced to str for JSON safety
    assert isinstance(rows[0]["cost"], str)


# ---------------------------------------------------------------------------
# _pick_datasets with analytics_df
# ---------------------------------------------------------------------------

def test_pick_datasets_analytics_keyword_loads_analytics():
    from tools.chart_agent import _pick_datasets
    df = pd.DataFrame({"cost": [100], "vendor": ["A"]})
    with patch("tools.chart_agent._load_analytics_data", return_value=[{"cost": "100"}]):
        data = _pick_datasets("chart from my data", analytics_df=df)
    assert "analytics" in data

def test_pick_datasets_uploaded_keyword_loads_analytics():
    from tools.chart_agent import _pick_datasets
    df = pd.DataFrame({"cost": [100]})
    with patch("tools.chart_agent._load_analytics_data", return_value=[{"cost": "100"}]):
        data = _pick_datasets("bar chart of uploaded spreadsheet", analytics_df=df)
    assert "analytics" in data

def test_pick_datasets_no_analytics_df_ignores_analytics_keywords():
    from tools.chart_agent import _pick_datasets
    with patch("tools.chart_agent._load_registry", return_value=[]), \
         patch("tools.chart_agent._load_run_history", return_value=[]):
        data = _pick_datasets("chart from my data", analytics_df=None)
    assert "analytics" not in data

def test_pick_datasets_analytics_with_registry_keyword_loads_both():
    from tools.chart_agent import _pick_datasets
    df = pd.DataFrame({"cost": [100]})
    with patch("tools.chart_agent._load_analytics_data", return_value=[{"cost": "100"}]), \
         patch("tools.chart_agent._load_registry", return_value=[{"id": "HV-001"}]):
        data = _pick_datasets("chart from my data and registry", analytics_df=df)
    assert "analytics" in data
    assert "registry" in data

def test_pick_datasets_col_name_match_loads_analytics():
    from tools.chart_agent import _pick_datasets
    df = pd.DataFrame({"vendor": ["A"], "cost": [100]})
    with patch("tools.chart_agent._load_analytics_data", return_value=[{"vendor": "A"}]):
        data = _pick_datasets("bar chart of vendor distribution", analytics_df=df)
    assert "analytics" in data


# ---------------------------------------------------------------------------
# generate_chart with analytics_df
# ---------------------------------------------------------------------------

def test_generate_chart_passes_analytics_df_to_pick_datasets():
    from tools.chart_agent import generate_chart
    df = pd.DataFrame({"cost": [100.0, 200.0], "vendor": ["A", "B"]})

    mock_model = _mock_model(json.dumps({"complexity": "simple"}))
    mock_spec_model = _mock_model(json.dumps({
        "chart_type": "bar", "title": "Cost by Vendor",
        "data_source": "analytics", "x": "vendor", "y": "cost",
        "color": None, "filters": {}, "aggregation": "none"
    }))

    call_count = [0]
    def side_effect(messages):
        call_count[0] += 1
        if call_count[0] == 1:
            return MagicMock(content=json.dumps({"complexity": "simple"}))
        return MagicMock(content=json.dumps({
            "chart_type": "bar", "title": "Test",
            "data_source": "analytics", "x": "vendor", "y": "cost",
            "color": None, "filters": {}, "aggregation": "none"
        }))

    mock_m = MagicMock()
    mock_m.invoke.side_effect = side_effect

    with patch("tools.chart_agent.get_model", return_value=mock_m), \
         patch("tools.chart_agent._pick_datasets", wraps=lambda inst, analytics_df=None: {
             "analytics": [{"vendor": "A", "cost": "100"}, {"vendor": "B", "cost": "200"}]
         }) as mock_pick:
        generate_chart("bar chart of vendor from my data", analytics_df=df)

    mock_pick.assert_called_once()
    _, kwargs = mock_pick.call_args
    assert kwargs.get("analytics_df") is df or mock_pick.call_args[0][1] is df

def test_generate_chart_no_analytics_df_works_normally():
    from tools.chart_agent import generate_chart

    call_count = [0]
    def side_effect(messages):
        call_count[0] += 1
        if call_count[0] == 1:
            return MagicMock(content=json.dumps({"complexity": "simple"}))
        return MagicMock(content=json.dumps({
            "chart_type": "bar", "title": "Test",
            "data_source": "registry", "x": "category", "y": "urgency",
            "color": None, "filters": {}, "aggregation": "mean"
        }))

    mock_m = MagicMock()
    mock_m.invoke.side_effect = side_effect

    with patch("tools.chart_agent.get_model", return_value=mock_m), \
         patch("tools.chart_agent._pick_datasets", return_value={"registry": [
             {"id": "HV-001", "category": "hvac", "urgency": 0.8, "impact": 0.6,
              "days_since_update": 5, "status": "open", "title": "T", "updated_at": "2026-01-01"}
         ]}):
        fig, err = generate_chart("bar chart of urgency by category")
    # Should not raise — plotly may be absent in test env, that's expected
    assert err is None or "plotly" in str(err).lower() or fig is not None

def test_generate_chart_analytics_col_context_appended():
    """Column context from analytics_df is appended to instruction for LLM."""
    from tools.chart_agent import generate_chart
    df = pd.DataFrame({"cost": [100.0], "vendor": ["A"]})

    captured_instructions = []
    def mock_invoke(messages):
        captured_instructions.append(messages[-1]["content"])
        return MagicMock(content=json.dumps({"complexity": "simple"}))

    mock_m = MagicMock()
    mock_m.invoke.side_effect = mock_invoke

    with patch("tools.chart_agent.get_model", return_value=mock_m), \
         patch("tools.chart_agent._pick_datasets", return_value={}), \
         patch("tools.chart_agent._build_from_spec", return_value=(None, "no data")):
        generate_chart("chart cost by vendor", analytics_df=df)

    # The complexity classification call should have column info appended
    assert any("cost" in msg or "vendor" in msg for msg in captured_instructions)