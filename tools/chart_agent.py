"""
tools/chart_agent.py  —  AI-generated charts from plain language instructions.

Two-tier approach:
  Simple  → LLM returns a structured spec (chart_type, title, x, y, filters)
            and we build the Plotly figure deterministically.
  Complex → LLM receives the full dataset + instruction and returns a complete
            Plotly Figure dict (data + layout); we hydrate go.Figure() from it.
            No exec() — pure data, not code.

Allowed trace types (whitelist): bar, scatter, line, pie, heatmap, box, histogram
"""

import json
from tools.llm_tools import get_model
from tools.db import get_conn

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_registry() -> list[dict]:
    """Return all non-closed registry items as list of dicts."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, category, title, urgency, impact, days_since_update, status FROM registry"
    ).fetchall()
    conn.close()
    keys = ["id", "category", "title", "urgency", "impact", "days_since_update", "status"]
    return [dict(zip(keys, r)) for r in rows]


def _load_run_history() -> list[dict]:
    """Return run history rows as list of dicts."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT run_id, timestamp, trigger, category_filter, item_count, "
        "stale_count, hitl_approved FROM run_history ORDER BY timestamp ASC"
    ).fetchall()
    conn.close()
    keys = ["run_id", "timestamp", "trigger", "category_filter",
            "item_count", "stale_count", "hitl_approved"]
    records = []
    for i, r in enumerate(rows, 1):
        d = dict(zip(keys, r))
        # run_label: short categorical label safe for x-axis (avoids datetime parsing issues)
        ts = d.get("timestamp", "")
        if ts and len(ts) >= 16:
            d["run_label"] = f"Run {i}  {ts[5:10]} {ts[11:16]}"  # e.g. "Run 1  03-10 14:32"
        else:
            d["run_label"] = f"Run {i}"
        d["run_index"] = i
        # Keep date for queries that genuinely need it
        d["date"] = ts[:10] if ts else ""
        records.append(d)
    return records


def _pick_datasets(instruction: str) -> dict:
    """Heuristically decide which tables to load based on instruction keywords."""
    low = instruction.lower()
    history_kws = {"history", "run", "trend", "over time", "past", "stale count",
                   "item count", "hitl", "approved", "date", "month", "week"}
    registry_kws = {"urgency", "impact", "category", "status", "open", "closed",
                    "in progress", "quadrant", "scatter", "registry", "items"}

    want_history  = any(k in low for k in history_kws)
    want_registry = any(k in low for k in registry_kws)

    # Default: if neither matches clearly, load both (LLM will decide what to use)
    if not want_history and not want_registry:
        want_history = want_registry = True

    data = {}
    if want_registry:
        data["registry"] = _load_registry()
    if want_history:
        data["run_history"] = _load_run_history()
    return data


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

COMPLEXITY_PROMPT = """You are deciding how to build a Plotly chart from a user instruction.

Return ONLY a JSON object:
{"complexity": "simple" or "complex"}

"simple" = a single-series bar, line, pie, scatter, or histogram that can be built
           from a direct column mapping (e.g. "bar chart of urgency by category").
"complex" = multi-series, filtered, computed, comparative, or conditional charts
           (e.g. "compare stale count vs item count per run where hitl_approved=1").

No explanation. No markdown. Only JSON."""


SIMPLE_SPEC_PROMPT = """You are a data visualization assistant.

Given a user instruction and available dataset columns, return a JSON chart spec:

{
  "chart_type": "bar" | "line" | "scatter" | "pie" | "heatmap" | "box" | "histogram",
  "title": "descriptive chart title",
  "data_source": "registry" | "run_history",
  "x": "column name for x-axis (or labels for pie)",
  "y": "column name for y-axis (or values for pie) — null for histogram",
  "color": "column name to color-code by — or null",
  "filters": {"column": "value"} or {},
  "aggregation": "none" | "count" | "mean" | "sum" | "max"
}

Available registry columns: id, category, title, urgency, impact, days_since_update, status
Available run_history columns: run_label, run_index, date, trigger, category_filter, item_count, stale_count, hitl_approved

Rules:
- For pie charts x = label column, y = value column (or use aggregation: "count" with x as the grouping)
- For histogram, set y to null and x to the numeric column to distribute
- aggregation applies to y grouped by x
- Only use columns that exist in the listed schema
- For run_history time-series charts, ALWAYS use "run_label" as the x-axis — never use "timestamp" or "date" as x (causes axis rendering issues)
- For multi-series run_history charts, use complexity "complex" not "simple"

Return ONLY valid JSON. No explanation. No markdown."""


COMPLEX_CHART_PROMPT = """You are a Plotly chart builder. Given data and a user instruction,
return a complete Plotly Figure as a JSON object with "data" and "layout" keys.

Rules:
- Use only these trace types: bar, scatter, line (as scatter mode), pie, heatmap, box, histogram
- "data" is a list of trace dicts (each must have a "type" field)
- "layout" must include: title (with text field), paper_bgcolor, plot_bgcolor, font color
- Use dark theme: paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font color="#e6edf3"
- Grid lines: gridcolor="#21262d"
- Use colors from this palette: ["#58a6ff","#56d364","#fbbf24","#ff6b6b","#d2a8ff","#79c0ff"]
- Keep it readable — max 2-3 series, clear axis labels
- For run_history data, use "run_label" as x-axis values (e.g. "Run 1  03-10 14:32") — never use raw timestamps or "date" as x
- x/y axis: color="#8b949e", gridcolor="#21262d"

Return ONLY the JSON figure dict. No explanation. No markdown. No code fences."""


# ---------------------------------------------------------------------------
# Tier 1: Simple spec → deterministic Plotly build
# ---------------------------------------------------------------------------

ALLOWED_TYPES = {"bar", "scatter", "line", "pie", "heatmap", "box", "histogram"}

def _build_from_spec(spec: dict, data: dict):
    """Build a go.Figure from a simple LLM spec dict."""
    import plotly.graph_objects as go
    import pandas as pd

    src      = spec.get("data_source", "registry")
    rows     = data.get(src, [])
    if not rows:
        return None, f"No data available for source '{src}'"

    df = pd.DataFrame(rows)

    # Apply filters
    for col, val in (spec.get("filters") or {}).items():
        if col in df.columns:
            df = df[df[col] == val]

    if df.empty:
        return None, "Filters returned no rows."

    chart_type  = spec.get("chart_type", "bar")
    if chart_type not in ALLOWED_TYPES:
        chart_type = "bar"

    x_col  = spec.get("x")
    y_col  = spec.get("y")
    color  = spec.get("color")
    agg    = spec.get("aggregation", "none")
    title  = spec.get("title", "Chart")

    PALETTE = ["#58a6ff","#56d364","#fbbf24","#ff6b6b","#d2a8ff","#79c0ff"]
    DARK    = dict(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                   font=dict(color="#e6edf3", family="IBM Plex Mono"),
                   title=dict(text=title, font=dict(color="#e6edf3")),
                   xaxis=dict(color="#8b949e", gridcolor="#21262d"),
                   yaxis=dict(color="#8b949e", gridcolor="#21262d"),
                   margin=dict(l=40, r=20, t=50, b=40))

    # Aggregation
    if agg != "none" and x_col in df.columns:
        if agg == "count":
            df = df.groupby(x_col).size().reset_index(name=y_col or "count")
            y_col = y_col or "count"
        elif y_col and y_col in df.columns:
            fn = {"mean": "mean", "sum": "sum", "max": "max"}.get(agg, "mean")
            df = df.groupby(x_col)[y_col].agg(fn).reset_index()

    fig = go.Figure()

    if chart_type == "bar":
        if color and color in df.columns:
            for i, grp_val in enumerate(df[color].unique()):
                sub = df[df[color] == grp_val]
                fig.add_trace(go.Bar(x=sub[x_col], y=sub[y_col],
                                     name=str(grp_val),
                                     marker_color=PALETTE[i % len(PALETTE)]))
        else:
            fig.add_trace(go.Bar(x=df[x_col], y=df[y_col],
                                 marker_color=PALETTE[0]))

    elif chart_type in ("line", "scatter"):
        mode = "lines+markers" if chart_type == "line" else "markers"
        if color and color in df.columns:
            for i, grp_val in enumerate(df[color].unique()):
                sub = df[df[color] == grp_val]
                fig.add_trace(go.Scatter(x=sub[x_col], y=sub[y_col],
                                         mode=mode, name=str(grp_val),
                                         marker=dict(color=PALETTE[i % len(PALETTE)])))
        else:
            fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col],
                                     mode=mode, marker=dict(color=PALETTE[0])))

    elif chart_type == "pie":
        fig.add_trace(go.Pie(labels=df[x_col], values=df[y_col] if y_col else None,
                             marker=dict(colors=PALETTE)))

    elif chart_type == "histogram":
        fig.add_trace(go.Histogram(x=df[x_col], marker_color=PALETTE[0]))

    elif chart_type == "box":
        fig.add_trace(go.Box(x=df[x_col] if x_col else None,
                             y=df[y_col] if y_col else df[x_col],
                             marker_color=PALETTE[0]))

    elif chart_type == "heatmap":
        pivot = df.pivot_table(index=y_col, columns=x_col, aggfunc="size", fill_value=0)
        fig.add_trace(go.Heatmap(z=pivot.values.tolist(),
                                 x=list(pivot.columns),
                                 y=list(pivot.index),
                                 colorscale="Blues"))

    fig.update_layout(**DARK)
    return fig, None


# ---------------------------------------------------------------------------
# Tier 2: Complex — LLM returns full figure dict
# ---------------------------------------------------------------------------

def _build_complex(instruction: str, data: dict, api_key=None):
    """LLM generates full Plotly figure dict from raw data + instruction."""
    import plotly.graph_objects as go

    # Trim data to avoid token bloat — max 200 rows per source
    trimmed = {src: rows[:200] for src, rows in data.items()}

    model = get_model(api_key=api_key)
    user_msg = (
        f"Data:\n{json.dumps(trimmed, indent=2)}\n\n"
        f"Instruction: {instruction}"
    )
    response = model.invoke([
        {"role": "system", "content": COMPLEX_CHART_PROMPT},
        {"role": "user",   "content": user_msg},
    ])
    raw = response.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

    try:
        fig_dict = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"Complex chart parse error: {e}"

    # Whitelist trace types
    for trace in fig_dict.get("data", []):
        if trace.get("type") not in ALLOWED_TYPES:
            trace["type"] = "bar"

    fig = go.Figure(fig_dict)
    return fig, None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_chart(instruction: str, api_key=None) -> tuple:
    """
    Generate a Plotly figure from a plain-language instruction.

    Returns (fig, error_str) — fig is None on failure.
    """
    try:
        data = _pick_datasets(instruction)

        model = get_model(api_key=api_key)

        # Step 1: classify complexity
        complexity_resp = model.invoke([
            {"role": "system", "content": COMPLEXITY_PROMPT},
            {"role": "user",   "content": instruction},
        ])
        raw_c = complexity_resp.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        complexity = json.loads(raw_c).get("complexity", "simple")

        if complexity == "complex":
            return _build_complex(instruction, data, api_key=api_key)

        # Step 2: simple — get spec
        spec_resp = model.invoke([
            {"role": "system", "content": SIMPLE_SPEC_PROMPT},
            {"role": "user",   "content": instruction},
        ])
        raw_s = spec_resp.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        spec  = json.loads(raw_s)

        return _build_from_spec(spec, data)

    except json.JSONDecodeError as e:
        return None, f"Chart spec parse error: {e}"
    except Exception as e:
        return None, f"Chart generation failed: {e}"