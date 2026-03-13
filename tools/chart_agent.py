"""
tools/chart_agent.py  —  AI-generated charts from plain language instructions.

Two-tier approach:
  Simple  → LLM returns a structured spec (chart_type, title, x, y, filters)
            and we build the Plotly figure deterministically.
  Complex → LLM receives the full dataset + instruction and returns a complete
            Plotly Figure dict (data + layout); we hydrate go.Figure() from it.
            No exec() — pure data, not code.

Allowed trace types (whitelist): bar, scatter, line, pie, heatmap, box, histogram

Supports three data sources:
  registry     — live registry items from SQLite
  run_history  — agent run history from SQLite
  analytics    — user-uploaded spreadsheet (passed as analytics_df kwarg)
"""

import json
import pandas as pd
from tools.llm_tools import get_model
from tools.db import get_conn

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_registry() -> list[dict]:
    """Return all non-closed registry items as list of dicts."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, category, title, urgency, impact, CAST((julianday('now') - julianday(updated_at)) AS INTEGER) AS days_since_update, updated_at, status FROM registry"
    ).fetchall()
    conn.close()
    keys = ["id", "category", "title", "urgency", "impact", "days_since_update", "updated_at", "status"]
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
        ts = d.get("timestamp", "")
        if ts and len(ts) >= 16:
            d["run_label"] = f"Run {i}  {ts[5:10]} {ts[11:16]}"
        else:
            d["run_label"] = f"Run {i}"
        d["run_index"] = i
        d["date"] = ts[:10] if ts else ""
        records.append(d)
    return records


def _load_analytics_data(df: pd.DataFrame) -> list[dict]:
    """Convert an uploaded DataFrame into a list of row dicts (max 200 rows)."""
    if df is None or df.empty:
        return []
    return df.head(200).astype(str).to_dict(orient="records")


_ANALYTICS_KEYWORDS = {
    "uploaded", "spreadsheet", "my data", "my file", "csv", "xlsx", "ods",
    "from the file", "from my data", "from the spreadsheet", "from the upload",
    "the data i uploaded", "uploaded data", "uploaded file",
}


def _pick_datasets(instruction: str, analytics_df: pd.DataFrame | None = None) -> dict:
    """Heuristically decide which tables to load based on instruction keywords."""
    low = instruction.lower()
    history_kws = {"history", "run", "trend", "over time", "past", "stale count",
                   "item count", "hitl", "approved", "date", "month", "week"}
    registry_kws = {"urgency", "impact", "category", "status", "open", "closed",
                    "in progress", "quadrant", "scatter", "registry", "items"}

    # Analytics data: explicit keywords OR column name match from uploaded df
    want_analytics = any(k in low for k in _ANALYTICS_KEYWORDS)
    if not want_analytics and analytics_df is not None:
        col_names = {c.lower() for c in analytics_df.columns}
        want_analytics = bool(col_names & set(low.split()))

    want_history  = any(k in low for k in history_kws)
    want_registry = any(k in low for k in registry_kws)

    # If analytics data is available and referenced, prefer it
    if want_analytics and analytics_df is not None:
        data = {"analytics": _load_analytics_data(analytics_df)}
        # Still load registry/history if also referenced
        if want_registry:
            data["registry"] = _load_registry()
        if want_history:
            data["run_history"] = _load_run_history()
        return data

    # Default: if neither matches clearly, load both registry sources
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
  "data_source": "registry" | "run_history" | "analytics",
  "x": "column name for x-axis (or labels for pie)",
  "y": "column name for y-axis (or values for pie) — null for histogram",
  "color": "column name to color-code by — or null",
  "filters": {"column": "value"} or {},
  "aggregation": "none" | "count" | "mean" | "sum" | "max"
}

Available registry columns: id, category, title, urgency, impact, days_since_update, updated_at, status
Available run_history columns: run_label, run_index, date, trigger, category_filter, item_count, stale_count, hitl_approved
Analytics columns: use whatever columns are present in the data (described in the user message)

Rules:
- Use data_source="analytics" when the instruction references uploaded data, a spreadsheet, CSV, or specific columns from the uploaded file
- For pie charts x = label column, y = value column (or use aggregation: "count" with x as the grouping)
- For histogram, set y to null and x to the numeric column to distribute
- aggregation applies to y grouped by x
- Only use columns that exist in the listed schema
- For run_history time-series charts, ALWAYS use "run_label" as the x-axis — never use "timestamp" or "date" as x
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
- For run_history data, use "run_label" as x-axis values — never use raw timestamps
- x/y axis: color="#8b949e", gridcolor="#21262d"
- For analytics data, use the actual column names from the provided dataset
- CRITICAL: Keep x/y arrays SHORT — max 20 data points per series. Aggregate or sample if needed.
- CRITICAL: Use compact JSON — no extra whitespace in arrays. The response must be valid complete JSON.

Return ONLY the JSON figure dict. No explanation. No markdown. No code fences."""


# ---------------------------------------------------------------------------
# Tier 1: Simple spec → deterministic Plotly build
# ---------------------------------------------------------------------------

ALLOWED_TYPES = {"bar", "scatter", "line", "pie", "heatmap", "box", "histogram"}

def _build_from_spec(spec: dict, data: dict):
    """Build a go.Figure from a simple LLM spec dict."""
    import plotly.graph_objects as go

    src  = spec.get("data_source", "registry")
    rows = data.get(src, [])
    if not rows:
        return None, f"No data available for source '{src}'"

    df = pd.DataFrame(rows)

    # Apply filters
    for col, val in (spec.get("filters") or {}).items():
        if col in df.columns:
            df = df[df[col] == val]

    if df.empty:
        return None, "Filters returned no rows."

    chart_type = spec.get("chart_type", "bar")
    if chart_type not in ALLOWED_TYPES:
        chart_type = "bar"

    x_col = spec.get("x")
    y_col = spec.get("y")
    color = spec.get("color")
    agg   = spec.get("aggregation", "none")
    title = spec.get("title", "Chart")

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

    # Aggressively trim to stay well under Groq context limits.
    # For analytics data: pre-aggregate to reduce payload size.
    trimmed = {}
    for src, rows in data.items():
        if src == "analytics" and rows:
            # Send max 50 rows — enough for the LLM to understand structure
            # and build aggregated series without token overflow
            trimmed[src] = rows[:50]
        else:
            trimmed[src] = rows[:50]

    model = get_model(api_key=api_key)
    user_msg = (
        f"Data (sample — up to 50 rows shown):\n{json.dumps(trimmed, indent=2)}\n\n"
        f"Instruction: {instruction}\n\n"
        f"Important: Build the chart from the data sample provided. "
        f"Aggregate rows as needed (group by category, sum costs, etc.) "
        f"directly in the Plotly trace — do not reference data outside this sample."
    )
    response = model.invoke([
        {"role": "system", "content": COMPLEX_CHART_PROMPT},
        {"role": "user",   "content": user_msg},
    ])
    raw = response.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

    # Truncation guard — if JSON is incomplete, attempt partial recovery
    if raw and not raw.endswith("}"):
        # Try to find the last valid closing brace
        last_brace = raw.rfind("}")
        if last_brace > 0:
            raw = raw[:last_brace + 1]

    try:
        fig_dict = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"Complex chart parse error: {e}"

    for trace in fig_dict.get("data", []):
        if trace.get("type") not in ALLOWED_TYPES:
            trace["type"] = "bar"

    fig = go.Figure(fig_dict)
    return fig, None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_chart(
    instruction: str,
    api_key: str | None = None,
    analytics_df: pd.DataFrame | None = None,
) -> tuple:
    """
    Generate a Plotly figure from a plain-language instruction.

    Args:
        instruction:   Natural language chart request.
        api_key:       Groq API key for LLM calls.
        analytics_df:  Optional uploaded DataFrame — passed through to _pick_datasets()
                       so the agent can chart user-uploaded spreadsheet data.

    Returns:
        (fig, error_str) — fig is None on failure.
    """
    try:
        data = _pick_datasets(instruction, analytics_df=analytics_df)

        model = get_model(api_key=api_key)

        # Build column context for the prompt when analytics data is present
        analytics_col_context = ""
        if analytics_df is not None and not analytics_df.empty:
            col_info = ", ".join(
                f"{c} ({str(analytics_df[c].dtype)})"
                for c in analytics_df.columns
            )
            analytics_col_context = f"\n\nUploaded spreadsheet columns: {col_info}"

        instruction_with_context = instruction + analytics_col_context

        # Step 1: classify complexity
        complexity_resp = model.invoke([
            {"role": "system", "content": COMPLEXITY_PROMPT},
            {"role": "user",   "content": instruction_with_context},
        ])
        raw_c = complexity_resp.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        complexity = json.loads(raw_c).get("complexity", "simple")

        if complexity == "complex":
            return _build_complex(instruction_with_context, data, api_key=api_key)

        # Step 2: simple — get spec
        spec_resp = model.invoke([
            {"role": "system", "content": SIMPLE_SPEC_PROMPT},
            {"role": "user",   "content": instruction_with_context},
        ])
        raw_s = spec_resp.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        spec  = json.loads(raw_s)

        return _build_from_spec(spec, data)

    except json.JSONDecodeError as e:
        return None, f"Chart spec parse error: {e}"
    except Exception as e:
        return None, f"Chart generation failed: {e}"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_registry() -> list[dict]:
    """Return all non-closed registry items as list of dicts."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, category, title, urgency, impact, CAST((julianday('now') - julianday(updated_at)) AS INTEGER) AS days_since_update, updated_at, status FROM registry"
    ).fetchall()
    conn.close()
    keys = ["id", "category", "title", "urgency", "impact", "days_since_update", "updated_at", "status"]
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