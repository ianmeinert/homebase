"""
analytics_agent.py  -  Spreadsheet Analytics Agent for HOMEBASE

Accepts a CSV, XLSX, or ODS file (≤ 500 rows). Profiles the data with pandas
then makes two optional Gemini calls:

  1. analyze_spreadsheet() — extracts computable metrics, trends, and anomalies
  2. correlate_findings()  — cross-references findings against open registry items

All registry writes are the caller's responsibility after explicit HITL approval.
The agent never writes to the registry directly.

Returns:
    profile:      DataProfile — pandas-derived structural summary
    findings:     list[MetricFinding] — 3–8 LLM-identified metrics
    narrative:    str — 2–3 paragraph agent summary
    correlations: list[CorrelationMatch] — registry items related to findings
    confidence:   float 0.0–1.0
    error:        str | None

Provider: Gemini (google-genai SDK — same pattern as intake_agent.py)
"""

import io
import json
import os
import re
from typing import TypedDict

import pandas as pd
from google import genai

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROW_LIMIT = 500

_VALID_TRENDS    = {"increasing", "decreasing", "stable", "unknown"}
_VALID_SEVERITIES = {"info", "warning", "critical"}
_VALID_DTYPES    = {"numeric", "datetime", "categorical", "text"}

_SYSTEM_PROMPT_ANALYZE = (
    "You are a data analyst agent for a home maintenance tracking system. "
    "You receive a structured profile of a user-uploaded spreadsheet and identify "
    "computable metrics, trends, and anomalies relevant to home ownership: "
    "costs, vendor frequency, failure rates, seasonal patterns, budget variance, "
    "warranty utilization, repeat repairs, etc.\n\n"
    "Return ONLY valid JSON — no preamble, no markdown fences:\n"
    "{\n"
    '  "findings": [\n'
    "    {\n"
    '      "metric":   "<metric name>",\n'
    '      "value":    "<computed value with units>",\n'
    '      "trend":    "increasing|decreasing|stable|unknown",\n'
    '      "insight":  "<1-2 sentence interpretation>",\n'
    '      "severity": "info|warning|critical"\n'
    "    }\n"
    "  ],\n"
    '  "narrative":   "<2-3 paragraph summary of what this data reveals>",\n'
    '  "confidence":  <0.0-1.0>\n'
    "}\n\n"
    "Rules:\n"
    "- Minimum 3 findings, maximum 8\n"
    "- severity=critical only for anomalies requiring action (cost spikes, "
    "recurring failures, safety system gaps, vendor monopoly risk)\n"
    "- Derive values from the numeric_stats and cat_counts provided — "
    "do not invent values not present in the profile\n"
    "- confidence reflects how much insight is derivable given data quality\n"
    "- trend=unknown when direction cannot be determined from the profile alone"
)

_SYSTEM_PROMPT_CORRELATE = (
    "You are correlating data analysis findings with an active home maintenance "
    "registry. Match findings to registry items only when the relationship is "
    "substantive — not superficial keyword overlap.\n\n"
    "Return ONLY valid JSON — no preamble, no markdown fences:\n"
    "{\n"
    '  "correlations": [\n'
    "    {\n"
    '      "item_id":       "<registry ID e.g. HV-001>",\n'
    '      "item_title":    "<item title>",\n'
    '      "relevance":     "<one sentence explaining the connection>",\n'
    '      "proposed_note": "<factual, concise text to append to item description>"\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "If no substantive matches exist, return: {\"correlations\": []}"
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class ColumnProfile(TypedDict):
    name:          str
    dtype:         str        # "numeric" | "datetime" | "categorical" | "text"
    null_pct:      float
    unique_count:  int
    sample_values: list       # up to 5 as strings


class DataProfile(TypedDict):
    row_count:     int
    col_count:     int
    truncated:     bool
    original_rows: int
    columns:       list
    date_range:    str | None
    numeric_cols:  list
    date_cols:     list
    cat_cols:      list
    numeric_stats: dict       # {col: {min, max, mean, std, sum}}
    cat_counts:    dict       # {col: {value: count}} top 5 per col


class MetricFinding(TypedDict):
    metric:   str
    value:    str
    trend:    str
    insight:  str
    severity: str


class CorrelationMatch(TypedDict):
    item_id:       str
    item_title:    str
    relevance:     str
    proposed_note: str


class AnalyticsResult(TypedDict):
    profile:       dict
    findings:      list
    narrative:     str
    correlations:  list
    confidence:    float
    error:         str | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_client(api_key: str | None = None) -> genai.Client:
    key = api_key or os.environ.get("GOOGLE_API_KEY", "")
    return genai.Client(api_key=key)


def _parse_response(raw: str) -> dict:
    """Strip markdown fences and parse JSON from LLM response."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw.strip())


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _normalize_finding(f: dict) -> MetricFinding:
    trend = str(f.get("trend", "unknown")).lower().strip()
    if trend not in _VALID_TRENDS:
        trend = "unknown"
    severity = str(f.get("severity", "info")).lower().strip()
    if severity not in _VALID_SEVERITIES:
        severity = "info"
    return MetricFinding(
        metric=str(f.get("metric", "")).strip(),
        value=str(f.get("value", "")).strip(),
        trend=trend,
        insight=str(f.get("insight", "")).strip(),
        severity=severity,
    )


def _empty_result(profile: dict, error: str) -> AnalyticsResult:
    return AnalyticsResult(
        profile=profile,
        findings=[],
        narrative="",
        correlations=[],
        confidence=0.0,
        error=error,
    )


def _format_profile_for_llm(profile: dict) -> str:
    """Serialize profiled summary into a compact LLM-readable context block."""
    lines = []
    lines.append(f"Rows: {profile['row_count']} | Columns: {profile['col_count']}")
    if profile.get("truncated"):
        lines.append(f"Note: file truncated from {profile['original_rows']} rows to {ROW_LIMIT}")
    if profile.get("date_range"):
        lines.append(f"Date range: {profile['date_range']}")
    lines.append("")

    lines.append("Column profiles:")
    for col in profile.get("columns", []):
        lines.append(
            f"  {col['name']} [{col['dtype']}] "
            f"nulls={col['null_pct']:.0%} unique={col['unique_count']} "
            f"samples={col['sample_values']}"
        )

    if profile.get("numeric_stats"):
        lines.append("")
        lines.append("Numeric statistics:")
        for col, stats in profile["numeric_stats"].items():
            lines.append(
                f"  {col}: min={stats['min']} max={stats['max']} "
                f"mean={stats['mean']:.2f} std={stats['std']:.2f} sum={stats['sum']:.2f}"
            )

    if profile.get("cat_counts"):
        lines.append("")
        lines.append("Categorical top values:")
        for col, counts in profile["cat_counts"].items():
            top = ", ".join(f"{v}({n})" for v, n in counts.items())
            lines.append(f"  {col}: {top}")

    return "\n".join(lines)


def _format_registry_for_llm(registry: list[dict]) -> str:
    lines = ["Registry items:"]
    for item in registry:
        lines.append(
            f"  {item['id']} [{item['category']}] {item['title']}"
            f" — status: {item.get('status', 'open')}"
            f" — desc: {str(item.get('description', ''))[:120]}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Parse CSV, XLSX, or ODS bytes into a DataFrame.

    Args:
        file_bytes: Raw file bytes.
        filename:   Original filename — used to determine format by extension.

    Returns:
        pd.DataFrame

    Raises:
        ValueError: Unsupported file extension.
        Exception:  Parse errors from pandas (propagated to caller).
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    buf = io.BytesIO(file_bytes)

    if ext == "csv":
        return pd.read_csv(buf)
    elif ext in ("xlsx", "xls"):
        return pd.read_excel(buf, engine="openpyxl")
    elif ext == "ods":
        return pd.read_excel(buf, engine="odf")
    else:
        raise ValueError(f"Unsupported file type: .{ext}. Use CSV, XLSX, or ODS.")


def profile_dataframe(df: pd.DataFrame) -> DataProfile:
    """
    Build a structural profile of a DataFrame using pandas only — no LLM.

    Truncates to ROW_LIMIT rows if needed.

    Returns:
        DataProfile dict.
    """
    if df is None or df.empty:
        return DataProfile(
            row_count=0, col_count=0, truncated=False, original_rows=0,
            columns=[], date_range=None, numeric_cols=[], date_cols=[],
            cat_cols=[], numeric_stats={}, cat_counts={},
        )

    original_rows = len(df)
    truncated = original_rows > ROW_LIMIT
    if truncated:
        df = df.iloc[:ROW_LIMIT].copy()

    # Attempt datetime parsing on object columns that look like dates
    for col in df.select_dtypes(include=["object", "string"]).columns:
        try:
            converted = pd.to_datetime(df[col], format="mixed", dayfirst=False)
            if converted.notna().sum() > len(df) * 0.5:
                df[col] = converted
        except Exception:
            pass

    numeric_cols = list(df.select_dtypes(include=["number"]).columns)
    date_cols    = list(df.select_dtypes(include=["datetime", "datetimetz"]).columns)

    # Categorical: object/bool cols that aren't datetime, with low cardinality
    cat_cols = []
    for col in df.select_dtypes(include=["object", "string", "bool", "category"]).columns:
        if col not in date_cols:
            cat_cols.append(col)

    # Date range
    date_range = None
    if date_cols:
        all_dates = pd.concat([df[c] for c in date_cols]).dropna()
        if not all_dates.empty:
            date_range = f"{all_dates.min().date()} → {all_dates.max().date()}"

    # Column profiles
    columns = []
    for col in df.columns:
        if col in numeric_cols:
            dtype = "numeric"
        elif col in date_cols:
            dtype = "datetime"
        elif col in cat_cols:
            dtype = "categorical"
        else:
            dtype = "text"

        null_pct     = float(df[col].isna().mean())
        unique_count = int(df[col].nunique())
        sample_vals  = [str(v) for v in df[col].dropna().unique()[:5]]

        columns.append(ColumnProfile(
            name=col,
            dtype=dtype,
            null_pct=round(null_pct, 4),
            unique_count=unique_count,
            sample_values=sample_vals,
        ))

    # Numeric stats
    numeric_stats = {}
    for col in numeric_cols:
        s = df[col].dropna()
        if not s.empty:
            numeric_stats[col] = {
                "min":  round(float(s.min()), 4),
                "max":  round(float(s.max()), 4),
                "mean": round(float(s.mean()), 4),
                "std":  round(float(s.std()), 4) if len(s) > 1 else 0.0,
                "sum":  round(float(s.sum()), 4),
            }

    # Categorical top-5 value counts
    cat_counts = {}
    for col in cat_cols:
        vc = df[col].value_counts().head(5)
        cat_counts[col] = {str(k): int(v) for k, v in vc.items()}

    return DataProfile(
        row_count=len(df),
        col_count=len(df.columns),
        truncated=truncated,
        original_rows=original_rows,
        columns=columns,
        date_range=date_range,
        numeric_cols=numeric_cols,
        date_cols=date_cols,
        cat_cols=cat_cols,
        numeric_stats=numeric_stats,
        cat_counts=cat_counts,
    )


def analyze_spreadsheet(
    profile: DataProfile,
    api_key: str | None = None,
) -> AnalyticsResult:
    """
    Call Gemini to extract metric findings and narrative from a DataProfile.

    Args:
        profile:  DataProfile from profile_dataframe().
        api_key:  Google API key. Falls back to GOOGLE_API_KEY env var.

    Returns:
        AnalyticsResult with findings, narrative, confidence populated.
        correlations is always [] — call correlate_findings() separately.
        Never raises — errors captured in error field.
    """
    if not profile or profile.get("row_count", 0) == 0:
        return _empty_result(profile or {}, "Empty profile — no data to analyze.")

    try:
        client = _get_client(api_key=api_key)
        profile_text = _format_profile_for_llm(profile)

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[_SYSTEM_PROMPT_ANALYZE, profile_text],
        )

        parsed = _parse_response(response.text)

        # Normalize findings
        raw_findings = parsed.get("findings", [])
        if not isinstance(raw_findings, list):
            raw_findings = []
        findings = [_normalize_finding(f) for f in raw_findings if isinstance(f, dict)]
        # Clamp count: 3 min (if possible), 8 max
        findings = findings[:8]

        # Confidence
        try:
            confidence = _clamp(float(parsed.get("confidence", 0.5)))
        except (TypeError, ValueError):
            confidence = 0.5

        return AnalyticsResult(
            profile=profile,
            findings=findings,
            narrative=str(parsed.get("narrative", "")).strip(),
            correlations=[],
            confidence=confidence,
            error=None,
        )

    except json.JSONDecodeError as e:
        return _empty_result(profile, f"Analytics parse error: {e}")
    except Exception as e:
        return _empty_result(profile, f"{type(e).__name__}: {e}")


def correlate_findings(
    result: AnalyticsResult,
    registry: list[dict],
    api_key: str | None = None,
) -> AnalyticsResult:
    """
    Call Gemini to cross-reference analytics findings against registry items.

    Args:
        result:   AnalyticsResult from analyze_spreadsheet().
        registry: Current registry item list.
        api_key:  Google API key. Falls back to GOOGLE_API_KEY env var.

    Returns:
        Updated AnalyticsResult with correlations populated.
        Never raises — errors appended to error field.
    """
    if not registry:
        return AnalyticsResult(**{**result, "correlations": []})

    if not result.get("findings"):
        return AnalyticsResult(**{**result, "correlations": []})

    try:
        client = _get_client(api_key=api_key)

        findings_text = "\n".join(
            f"- {f['metric']}: {f['value']} ({f['severity']}) — {f['insight']}"
            for f in result.get("findings", [])
        )
        narrative_text = result.get("narrative", "")
        registry_text  = _format_registry_for_llm(registry)

        user_msg = (
            f"Analysis narrative:\n{narrative_text}\n\n"
            f"Findings:\n{findings_text}\n\n"
            f"{registry_text}"
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[_SYSTEM_PROMPT_CORRELATE, user_msg],
        )

        parsed = _parse_response(response.text)

        raw_correlations = parsed.get("correlations", [])
        if not isinstance(raw_correlations, list):
            raw_correlations = []

        valid_ids = {item["id"] for item in registry}
        correlations = []
        for c in raw_correlations:
            if not isinstance(c, dict):
                continue
            item_id = str(c.get("item_id", "")).strip()
            if item_id not in valid_ids:
                continue
            correlations.append(CorrelationMatch(
                item_id=item_id,
                item_title=str(c.get("item_title", "")).strip(),
                relevance=str(c.get("relevance", "")).strip(),
                proposed_note=str(c.get("proposed_note", "")).strip(),
            ))

        return AnalyticsResult(**{**result, "correlations": correlations})

    except json.JSONDecodeError as e:
        existing_error = result.get("error") or ""
        suffix = f"Correlation parse error: {e}"
        return AnalyticsResult(**{
            **result,
            "correlations": [],
            "error": f"{existing_error}; {suffix}".lstrip("; "),
        })
    except Exception as e:
        existing_error = result.get("error") or ""
        suffix = f"{type(e).__name__}: {e}"
        return AnalyticsResult(**{
            **result,
            "correlations": [],
            "error": f"{existing_error}; {suffix}".lstrip("; "),
        })