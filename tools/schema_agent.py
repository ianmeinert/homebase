"""
Schema-Aware Metric Discovery Agent — v1.16.0

Accepts two input types:
  - CSV / XLSX / ODS  : pandas-profiled into a column schema summary
  - Mermaid ERD (.mmd / .mermaid / pasted text) : parsed into entity/field/type table

Gemini 2.5 Flash-Lite then reasons over the schema to surface:
  - Computable metrics (derivable today from existing fields)
  - Recommended derived fields (new columns that would unlock additional analysis)
  - Data quality observations (nullability, type ambiguity, redundancy)
  - Schema gap analysis (fields missing that would improve RCA, trending, scoring)

Output is a structured DiscoveryReport TypedDict.
"""

from __future__ import annotations

import io
import json
import re
from typing import TypedDict

from google import genai
from google.genai import types
import pandas as pd


# ---------------------------------------------------------------------------
# TypedDicts
# ---------------------------------------------------------------------------

class SchemaField(TypedDict):
    name: str
    data_type: str           # inferred: numeric | datetime | categorical | text | boolean | unknown
    nullable: bool
    unique_values: int | None
    notes: str               # e.g. "range 0.0–1.0", "ISO datetime", "JSON-encoded"


class SchemaSource(TypedDict):
    source_type: str         # "csv" | "xlsx" | "ods" | "mermaid"
    entity_name: str         # table/entity name or filename stem
    row_count: int | None    # None for ERD inputs
    field_count: int
    fields: list[SchemaField]
    raw_summary: str         # human-readable schema block sent to LLM


class MetricEntry(TypedDict):
    metric_name: str
    description: str
    fields_required: list[str]
    confidence: float        # 0.0–1.0


class DerivedFieldEntry(TypedDict):
    field_name: str
    description: str
    rationale: str
    source_fields: list[str]


class QualityObservation(TypedDict):
    field: str
    observation: str
    severity: str            # "info" | "warning" | "critical"


class GapEntry(TypedDict):
    suggested_field: str
    description: str
    impact: str              # what analysis it would unlock


class DiscoveryReport(TypedDict):
    entity_name: str
    source_type: str
    computable_metrics: list[MetricEntry]
    derived_fields: list[DerivedFieldEntry]
    quality_observations: list[QualityObservation]
    schema_gaps: list[GapEntry]
    narrative: str
    confidence: float


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

_MERMAID_TRIGGERS = re.compile(
    r"^\s*(erDiagram|classDiagram|graph\s|flowchart\s|sequenceDiagram)",
    re.IGNORECASE | re.MULTILINE,
)

_MERMAID_ENTITY = re.compile(r"^\s*(\w+)\s*\{", re.MULTILINE)
_MERMAID_FIELD = re.compile(
    r"^\s+(\w+)\s+(\w+)(?:\s+(PK|FK|UK))?(?:\s+\"([^\"]*)\")?\s*$",
    re.MULTILINE,
)
_MERMAID_RELATION = re.compile(
    r"^\s*(\w+)\s+[|o}{]+[-–.]+[|o}{]+\s+(\w+)\s*:\s*\"?([^\"\\n]*)\"?",
    re.MULTILINE,
)


def _infer_type_from_mermaid(raw_type: str) -> str:
    t = raw_type.upper()
    if t in ("TEXT", "VARCHAR", "CHAR", "STRING"):
        return "text"
    if t in ("INTEGER", "INT", "BIGINT", "SMALLINT"):
        return "numeric"
    if t in ("REAL", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC"):
        return "numeric"
    if t in ("BOOLEAN", "BOOL"):
        return "boolean"
    if t in ("DATE", "DATETIME", "TIMESTAMP", "TIME"):
        return "datetime"
    return "unknown"


def parse_mermaid(text: str) -> list[SchemaSource]:
    """Parse a Mermaid ERD string into one SchemaSource per entity."""
    sources: list[SchemaSource] = []

    # Find all entity blocks
    entity_blocks: dict[str, str] = {}
    for m in _MERMAID_ENTITY.finditer(text):
        entity = m.group(1)
        start = m.end()
        # Find closing brace
        depth = 1
        pos = start
        while pos < len(text) and depth > 0:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1
        entity_blocks[entity] = text[start:pos - 1]

    for entity, block in entity_blocks.items():
        fields: list[SchemaField] = []
        for fm in _MERMAID_FIELD.finditer(block):
            raw_type, name, constraint, comment = (
                fm.group(1), fm.group(2), fm.group(3) or "", fm.group(4) or ""
            )
            # Mermaid ERD format: TYPE name [constraint] ["comment"]
            # swap if first token looks like a name (lowercase) and second like a type
            # Standard mermaid is: TYPE name — so group(1)=type, group(2)=name
            fields.append(SchemaField(
                name=name,
                data_type=_infer_type_from_mermaid(raw_type),
                nullable=constraint not in ("PK",),
                unique_values=None,
                notes=f"{constraint} {comment}".strip(),
            ))

        summary_lines = [f"Entity: {entity}", f"Fields ({len(fields)}):"]
        for f in fields:
            summary_lines.append(
                f"  {f['name']} ({f['data_type']})"
                + (f" — {f['notes']}" if f['notes'] else "")
            )

        # Add relationship context
        relations = []
        for rm in _MERMAID_RELATION.finditer(text):
            a, b, label = rm.group(1), rm.group(2), rm.group(3).strip()
            if entity in (a, b):
                other = b if entity == a else a
                relations.append(f"  relates to {other}: {label}")
        if relations:
            summary_lines.append("Relationships:")
            summary_lines.extend(relations)

        sources.append(SchemaSource(
            source_type="mermaid",
            entity_name=entity,
            row_count=None,
            field_count=len(fields),
            fields=fields,
            raw_summary="\n".join(summary_lines),
        ))

    return sources


def _infer_type_from_series(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    # Handle both legacy object dtype and pandas 2.x StringDtype
    is_string_like = series.dtype == object or pd.api.types.is_string_dtype(series)
    if is_string_like:
        non_null = series.dropna()
        if len(non_null) == 0:
            return "unknown"
        # Boolean-like strings — check before datetime
        uniq_lower = set(non_null.astype(str).str.lower().unique())
        if uniq_lower.issubset({"true", "false", "yes", "no", "1", "0"}):
            return "boolean"
        # Try datetime parse
        sample = non_null.head(20).astype(str)
        try:
            pd.to_datetime(sample, format="mixed")
            return "datetime"
        except Exception:
            pass
        n_unique = series.nunique()
        n_total = len(series)
        if n_unique <= 20 or (n_unique / max(n_total, 1)) < 0.3:
            return "categorical"
        return "text"
    return "unknown"


def parse_tabular(file_bytes: bytes, filename: str) -> SchemaSource:
    """Load CSV/XLSX/ODS and profile its schema."""
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        df = pd.read_csv(io.BytesIO(file_bytes))
        src_type = "csv"
    elif ext in ("xlsx", "xls"):
        df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
        src_type = "xlsx"
    elif ext == "ods":
        df = pd.read_excel(io.BytesIO(file_bytes), engine="odf")
        src_type = "ods"
    else:
        raise ValueError(f"Unsupported file type: .{ext}. Use CSV, XLSX, or ODS.")

    # Cap at 500 rows for profiling
    capped = len(df) > 500
    if capped:
        df = df.head(500)

    entity_name = filename.rsplit(".", 1)[0].replace("_", " ").title()
    fields: list[SchemaField] = []
    summary_lines = [
        f"Entity: {entity_name}",
        f"Source: {filename} ({len(df)} rows{', capped at 500' if capped else ''})",
        f"Fields ({len(df.columns)}):",
    ]

    for col in df.columns:
        series = df[col]
        dtype = _infer_type_from_series(series)
        null_count = series.isna().sum()
        nullable = null_count > 0
        n_unique = series.nunique()

        notes_parts = []
        if dtype == "numeric":
            notes_parts.append(f"range {series.min():.2g}–{series.max():.2g}")
        if dtype == "categorical":
            top = series.value_counts().head(5).index.tolist()
            notes_parts.append(f"values: {top}")
        if nullable:
            notes_parts.append(f"{null_count} nulls")
        if dtype == "datetime":
            try:
                parsed = pd.to_datetime(series.dropna(), format="mixed")
                notes_parts.append(f"{parsed.min().date()} – {parsed.max().date()}")
            except Exception:
                pass

        notes = "; ".join(notes_parts)
        fields.append(SchemaField(
            name=col,
            data_type=dtype,
            nullable=nullable,
            unique_values=int(n_unique),
            notes=notes,
        ))
        summary_lines.append(
            f"  {col} ({dtype}, {n_unique} unique)"
            + (f" — {notes}" if notes else "")
        )

    return SchemaSource(
        source_type=src_type,
        entity_name=entity_name,
        row_count=len(df),
        field_count=len(df.columns),
        fields=fields,
        raw_summary="\n".join(summary_lines),
    )


def is_mermaid(text: str) -> bool:
    """Heuristic: does this look like Mermaid ERD text?"""
    return bool(_MERMAID_TRIGGERS.search(text))


# ---------------------------------------------------------------------------
# LLM — Gemini 2.5 Flash-Lite
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a schema analyst for a multi-agent home management system called HOMEBASE.
You receive a structured schema summary (either from a database ERD or a tabular data file)
and produce a metric discovery report.

Your job:
1. **Computable Metrics** — metrics derivable TODAY from existing fields (no new data needed).
   Include aggregations, ratios, trends, and derived scores. Be specific about which fields are required.
2. **Derived Fields** — new columns or computed values that would significantly improve analysis
   if added. Explain the rationale and what analysis they unlock.
3. **Quality Observations** — data quality issues: nullable fields that shouldn't be, type
   ambiguities, JSON-encoded columns that limit queryability, fields with low signal, redundancy.
4. **Schema Gaps** — fields entirely absent from the schema that would meaningfully improve
   RCA quality, trend analysis, or operational scoring. Be concrete.
5. **Narrative** — 3–5 sentence executive summary of the schema's analytic maturity and
   the highest-priority improvements.

Rules:
- Return ONLY a JSON object. No markdown. No code fences. No preamble.
- confidence: overall 0.0–1.0 score reflecting how analytically mature this schema is today.
- computable_metrics: 4–8 entries. Each must include fields_required (list of field names).
- derived_fields: 3–6 entries.
- quality_observations: 2–6 entries. severity must be "info", "warning", or "critical".
- schema_gaps: 3–6 entries.
- All string values must be properly escaped. Keep arrays SHORT — max 6 items per list.
- Use compact JSON — no extra whitespace in arrays."""

_SCHEMA_JSON_SHAPE = """{
  "entity_name": "...",
  "source_type": "...",
  "computable_metrics": [
    {"metric_name": "...", "description": "...", "fields_required": ["..."], "confidence": 0.0}
  ],
  "derived_fields": [
    {"field_name": "...", "description": "...", "rationale": "...", "source_fields": ["..."]}
  ],
  "quality_observations": [
    {"field": "...", "observation": "...", "severity": "info"}
  ],
  "schema_gaps": [
    {"suggested_field": "...", "description": "...", "impact": "..."}
  ],
  "narrative": "...",
  "confidence": 0.0
}"""


def discover_metrics(
    sources: list[SchemaSource],
    api_key: str,
) -> DiscoveryReport:
    """
    Run Gemini over one or more SchemaSource objects and return a DiscoveryReport.
    Multiple sources (e.g. CSV + ERD entity) are combined into one prompt.
    """
    client = genai.Client(api_key=api_key)

    # Build combined schema block
    schema_blocks = "\n\n---\n\n".join(s["raw_summary"] for s in sources)
    entity_name = " + ".join(s["entity_name"] for s in sources)
    source_type = " + ".join(sorted({s["source_type"] for s in sources}))

    user_content = (
        f"Analyze the following schema and produce a metric discovery report.\n\n"
        f"{schema_blocks}\n\n"
        f"Return a JSON object matching this shape exactly:\n{_SCHEMA_JSON_SHAPE}"
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=_SYSTEM_PROMPT + "\n\n" + user_content,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=2048,
        ),
    )

    raw = response.text.strip()
    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw).strip()

    # Truncation guard
    if raw and not raw.endswith("}"):
        last = raw.rfind("}")
        if last > 0:
            raw = raw[:last + 1]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Schema discovery parse error: {e}") from e

    # Normalize and clamp
    def _clamp(v, lo=0.0, hi=1.0):
        try:
            return max(lo, min(hi, float(v)))
        except (TypeError, ValueError):
            return 0.5

    for m in data.get("computable_metrics", []):
        m["confidence"] = _clamp(m.get("confidence", 0.5))

    severities = {"info", "warning", "critical"}
    for o in data.get("quality_observations", []):
        if o.get("severity") not in severities:
            o["severity"] = "info"

    return DiscoveryReport(
        entity_name=data.get("entity_name", entity_name),
        source_type=data.get("source_type", source_type),
        computable_metrics=data.get("computable_metrics", []),
        derived_fields=data.get("derived_fields", []),
        quality_observations=data.get("quality_observations", []),
        schema_gaps=data.get("schema_gaps", []),
        narrative=data.get("narrative", ""),
        confidence=_clamp(data.get("confidence", 0.5)),
    )