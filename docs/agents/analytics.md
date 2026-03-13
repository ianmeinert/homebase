# Analytics Agents

## Document Intake Agent

**Module:** `tools/intake_agent.py`  
**Model:** Gemini 2.5 Flash-Lite (multimodal)  
**Introduced:** v1.14.0

Accepts uploaded PDF, PNG, JPG, JPEG, or WEBP files. Extracts structured fields from
warranty documents, contractor invoices, work receipts, and inspection reports.

### Pipeline

1. File uploaded via `⬡ Document Intake` expander in the Dashboard tab
2. Gemini classifies document type: `warranty | invoice | receipt | inspection | unknown`
3. Structured fields extracted: date, contractor, cost, scope, item reference, notes
4. Document matched to closest registry item by content reasoning
5. Proposed field updates surface in HITL review panel
6. User selects/overrides target item, edits fields, approves or discards
7. `update_item()` called only after explicit approval

### Key Design Decisions

- **Field sanitization** — `proposed_updates` restricted to `title`, `description`, `status`.
  The LLM cannot propose changes to urgency, impact, ID, or category.
- **Invalid item ID guard** — registry item IDs not found in the DB are cleared with a
  confidence penalty before the HITL panel renders.
- **Confidence scoring** — document type confidence 0.0–1.0, color-coded bar in the UI.
- **Google API key** — entered separately in the sidebar (`AIza...`). Does not block
  Groq-backed features when absent.

---

## Spreadsheet Analytics Agent

**Module:** `tools/analytics_agent.py`  
**Model:** Gemini 2.5 Flash-Lite  
**Introduced:** v1.15.0

Accepts CSV, XLSX, and ODS files (≤500 rows). Uses a two-pass architecture: pandas
profiling first, then LLM analysis — raw data is never sent to the LLM.

### Pipeline

1. File uploaded via `📊 Spreadsheet Analytics` expander
2. **Profiling pass** — pandas extracts column types, stats, value counts, date ranges
3. **Analysis call** — Gemini receives the profile (not raw data); returns 3–8 ranked findings
   with normalized trend, severity, and confidence (clamped 0.0–1.0)
4. **Correlation call** — second Gemini call cross-references findings against the live registry;
   validates item IDs before surfacing in the HITL panel
5. **HITL review** — per-item approval; proposed note pre-filled and editable; `update_item()`
   only called on explicit approve; notes prefixed with `[Analytics]`

### Chart Integration

The analytics DataFrame is available to `chart_agent.py` via:

- **Option A** — `📈 Chart this data` button within the analytics expander
- **Option B** — Unified NL command field; column name matching routes analytics data automatically

---

## Schema Metric Discovery Agent

**Module:** `tools/schema_agent.py`  
**Model:** Gemini 2.5 Flash-Lite  
**Introduced:** v1.16.0

Analyzes data schemas and surfaces computable metrics, derived field recommendations,
data quality observations, and schema gaps. Accepts tabular files or Mermaid ERD markdown.

### Inputs

| Input Type | Format | Processing |
|---|---|---|
| Tabular file | CSV, XLSX, ODS | pandas profile → `SchemaSource` TypedDict |
| Mermaid ERD | Pasted markdown text | `parse_mermaid()` → entity/field/type table → `SchemaSource` |

Multiple sources can be combined (e.g. CSV + ERD) in a single discovery run — all sources
normalize to `SchemaSource` before a single Gemini call.

### Output — `DiscoveryReport`

| Field | Description |
|---|---|
| `computable_metrics` | 4–8 metrics, each with confidence score and contributing fields |
| `derived_fields` | 3–6 recommended derived fields with formulas/logic |
| `quality_observations` | Severity-tagged observations (`info` / `warning` / `critical`) |
| `schema_gaps` | 3–6 missing fields or relationships that would unlock additional metrics |
| `narrative` | Prose summary of the schema's analytic potential |
| `confidence` | Analytic maturity score 0.0–1.0 |

### Results Panel

The UI renders results in a tabbed layout:

- **`📊 Metrics`** — metric cards with per-card confidence progress bar and highlighted field names
- **`🔧 Derived`** — derived field cards with purple top-border accent
- **`⚠ Gaps`** — gap cards with amber top-border accent
- **`🔍 Quality`** — observations with severity-tinted background and icon

Summary stat pills above the tabs show at-a-glance counts. A header bar displays the entity
name and analytic maturity percentage. The `⬇ Export Report` button downloads the full
report as a `.md` file.
