# Running HOMEBASE

## Streamlit UI

```bash
uv run streamlit run app.py
```

---

## UI Features

### Core Run Flow

| Feature | Description |
|---|---|
| **Prompt library sidebar** | 10 pre-built trigger phrases, one-click load into command field |
| **Unified command field** | Single input handles run triggers, registry commands, and chart requests via hybrid intent routing |
| **Live agent log** | Color-coded by agent type, streams in real time during a run |
| **Quadrant summary** | Metric counters with post-run deferred count |
| **Charts** | Scatter plot, category breakdown, stale donut, score distribution |
| **Classification table** | Sortable by quadrant; deferred items dimmed post-run |
| **Recommendation cards** | Tabbed HU/HI and LU/HI views with confidence scoring |
| **HITL checkpoint panel** | Approve or defer HU/HI and LU/HI items; add notes before finalizing |
| **Final report** | Groq-generated narrative with highlighted item IDs |
| **Export PDF** | Print-ready light-theme PDF download of the final report |
| **Run history tab** | Audit trail of all runs; expandable cards with quadrant breakdown, HITL decisions, deferred items, and full report |

### AI Agents (Dashboard Expanders)

| Agent | Access | Description |
|---|---|---|
| **Predictive Quadrant Preview** | Below command field | Predicts HU/HI, HU/LI, LU/HI, or LU/LI from free-text description before any run |
| **Completeness Scorer** | Inline with Quadrant Preview | Scores description against a per-category rubric; surfaces numbered follow-up questions for missing fields |
| **AI Chart Generation** | Command field (`chart ...`) | Plain language chart requests; two-tier LLM pipeline (simple spec or full Plotly figure dict) |
| **Cross-item RCA** | Command field (`rca ...`) | Root cause analysis across full registry or scoped to a category; pattern clusters, systemic narrative, recommendations |
| **5 Whys Agent** | Command field (`5 whys ...`) | Category-based causal chain; 5-level structured chain per category; auto-triggers RCA synthesis when 2+ categories analyzed |
| **Document Intake** | `⬡ Document Intake` expander | Upload PDF/image; Gemini extracts structured fields; HITL required before registry write |
| **Spreadsheet Analytics** | `📊 Spreadsheet Analytics` expander | Upload CSV/XLSX/ODS; pandas profiling + Gemini findings; HITL registry correlation |
| **Schema Metric Discovery** | `🔬 Schema Metric Discovery` expander | Upload tabular schema or paste Mermaid ERD; Gemini surfaces metrics, derived fields, gaps, quality observations |

---

## Prompt Library

| Trigger | Behavior |
|---|---|
| `"what needs immediate attention"` | Full registry, HU/HI items only |
| `"weekly home review"` | Full registry, all quadrants |
| `"morning briefing"` | Full registry, all quadrants |
| `"fire and safety inspection"` | Electrical + appliance + general only |
| `"plumbing systems audit"` | Plumbing category only |
| `"electrical systems inspection"` | Electrical category only |
| `"hvac seasonal maintenance check"` | HVAC category only |
| `"appliance status review"` | Appliance category only |
| `"exterior and grounds walkthrough"` | General category only |
| `"full home assessment"` | Full registry, all quadrants |

---

## CLI — Interactive HITL Mode

```bash
uv run python main.py
uv run python main.py --trigger "morning briefing"
```

Pauses at the HITL checkpoint for terminal input. Useful for testing graph behavior without
the Streamlit UI.

---

## CLI — Non-Interactive Mode

```bash
uv run python main.py --no-hitl
uv run python main.py --no-hitl --trigger "plumbing audit"
```

Skips the HITL checkpoint and runs through to completion automatically.

---

## Tests

No API key required — all LLM calls are mocked via `conftest.py`.
All DB calls use an isolated in-memory SQLite instance per test.

```bash
uv run pytest
uv run pytest -v
uv run pytest tests/test_hitl.py -v
```

**554 passing tests across 16 files.**

| File | Tests | Covers |
|---|---|---|
| `conftest.py` | — | Global LLM mock + in-memory SQLite fixture (no API key needed) |
| `test_registry_tools.py` | 34 | Classification logic, boundary conditions, stale detection |
| `test_orchestrator.py` | 20 | Orchestrator node, report formatting, delegation |
| `test_graph.py` | 10 | Graph structure, full invocation, state key completeness |
| `test_subagent_tools.py` | 33 | Domain recommendation functions, category router |
| `test_subagents.py` | 19 | Subagent nodes, category filtering, result structure |
| `test_hitl.py` | 31 | HITL briefing, deferral filtering, interrupt/resume behavior |
| `test_update_agent.py` | 16 | NL interpretation, field validation, clamping, apply_update path |
| `test_chart_agent.py` | 38 | Chart spec building, complex figure dict, intent routing, analytics data |
| `test_rca_agent.py` | 45 | Data loaders, RCA output structure, confidence scoring, category scoping |
| `test_whys_agent.py` | 34 | Causal chain structure, auto-category resolution, safety keyword routing |
| `test_quadrant_preview.py` | 47 | Input guards, confidence normalization, LLM output validation, error handling |
| `test_completeness_agent.py` | 60 | Input guards, score normalization, all five rubrics, category inference |
| `test_intake_agent.py` | 53 | Input guards, doc types, confidence, field sanitization, item ID validation |
| `test_analytics_agent.py` | 55 | File dispatch, pandas profiling, LLM normalization, registry correlation |
| `test_schema_agent.py` | 54 | is_mermaid, Mermaid parsing, tabular profiling, pandas 2.x StringDtype |
