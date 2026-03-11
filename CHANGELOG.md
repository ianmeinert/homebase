# CHANGELOG

All notable changes to HOMEBASE are documented here.

---

## v1.10.0

- Cross-item Root Cause Analysis (RCA) agent via unified command field
- `tools/rca_agent.py` — single LLM call with full registry + run history context
- RCA output: pattern clusters (items grouped by shared risk factor), systemic narrative (2-3 paragraphs), prioritized recommendations
- Confidence scoring on overall RCA and per cluster — float 0.0-1.0 with rationale string
- `rca` intent added to hybrid router in `update_agent.py` — keywords: root cause, rca, analyze patterns, systemic, what's driving, common factors, pattern analysis, cross-item analysis
- Full-width RCA results panel renders below command field — cluster cards, narrative block, priority-badged recommendations
- 32 new tests in `test_rca_agent.py` — data loaders, run_rca integration, confidence validation, intent routing
- conftest.py hardened — rca_agent patches use module object (not string path) to fix Windows collection-order issue

## v1.9.0

- AI chart generation — plain language chart requests via unified command field (e.g. "chart urgency by category", "plot item count and stale count over time")
- `chart_agent.py` — two-tier LLM pipeline: simple requests return a structured spec built deterministically into a Plotly figure; complex requests (multi-series, filtered, comparative) have the LLM return a full Plotly figure dict hydrated via `go.Figure()`
- Chart intent added to hybrid router — `chart` keyword set triggers classification before run/registry heuristics
- Run history x-axis fix — `run_label` field (e.g. "Run 1  03-10 14:32") replaces raw timestamp to avoid Plotly datetime axis rendering issues
- Plotly deprecation fix — all `use_container_width` replaced with `width="stretch"` / `width="content"`
- Command field now loads empty on page load; prompt library populates via `pending_input` session state key
- 25 new tests in `test_chart_agent.py` — spec building, complex figure dict, trace type whitelisting, intent routing
- Ingest agent (v1.9.0 scope originally) deferred pending Streamlit file widget investigation

## v1.8.0

- Unified NL command field — single input handles run triggers and registry commands via hybrid intent routing
- Hybrid intent router — heuristic first (regex + keyword), LLM fallback only for ambiguous input
- Full CRUD consolidated to dashboard; Registry tab removed
- Prompt library bug fix — selected trigger now correctly populates unified input field

## v1.7.0

- Stale items alert panel — amber callout at top of right column, sorted by stalest first
- Update agent prompt hardening — explicit status mapping rules, `in_progress` validation fix

## v1.6.0

- Item detail drawer — expandable rows in classification table with full item detail panel
- Natural language item updates — UPDATE ITEM panel with free-text instruction interpreter
- API key carried in graph state — survives `MemorySaver` checkpoint across HITL interrupt/resume

## v1.5.0

- LangSmith tracing integration — env var activation, per-run tags and metadata, sidebar status badge

## v1.4.0

- SQLite backend — `registry` and `run_history` tables in `data/homebase.db`
- Auto-seed from `registry.json` on first run
- In-memory SQLite fixture in test suite (no file I/O in tests)

## v1.3.0

- PDF export — print-ready light theme report via reportlab
- Auto-named with trigger slug and timestamp

## v1.2.0

- Run history tab — persisted audit trail of every completed run
- Expandable run cards with quadrant breakdown, HITL decisions, deferred items, full report

## v1.1.0

- Plotly charts — scatter, category bar, stale donut, score distribution
- Trigger-based category filtering (plumbing, electrical, hvac, appliance, general)
- HU/HI-only mode for immediate/urgent/critical triggers
- Post-run chart updates reflecting active (non-deferred) items
- Confidence scoring — LLM returns 0.0–1.0 per recommendation, color-coded progress bar

## v1.0.0

- Initial Groq/Llama 3.3 70B integration
- Orchestrator + quadrant classification + registry tools
- 5 specialist subagents + parallel fan-out
- HITL checkpoint + `MemorySaver` + deferral logic
- Streamlit UI + prompt library + recommendation cards
- Auto-generated item IDs with sequential numbering per category
