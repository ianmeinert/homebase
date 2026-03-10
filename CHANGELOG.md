# CHANGELOG

All notable changes to HOMEBASE are documented here.

---

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
