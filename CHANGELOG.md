# CHANGELOG

All notable changes to HOMEBASE are documented here.

---

## v1.12.0

- **Predictive Quadrant Preview** (`tools/quadrant_preview.py`) — Groq/Llama 3.3 70B predicts urgency×impact quadrant (HU/HI, HU/LI, LU/HI, LU/LI) from a free-text issue description before any agent run is triggered
- **Inline preview badge** — collapsible expander below the command field renders the predicted quadrant badge, a confidence percentage bar (color-coded green/amber/red), and a one-sentence rationale; powered by `on_change` callback to avoid unnecessary API calls
- **Deduplication guard** — preview skips the LLM call if the input hasn't changed since the last prediction (compares against `qp_input` in session state)
- **Graceful degradation** — errors surface inline without crashing the UI; badge renders only when a valid quadrant is returned
- **Enterprise analog:** ticket severity/routing prediction before submission — reduces SME group misassignment in high-volume intake pipelines

## v1.11.0

- **5 Whys causal chain agent** (`tools/whys_agent.py`) — operates directly on registry items for a given category; no prior RCA dependency. Builds a structured 5-level causal chain (each "because" becomes the next "why"), produces a root cause statement, corrective action, and confidence score with rationale
- **Correct RCA flow** — 5 Whys is the root cause method; RCA synthesis aggregates results. Data flow: `registry items → 5 Whys (per category) → whys_results[] → (auto) RCA synthesis → rca_result`
- **RCA synthesis mode** (`run_rca_synthesis()` in `rca_agent.py`) — synthesizes multiple 5 Whys results into a cross-category narrative with pattern clusters and recommendations. Auto-triggers when 2+ valid 5 Whys results exist in session
- **Safety/fire keyword resolution** — `extract_rca_category()` now recognizes safety intent keywords (`fire`, `safety`, `fire risk`, `smoke`, `carbon monoxide`, `hazard`, `risk`, etc.) and resolves to the highest-urgency open category via DB query, enabling natural queries like "5 whys on the fire safety cluster"
- **Auto-category fallback** — `_highest_severity_category()` in `whys_agent.py` selects the category with the highest average `urgency × impact` among open items when no category is specified
- **Stacked 5 Whys UI panels** — each category run appends to `whys_results` list in session state; panels stack per category with problem statement, cascading indented chain cards, root cause callout, and corrective action side-by-side
- **Confidence rationale layout fix** — rationale text now renders on its own line below the badge percentage, eliminating overflow in the flex row
- **Sample documents PDF** (`data/homebase_documents.pdf`) — 16 realistic but fictional home management documents (5 warranties, 6 work invoices, 5 parts receipts) across fictional vendors; registry ID labels removed so document intake agent must reason its own mappings
- 34 new tests in `test_whys_agent.py` — guards, auto-category resolution, chain structure, error handling, category loader, safety keyword routing, classify_input integration
- 13 new synthesis tests in `test_rca_agent.py` — empty/errored input guards, output structure, synthesized_from tracking, item count aggregation, error handling
- `conftest.py` updated — `tools.whys_agent` patched globally alongside `rca_agent`

## v1.10.0

- Cross-item RCA agent (`tools/rca_agent.py`) — single LLM call returning pattern clusters, systemic narrative, prioritized recommendations, and overall confidence score with rationale
- Category-scoped RCA — natural language category extraction (`extract_rca_category`) plus UI dropdown selector; re-runs analysis on scope change
- `updated_at` timestamp column added to registry schema — `days_since_update` now computed on read from actual timestamp; every write sets `updated_at = datetime('now')`
- Auto-migration on startup — existing DBs back-filled from `days_since_update` integer values
- Registry seed expanded to 30 items across 5 categories (6 per category); mix of open/in_progress/closed statuses and realistic staleness distribution
- `scripts/seed_run_history.py` — inserts synthetic run history records with realistic quadrant arcs, staleness trends, and HITL decisions; supports `--clear` and `--count` flags
- 45 new tests in `test_rca_agent.py` — data loaders, output structure, confidence scoring, category scoping, intent routing
- `chart_agent.py` updated to compute `days_since_update` via `julianday()` expression
- Category prefixes corrected to HV/PLB/EL across registry tools

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
