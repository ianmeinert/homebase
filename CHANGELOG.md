# CHANGELOG

All notable changes to HOMEBASE are documented here.

---

## v1.19.0

- **Guided Intake Flow** (`app.py` — `📋 Submit New Issue` expander) — 5-step HITL intake workflow that mirrors the VA RMA submitter checklist; all intelligence reuses existing agents with no new backend code
- **Step 1 — Describe:** Free-text description with live quadrant preview (confidence badge + rationale) and completeness scoring (score bar + numbered follow-up questions) firing as the user types; deduped to avoid redundant API calls
- **Step 2 — Duplicate Check:** `interpret_add()` extracts structured fields, then `check_duplicates()` runs TF-IDF against the registry; green "No duplicates found" or amber warning panel with all matches and similarity scores; user can review and proceed or go back
- **Step 3 — Triage:** Predicted quadrant rendered as a colored disposition badge (`HU/HI` → Immediate Action, `HU/LI` → Schedule Soon, `LU/HI` → Contingency, `LU/LI` → Defer/Idea); extracted fields (title, category, urgency, impact) displayed for review
- **Step 4 — Review & Approve (HITL):** Editable title and description fields; `✅ Approve & Submit` is the only path to registry write; `✕ Reject` discards without writing; `← Back` navigation at every step
- **Step 5 — Done:** Green confirmation with assigned item ID; `▶ Run Full Assessment` loads trigger into command field; `+ Submit Another` resets flow
- **Step indicator:** 5-column progress bar at top of expander — grey (future), amber (current), green (complete)
- **Enterprise analog:** Direct mapping to VA RMA Submitter Checklist — duplicate search (Step 2), triage criteria (Step 3), SBAR structured review (Step 4), HITL approval gate before any write (Step 4); demonstrates the "automate and AI this" vision with human always in the loop
- **No new tests** — guided intake is a UI orchestration layer over existing agents; all underlying agent logic already covered by existing test suite (619 passing)


---

## v1.18.0

- **TF-IDF Duplicate Detection** (`tools/duplicate_detector.py`) — deterministic cosine similarity check against existing open registry items before any new item is written; `TfidfVectorizer` with bigram support, sublinear TF normalization, and English stopword removal fits on the live registry corpus at call time; configurable similarity threshold (default 0.55, dual-channel title+full-text); closed items excluded from comparison by default via `status_filter`
- **`check_duplicates(title, description, threshold, status_filter)`** — returns a ranked `list[DuplicateMatch]` (item_id, title, category, status, score, score_pct); `has_duplicates()` and `top_match()` convenience wrappers provided
- **`execute_add()` integration** — duplicate check fires after `interpret_add()` extracts fields, before any `add_item()` DB write; returns `{_duplicates: [...], _fields: {...}}` when matches found; `force=True` parameter bypasses the check for explicit user override
- **`execute_command()` extended** — accepts `force_add` and `duplicate_threshold` params; result dict gains `"duplicates"` and `"pending_fields"` keys; non-add intents always return `None` for these keys
- **Duplicate warning UI** — amber-bordered panel surfaces candidate matches with item ID, title, and similarity percentage; two-button HITL: "➕ Add anyway" re-calls with `force_add=True`, "✕ Cancel" clears pending state; same HITL philosophy as document intake and analytics agents
- **`scikit-learn>=1.3.0`** added to `pyproject.toml` dependencies
- **Enterprise analog:** deduplication pipeline for intake queues (RMA, ServiceNow, Jira) — mirrors the VA RMA submitter checklist step "search for a duplicate or similar request" before creating a new ticket
- **36 new tests** (`tests/test_duplicate_detector.py`) — covers `_build_corpus_text`, `check_duplicates` (empty registry, exact match, unrelated item, score ranking, result fields, empty candidate), threshold behavior (default, low, high, 0.0, 1.0, custom), status filter (closed excluded by default, in_progress included, custom filter, empty corpus after filter), `has_duplicates`, `top_match`, and edge cases (single word, long description, special characters, single-item registry); total suite: 619 passing


---

## v1.17.0

- **Multi-provider LLM architecture** (`tools/llm_providers.py`) — provider abstraction layer supporting Groq/Llama (subagents) and Anthropic/Claude Sonnet (synthesizer); `active_provider()` detects `ANTHROPIC_API_KEY` from env or state at runtime; `get_synthesizer_model()` returns `ChatAnthropic` when a key is present, falls back to `ChatGroq` transparently; `get_subagent_model()` always uses Groq (parallel batch calls remain on the cheaper, higher-throughput provider); `provider_meta()` returns display label, model string, vendor, and brand color for sidebar rendering
- **Claude Sonnet synthesizer** — when `ANTHROPIC_API_KEY` is set, the synthesizer node routes the final action plan narrative to `claude-sonnet-4-20250514` instead of Llama 3.3 70B; subagent recommendation calls (HVAC, Plumbing, Electrical, Appliance, General) remain on Groq; provider attribution footer appended to every synthesized report (`[Synthesized by Claude Sonnet]` or `[Synthesized by Llama 3.3 70B]`)
- **`langchain-anthropic>=0.3.0`** added as a core dependency in `pyproject.toml`; `tools/llm_tools.py` `get_model()` now delegates to `get_subagent_model()` from the provider layer
- **`HombaseState` update** — `anthropic_api_key: str` field added; passed through `get_initial_state()` and `synthesizer_node`; never logged or surfaced in reports
- **Synthesizer message log** — provider selection logged at runtime: `[Synthesizer] Provider selected: CLAUDE — calling for synthesis narrative...` or `GROQ`
- **Sidebar provider status** — Anthropic API key input added below Google key; active state shown in purple (`OK Claude synthesizer active`); inactive shown as dim (`-- Claude key (synthesizer — optional)`); SYSTEM panel shows live synthesizer provider with brand color (`Claude Sonnet` in purple, `Llama 3.3 70B` in green)
- **29 new tests** (`tests/test_llm_providers.py`) — covers `active_provider()` (no key, empty string, whitespace, direct arg, env var), `is_claude_active()`, `get_synthesizer_model()` (returns `ChatAnthropic` vs `ChatGroq`, correct model names, key priority), `get_subagent_model()` (always Groq), `provider_meta()` (label, color, vendor per provider), and model constant assertions; total suite: 583 passing


---

## v1.16.0

- **Schema-Aware Metric Discovery Agent** (`tools/schema_agent.py`) — Gemini 2.5 Flash-Lite analyzes data schemas and surfaces computable metrics, derived field recommendations, data quality observations, and schema gap analysis
- **Dual input support** — accepts tabular files (CSV / XLSX / ODS) profiled via pandas, and Mermaid ERD markdown parsed into entity/field/type tables; both inputs normalize to `SchemaSource` before a single LLM call; multiple sources can be combined (e.g. CSV + ERD) in one discovery run
- **`DiscoveryReport` TypedDict** — structured output with `computable_metrics`, `derived_fields`, `quality_observations`, `schema_gaps`, `narrative`, and `confidence` (analytic maturity score 0.0–1.0)
- **Polished results panel** — tabbed layout (`📊 Metrics` / `🔧 Derived` / `⚠ Gaps` / `🔍 Quality`) with item counts per tab; summary stat pills (metrics, derived fields, gaps, critical/warning counts) above tabs; header bar with large analytic maturity % and slim colored progress bar; narrative rendered with blue left-border accent; metric cards with per-card confidence progress bar and field names highlighted in blue; derived/gap cards with colored top-border accent; quality observations with severity-tinted background + icon; markdown export via download button
- **File uploader lifecycle fix** — `on_change` callback persists file bytes to session state before button-click rerun wipes the uploader widget; empty-read guard prevents stale byte overwrite; debug warning surfaces when session state bytes are missing
- **HOMEBASE ERD** (`homebase_erd.md`) — Mermaid ERD documenting `registry` and `run_history` table schemas with field types, constraints, and relationship annotation; serves as both project documentation and a test input for the Mermaid path of the discovery agent
- **Dependency fixes** — `google-genai>=1.0.0`, `openpyxl>=3.1.0`, and `odfpy>=1.4.0` added as explicit dependencies in `pyproject.toml`; `schema_agent` updated from deprecated `google.generativeai` to the current `google.genai` SDK (same pattern as `intake_agent`); model string corrected to `gemini-2.5-flash-lite`
- **Proof of concept notice** — POC disclaimer added to `README.md` and `homebase_erd.md` clarifying that HOMEBASE has not undergone formal code review, security assessment, penetration testing, or production hardening
- **54 new tests** (`tests/test_schema_agent.py`) — covers `is_mermaid`, Mermaid type inference, `parse_mermaid` (entity extraction, field types, relationship context), `parse_tabular` (CSV/XLSX, 500-row cap, type detection, pandas 2.x StringDtype), and `discover_metrics` (mock LLM, confidence clamping, severity normalization, markdown fence stripping, truncation guard, multi-source input); total suite: 554 passing




## v1.15.0

- **Spreadsheet Analytics Agent** (`tools/analytics_agent.py`) — Gemini 2.5 Flash-Lite ingests CSV, XLSX, and ODS files (≤500 rows); pure-pandas profiling pass extracts column types, stats, value counts, and date ranges without sending raw data to the LLM; second Gemini call produces 3–8 ranked findings with normalized trend, severity, and confidence (clamped 0.0–1.0)
- **Registry correlation** — third Gemini call cross-references analytics findings against the live registry; validates item IDs before any write; never raises on empty or malformed registry
- **HITL review panel (analytics)** — per-item approval flow; proposed note pre-filled and editable; `update_item()` only called on explicit approve; no "approve all" shortcut; appends `[Analytics]` prefix to distinguish AI-proposed notes
- **`📊 Spreadsheet Analytics` expander** — wired into Dashboard tab below Document Intake; `st.file_uploader` placed outside any form to avoid lifecycle race conditions; 5-row preview + profile strip on upload; truncation warning when row count exceeds cap
- **Severity-coded metric cards** — 3-column layout; critical (red), warning (amber), info (green) color coding; trend arrows (↑↓→?); confidence bar per finding; narrative block below cards
- **Chart generation from uploaded data** — analytics DataFrame available to `chart_agent.py` via both the `📈 Chart this data` button (Option A, expander) and the unified NL command field (Option B); column name matching routes analytics data automatically when column names appear in the instruction
- **Complex chart token fix** — raw row limit in `_build_complex` reduced from 200 → 50; truncation guard attempts partial JSON recovery before failing; `COMPLEX_CHART_PROMPT` updated to cap x/y arrays at 20 points and enforce compact JSON output
- **Gemini model update** — both `intake_agent.py` and `analytics_agent.py` updated to `gemini-2.5-flash-lite`; prior strings (`gemini-3-flash-preview`, `gemini-2.0-flash`) removed
- **Streamlit deprecation fix** — all `use_container_width=True` replaced with `width="stretch"` across `app.py` (8 instances; deprecated after 2025-12-31)
- **Bug fixes (folded from post-v1.14.0)** — intent router: whys/rca keywords take priority over item ID presence; `run_whys()` accepts `item_id` for item-scoped analysis; `item_ids` normalized to `list[str]` in both RCA execution paths; defensive `str()` cast at cluster ID join in `app.py`
- **55 new tests** (`tests/test_analytics_agent.py`) — covers `load_file` dispatch, `profile_dataframe` (truncation, type detection, stats, nulls), `analyze_spreadsheet` (mock LLM, normalization, error handling), `correlate_findings` (empty registry guard, invalid ID filter, result merging); total suite: 485 passing




## v1.14.0

- **Document Intake Agent** (`tools/intake_agent.py`) — Gemini 2.0 Flash (multimodal) reads uploaded warranty documents, contractor invoices, work receipts, and inspection reports; extracts structured fields (date, contractor, cost, scope, item reference, notes); matches document to the closest registry item; proposes targeted field updates
- **HITL review panel** — proposed updates surface in a structured review panel before any registry write occurs; user can select/override the target registry item, edit the description, adjust status, and approve or discard; `update_item()` is only called after explicit approval
- **Multi-provider architecture** — introduces Gemini as a second LLM provider alongside Groq/Llama; Groq handles real-time orchestration and classification (speed-optimized), Gemini handles document understanding (multimodal-optimized); each model does what it does best, coordinated by the same LangGraph runtime
- **Document type classification** — normalizes to `warranty | invoice | receipt | inspection | unknown`; confidence scored 0.0–1.0 with color-coded bar; rationale rendered inline
- **Field sanitization** — `proposed_updates` restricted to `title`, `description`, `status`; LLM cannot propose urgency/impact/id/category changes; invalid registry item IDs cleared with confidence penalty
- **Google API key integration** — separate sidebar input (`AIza...`) for the Gemini key; displayed as muted hint when unset; does not block Groq-backed features
- **`⬡ Document Intake` expander** — wired into Dashboard tab alongside Predictive Quadrant Preview and Completeness Scorer; `st.file_uploader` accepts PDF, PNG, JPG, JPEG, WEBP; form wrapper prevents lifecycle race conditions
- **52 new tests** (`tests/test_intake_agent.py`) — covers input guards, all document types, confidence normalization, doc type normalization, field sanitization, item ID validation, markdown fence stripping, error handling, API key routing, and helper functions




## v1.13.0

- **Completeness Scorer** (`tools/completeness_agent.py`) — Groq/Llama 3.3 70B scores a free-text issue description against a per-category rubric (5 categories × 5 fields each); returns completeness score (0.0–1.0), list of missing/vague fields, and targeted follow-up questions
- **Per-category rubrics** — HVAC, plumbing, electrical, appliance, and general each define 5 high-value fields (symptom, location, duration, severity signals, category-specific context); rubric drives both the system prompt and the scoring logic
- **Integrated into Predictive Quadrant Preview expander** — completeness scorer fires automatically after quadrant resolves, using the same description and inferred category; no separate UI surface; renders as a labeled completeness bar + numbered follow-up question list below the quadrant badge
- **Keyword-based category inference** (`_infer_category_from_description`) — lightweight pre-LLM pass maps description to rubric category; appliance keywords checked before HVAC to prevent false matches (e.g. "dryer not heating" → appliance, not HVAC)
- **Dedup guard** — completeness call skipped if `desc + quadrant` key matches last scored input; avoids redundant API calls on re-render
- **Graceful degradation** — errors surfaced inline as a muted note; never blocks quadrant badge or crashes UI
- **Enterprise analog:** classifier-informed ticket creation assistant — predicts routing category, detects missing features that cause re-routing, prompts user to supply them before submission

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