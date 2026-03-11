# HOMEBASE

### Multi-Agent Home Management System — LangGraph + Groq

### v1.10.0

A multi-agent system built with LangGraph and Groq (Llama 3.3 70B) demonstrating
orchestrator/subagent delegation, parallel agent execution, live LLM reasoning,
human-in-the-loop (HITL) checkpoints, and state persistence. The domain is home
management; the architecture is enterprise-transferable.

---

## Overview

HOMEBASE applies a quadrant classification framework to a home task registry,
routes items to specialist subagents backed by live LLM calls, and requires human
approval before finalizing an action plan. The same orchestration pattern —
classification, delegation, escalation, HITL — maps directly to risk management,
service ticket triage, compliance tracking, and other enterprise workloads.

### Quadrant Model

Items are scored on two dimensions (0-1 scale):

| Quadrant | Condition | Disposition |
|---|---|---|
| **HU/HI** | Urgency >= 0.6 AND Impact >= 0.6 | Immediate — HITL escalation |
| **HU/LI** | Urgency >= 0.6, Impact < 0.6 | Schedule soon |
| **LU/HI** | Urgency < 0.6, Impact >= 0.6 | Contingency plan |
| **LU/LI** | Both < 0.6 | Defer / accept |

Items with no status update in **14+ days** are flagged as stale regardless of quadrant.

---

## Architecture

```
orchestrator  (trigger-based category filter + optional HU/HI-only mode)
    +-- hvac_agent        -+
    +-- plumbing_agent     |
    +-- electrical_agent   +- (parallel fan-out, one Groq call per agent)
    +-- appliance_agent    |
    +-- general_agent     -+
            |
      hitl_briefing        <- graph pauses here (interrupt_before synthesizer)
            |
       [human input]       <- approve / defer HU/HI + LU/HI items / add notes
            |
       synthesizer         <- Groq generates narrative, appends HITL decision record
            |
           END
```

**Key LangGraph features demonstrated:**

- `groq_api_key` carried in graph state — survives HITL interrupt/resume without env var dependency
- `StateGraph` with typed shared state (`TypedDict`)
- Parallel node fan-out with `Annotated[list, merge_lists]` reducer
- `MemorySaver` checkpointer for state persistence across the interrupt
- `interrupt_before` for HITL checkpoint
- `graph.update_state()` + `graph.stream(None, config)` for resume

---

## Project Structure

```
homebase/
+-- app.py                        # Streamlit UI
+-- main.py                       # CLI runner (interactive + non-interactive)
+-- pyproject.toml                # Package definition and dependencies
+-- .env.example                  # Environment variable template
|
+-- data/
|   +-- homebase.db               # SQLite database (auto-created on first run)
|   +-- registry.json             # Seed data — 30 home task items across 5 categories (read once on DB init)
|
+-- graph/
|   +-- state.py                  # HombaseState TypedDict schema
|   +-- graph.py                  # LangGraph graph definition
|
+-- agents/
|   +-- orchestrator.py           # Orchestrator, HITL briefing, synthesizer nodes
|   +-- subagents.py              # 5 specialist subagent nodes
|
+-- tools/
|   +-- db.py                     # SQLite connection manager, schema, auto-seed
|   +-- registry_tools.py         # Registry CRUD backed by SQLite
|   +-- history_tools.py          # Run history persistence backed by SQLite
|   +-- llm_tools.py              # Groq-backed recommendation functions + confidence scoring
|   +-- tracing.py                # LangSmith tracing init, status check, per-run metadata
|   +-- update_agent.py           # Natural language registry update agent + intent router
|   +-- chart_agent.py            # AI chart generation agent (two-tier: simple spec / complex figure dict)
|   +-- rca_agent.py              # Cross-item root cause analysis agent with category scoping
|   +-- subagent_tools.py         # Rule-based tools (reference/fallback)
|
+-- scripts/
|   +-- seed_run_history.py       # Inserts synthetic run history records for demo data quality
|
+-- tests/
    +-- conftest.py               # Global LLM mock + in-memory SQLite fixture (no API key needed)
    +-- test_registry_tools.py    # 34 tests — classification, stale detection, boundary cases
    +-- test_orchestrator.py      # 20 tests — orchestrator node, report builder
    +-- test_graph.py             # 10 integration tests — graph structure and execution
    +-- test_subagent_tools.py    # 33 tests — domain recommendation functions, router
    +-- test_subagents.py         # 19 tests — subagent nodes, category filtering
    +-- test_hitl.py              # 31 tests — HITL briefing, deferral logic, interrupt/resume
    +-- test_update_agent.py      # 16 tests — NL interpretation, field validation, apply_update
    +-- test_chart_agent.py       # 25 tests — chart spec building, complex figure, intent routing
```

---

## Setup

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/)

```bash
cd homebase
uv sync --dev
cp .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=gsk_...
```

Get a key at: <https://console.groq.com>

**Database:** `data/homebase.db` is created and seeded automatically on first run.
No migration step required.

**Demo data:**
The registry seeds automatically with 30 items across 5 categories on first run.
For meaningful RCA confidence scores and trend analysis, seed synthetic run history:

```bash
uv run python scripts/seed_run_history.py
```

This inserts 100 synthetic run history records spanning 90 days with realistic
quadrant distributions, staleness trends, and HITL decisions. Run `--clear` to
reset history first, or `--count N` to control the number of records.

**LangSmith tracing (optional):**
Add to `.env` to activate:

```
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=homebase
```

Get a key at: <https://smith.langchain.com> (free tier available).
Tracing status appears in the sidebar. Each run produces a full trace with node timing,
LLM calls (prompt/response/tokens/latency), HITL state, and searchable tags.

---

## Running

### Streamlit UI

```bash
uv run streamlit run app.py
```

The UI provides:

- **Prompt library sidebar** — 10 pre-built trigger phrases, one-click load
- **Live agent log** — color-coded by agent type, streams in real time
- **Quadrant summary** — metric counters with post-run deferred count
- **Charts** — scatter plot, category breakdown, stale donut, score distribution
- **Classification table** — sortable by quadrant; deferred items dimmed post-run
- **Recommendation cards** — tabbed HU/HI and LU/HI views with confidence scoring
- **HITL checkpoint panel** — approve, defer HU/HI and LU/HI items, add notes
- **Final report** — Groq-generated narrative, word-wrapped prose with highlighted item IDs
- **Export PDF** — print-ready light-theme PDF download of the final report
- **Unified command field** — single input handles run triggers, registry commands (add, update, close), and chart requests via hybrid intent routing; Enter or click to submit
- **AI chart generation** — plain language chart requests routed to `chart_agent`; two-tier LLM pipeline (simple spec or full Plotly figure dict); renders in right column alongside rule-based charts
- **Cross-item RCA** — natural language root cause analysis across the full registry or scoped to a category; pattern clusters, systemic narrative, prioritized recommendations, and confidence scoring; category override via dropdown selector
- **Run history tab** — audit trail of all runs; expandable cards with quadrant breakdown, HITL decisions, deferred items, and full report

### CLI — Interactive HITL mode

```bash
uv run python main.py
uv run python main.py --trigger "morning briefing"
```

### CLI — Non-interactive mode

```bash
uv run python main.py --no-hitl
uv run python main.py --no-hitl --trigger "plumbing audit"
```

---

## Tests

No API key required — all LLM calls are mocked via `conftest.py`.
All DB calls use an isolated in-memory SQLite instance per test.

```bash
uv run pytest
uv run pytest -v
uv run pytest tests/test_hitl.py -v
```

**217 tests across 9 files.**

| File | Tests | Covers |
|---|---|---|
| `test_registry_tools.py` | 34 | Classification logic, boundary conditions, stale detection |
| `test_orchestrator.py` | 20 | Orchestrator node, report formatting, delegation |
| `test_graph.py` | 10 | Graph structure, full invocation, state key completeness |
| `test_subagent_tools.py` | 33 | Domain recommendation functions, category router |
| `test_subagents.py` | 19 | Subagent nodes, category filtering, result structure |
| `test_hitl.py` | 31 | HITL briefing, deferral filtering, interrupt/resume behavior |
| `test_update_agent.py` | 16 | NL interpretation, field validation, clamping, apply_update path |
| `test_chart_agent.py` | 25 | Chart spec building, complex figure dict, intent routing, data loading |
| `test_rca_agent.py` | 45 | Data loaders, RCA output structure, confidence scoring, category scoping, intent routing |

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

## Stakeholder Bridge

| Home Management | Enterprise Equivalent |
|---|---|
| Home task registry | Risk register / ticket backlog / compliance tracker |
| Urgency x Impact quadrant | Likelihood x Risk Score scoring framework |
| HU/HI -> HITL escalation | High-priority item requiring human approval |
| Specialist subagents | Domain SME agents (security, ops, finance, legal) |
| Stale item detection | SLA breach / aging ticket detection |
| Deferral with notes | Documented risk acceptance / exception handling |
| `MemorySaver` checkpoint | Audit trail of human decisions |
| SQLite backend | Persistent, portable state store |
| LangSmith tracing | Audit trail of model reasoning for stakeholder validation |
| Cross-item RCA | Systemic root cause analysis across work item categories |
| Category-scoped RCA | Domain-targeted root cause analysis |
| Confidence scoring | Model uncertainty quantification for stakeholder trust |
| Document intake agent *(backlog)* | Attachment scraping and structured data extraction |
| Multi-provider architecture *(backlog)* | Provider-agnostic deployment for constrained environments |

---

## Agent-Contributed vs Rule-Based

Understanding where LLM reasoning is applied vs deterministic logic is important
for cost modeling, debugging, and enterprise justification.

### LLM / Agent-Driven

| Component | Where | Model Calls |
|---|---|---|
| Quadrant classification | `orchestrator.py` | 1 per run |
| Specialist recommendations | `subagents.py` (×5 parallel) | 1 per agent per run |
| Synthesis narrative | `orchestrator.py` — synthesizer node | 1 per run (post-HITL) |
| Confidence scoring | `llm_tools.py` — embedded in recommendation call | part of subagent call |
| Intent routing (ambiguous) | `update_agent.py` — `classify_input()` | 0–1 per command (heuristic first) |
| Chart complexity classification | `chart_agent.py` — `generate_chart()` | 1 per chart request |
| Chart spec generation (simple) | `chart_agent.py` — `_build_from_spec()` | 1 per simple chart |
| Chart figure generation (complex) | `chart_agent.py` — `_build_complex()` | 1 per complex chart |
| Registry command interpretation | `update_agent.py` — `route_intent`, `interpret_update`, `interpret_add` | 1–2 per command |

**Typical LLM calls per full run:** 7–8 (1 orchestrator + 5 subagents + 1 synthesizer + 0–1 command routing)

### Rule-Based / Deterministic

| Component | Logic |
|---|---|
| Stale detection | `days_since_update >= 14` — no LLM |
| Trigger → category filter | Keyword map in `orchestrator.py` — no LLM |
| HU/HI-only mode | Trigger keyword list — no LLM |
| HITL deferral filtering | Set operations on item IDs — no LLM |
| Item ID generation | Sequential counter per category prefix — no LLM |
| Command intent (unambiguous) | Regex + keyword heuristic in `update_agent.py` — no LLM |
| Chart rendering | Plotly, derived from classified state — no LLM |
| PDF export | reportlab template — no LLM |

## Planned Features

See [BACKLOG.md](BACKLOG.md) for the full feature backlog including next-up items,
medium-term extensions, and future research directions.

Highlights:

- **5 Whys follow-on agent** — interactive causal drill-down from RCA cluster
- **Predictive quadrant preview** — LLM predicts HU/HI before a full run from free-text input
- **Completeness scorer** — watches item creation, prompts for missing high-value fields
- **Document intake agent** — PDF/image → extract warranty, invoice, inspection data → registry (Gemini)
- **Schema-aware metric discovery** — RAG-backed agent analyzes data schema for metric potential and gaps

---

## Version History

See [CHANGELOG.md](CHANGELOG.md).

---

## Dependencies

| Package | Purpose |
|---|---|
| `langgraph>=1.0.10` | Agent graph, state management, HITL checkpointing |
| `langchain-core>=0.3.0` | LangChain base primitives |
| `langchain-groq>=0.2.0` | Groq/Llama model integration |
| `plotly>=5.0.0` | Interactive charts |
| `reportlab>=4.0.0` | PDF report generation |
| `python-dotenv>=1.0.0` | Environment variable loading |
| `streamlit>=1.55.0` | Demo UI |
| `pytest>=9.0.2` | Test runner (dev) |

---

## Notes

- The free Groq tier has per-minute rate limits. If you see a `429` error,
  wait 60 seconds and retry. Check usage at: <https://console.groq.com>
- `tools/subagent_tools.py` contains the original rule-based recommendation
  logic and is retained as a reference/fallback.
- `data/registry.json` is only read during initial DB seed. After `homebase.db`
  exists, all registry reads/writes go through SQLite.
- `days_since_update` is computed on read from the `updated_at` timestamp column.
  Every registry write (add, update, close) sets `updated_at = datetime('now')`.
- The document intake agent (backlog) will introduce Gemini 1.5 Flash as a second
  LLM provider for multimodal document understanding, alongside Groq/Llama for
  real-time orchestration. This demonstrates a provider-agnostic multi-model
  architecture — each model used where it performs best.
