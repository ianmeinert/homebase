# HOMEBASE

### Multi-Agent Home Management System — LangGraph + Groq

### v1.4.0

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
|   +-- registry.json             # Seed data — 15 home task items (read once on DB init)
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
|   +-- subagent_tools.py         # Rule-based tools (reference/fallback)
|
+-- tests/
    +-- conftest.py               # Global LLM mock + in-memory SQLite fixture (no API key needed)
    +-- test_registry_tools.py    # 34 tests — classification, stale detection, boundary cases
    +-- test_orchestrator.py      # 20 tests — orchestrator node, report builder
    +-- test_graph.py             # 10 integration tests — graph structure and execution
    +-- test_subagent_tools.py    # 33 tests — domain recommendation functions, router
    +-- test_subagents.py         # 19 tests — subagent nodes, category filtering
    +-- test_hitl.py              # 31 tests — HITL briefing, deferral logic, interrupt/resume
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
- **Registry editor tab** — add, edit, close items; live table with quadrant, status, and stale indicators
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

**131 tests across 6 files.**

| File | Tests | Covers |
|---|---|---|
| `test_registry_tools.py` | 34 | Classification logic, boundary conditions, stale detection |
| `test_orchestrator.py` | 20 | Orchestrator node, report formatting, delegation |
| `test_graph.py` | 10 | Graph structure, full invocation, state key completeness |
| `test_subagent_tools.py` | 33 | Domain recommendation functions, category router |
| `test_subagents.py` | 19 | Subagent nodes, category filtering, result structure |
| `test_hitl.py` | 31 | HITL briefing, deferral filtering, interrupt/resume behavior |

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
| Urgency x Impact quadrant | Likelihood x Risk Score (RMA framework) |
| HU/HI -> HITL escalation | High-priority item requiring manager approval |
| Specialist subagents | Domain SME agents (security, ops, finance, legal) |
| Stale item detection | SLA breach / aging ticket detection |
| Deferral with notes | Documented risk acceptance / exception handling |
| `MemorySaver` checkpoint | Audit trail of human decisions |
| SQLite backend | Persistent, portable state store |

---

## Completed Features

- [x] Orchestrator + quadrant classification + registry tools
- [x] 5 specialist subagents + parallel fan-out
- [x] HITL checkpoint + `MemorySaver` + deferral logic (HU/HI + LU/HI)
- [x] Streamlit UI + prompt library + recommendation cards
- [x] Groq/Llama 3.3 70B LLM integration (subagents + synthesizer)
- [x] Trigger-based category filtering (plumbing, electrical, hvac, appliance, general)
- [x] HU/HI-only mode for immediate/urgent/critical triggers
- [x] Plotly charts — scatter, category bar, stale donut, score distribution
- [x] Charts update post-run to reflect active (non-deferred) items
- [x] Deferred items dimmed in classification table with `[deferred]` label
- [x] Final report rendered as styled prose (word-wrapped, item IDs highlighted)
- [x] Registry editor tab — add, edit, and close items from the UI
- [x] Auto-generated item IDs with sequential numbering per category
- [x] Run history tab — persisted audit trail of every completed run with HITL decisions
- [x] Export report — download final report as print-ready PDF
- [x] Confidence scoring — LLM returns 0.0–1.0 confidence per recommendation, displayed as color-coded progress bar
- [x] SQLite backend — `registry` and `run_history` tables in `data/homebase.db`; auto-seeded from `registry.json` on first run; in-memory DB fixture in tests
- [x] 131-test suite with global LLM mock and isolated DB per test

## Planned Features

- [ ] **LangSmith tracing** — one env var enables full visual trace of agent execution
- [ ] **Item detail drawer** — click any item in the classification table to expand full details inline
- [ ] **Stale items alert panel** — dedicated callout at top of run, not just a badge

---

## Version History

| Version | Changes |
|---|---|
| v1.4.0 | SQLite backend for registry and run history; in-memory DB test fixture |
| v1.3.0 | PDF export (print-ready light theme, reportlab) |
| v1.2.0 | Run history tab with audit trail |
| v1.1.0 | Plotly charts, trigger-based filtering, post-run chart updates, confidence scoring |
| v1.0.0 | Initial Groq/Llama integration replacing Gemini |

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
