# HOMEBASE

### Multi-Agent Home Management System — LangGraph + Gemini

A multi-agent system built with LangGraph and Gemini 2.0 Flash demonstrating
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
orchestrator
    +-- hvac_agent        -+
    +-- plumbing_agent     |
    +-- electrical_agent   +- (parallel fan-out, one LLM call per agent)
    +-- appliance_agent    |
    +-- general_agent     -+
            |
      hitl_briefing        <- graph pauses here (interrupt_before synthesizer)
            |
       [human input]       <- approve / defer items / add notes
            |
       synthesizer         <- LLM generates narrative, appends HITL decision record
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
|   +-- registry.json             # 15 seeded home task items
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
|   +-- registry_tools.py         # get_registry(), classify_registry(), get_item_detail()
|   +-- llm_tools.py              # Gemini-backed recommendation functions
|   +-- subagent_tools.py         # Rule-based tools (reference/fallback)
|
+-- tests/
    +-- conftest.py               # Global LLM mock — no API key required for tests
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
# Clone and navigate to project root
cd homebase

# Install dependencies (including dev)
uv sync --dev

# Copy environment template and add your Gemini API key
cp .env.example .env
```

Edit `.env`:

```
GEMINI_API_KEY=AIza...
```

Get a key at: <https://aistudio.google.com/app/apikey>

---

## Running

### Streamlit UI

```bash
uv run streamlit run app.py
```

The UI provides:

- **Prompt library sidebar** — 10 pre-built trigger phrases, one-click load
- **Live agent log** — color-coded by agent type, streams in real time
- **Quadrant summary** — metric counters + full classification table
- **Recommendation cards** — tabbed HU/HI and LU/HI views with action/effort/cost
- **HITL checkpoint panel** — approve, defer specific items by ID, add notes
- **Final report** — LLM-generated narrative with HITL decision record appended

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

```bash
# Run full suite
uv run pytest

# Verbose
uv run pytest -v

# Single file
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

| Trigger | Focus |
|---|---|
| `"what needs immediate attention"` | HU/HI items only |
| `"weekly home review"` | Full registry analysis |
| `"morning briefing"` | Daily priority summary |
| `"fire and safety inspection"` | Safety-critical items |
| `"plumbing systems audit"` | Plumbing category deep-dive |
| `"electrical systems inspection"` | Electrical category deep-dive |
| `"hvac seasonal maintenance check"` | HVAC category deep-dive |
| `"appliance status review"` | Appliance category deep-dive |
| `"exterior and grounds walkthrough"` | Exterior / general items |
| `"full home assessment"` | Comprehensive run |

---

## Stakeholder Bridge

The architecture patterns here are domain-agnostic. The home registry is a
stand-in for any structured dataset requiring triage, delegation, and human oversight:

| Home Management | Enterprise Equivalent |
|---|---|
| Home task registry | Risk register / ticket backlog / compliance tracker |
| Urgency x Impact quadrant | Likelihood x Risk Score (RMA framework) |
| HU/HI -> HITL escalation | High-priority item requiring manager approval |
| Specialist subagents | Domain SME agents (security, ops, finance, legal) |
| Stale item detection | SLA breach / aging ticket detection |
| Deferral with notes | Documented risk acceptance / exception handling |
| `MemorySaver` checkpoint | Audit trail of human decisions |

---

## Dependencies

| Package | Purpose |
|---|---|
| `langgraph>=1.0.10` | Agent graph, state management, HITL checkpointing |
| `langchain-core>=0.3.0` | LangChain base primitives |
| `langchain-google-genai>=2.0.0` | Gemini model integration |
| `python-dotenv>=1.0.0` | Environment variable loading |
| `streamlit>=1.55.0` | Demo UI |
| `pytest>=9.0.2` | Test runner (dev) |

---

## Notes

- The free Gemini API tier has per-minute and daily rate limits. If you see
  a `RESOURCE_EXHAUSTED` error, wait 60 seconds and retry or check your
  daily quota at <https://aistudio.google.com>
- `tools/subagent_tools.py` contains the original rule-based recommendation
  logic from the POC and is retained as a reference/fallback
