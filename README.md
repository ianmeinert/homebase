# HOMEBASE

### Multi-Agent Home Management System  -  LangGraph POC

A proof-of-concept multi-agent system built with LangGraph, demonstrating orchestrator/subagent delegation, parallel agent execution, human-in-the-loop (HITL) checkpoints, and state persistence. The domain is home management; the architecture is enterprise-transferable.

---

## Overview

HOMEBASE applies a **quadrant classification framework** to a home task registry, routes items to specialist subagents, and requires human approval before finalizing an action plan. The same orchestration pattern  -  classification, delegation, escalation, HITL  -  maps directly to risk management, service ticket triage, compliance tracking, and other enterprise workloads.

### Quadrant Model

Items are scored on two dimensions (0-1 scale):

| Quadrant | Condition | Disposition |
|---|---|---|
| **HU/HI** | Urgency >= 0.6 AND Impact >= 0.6 | Immediate  -  HITL escalation |
| **HU/LI** | Urgency >= 0.6, Impact < 0.6 | Schedule soon |
| **LU/HI** | Urgency < 0.6, Impact >= 0.6 | Contingency plan |
| **LU/LI** | Both < 0.6 | Defer / accept |

Items with no status update in **14+ days** are flagged as stale regardless of quadrant.

---

## Architecture

```
orchestrator
    +-- hvac_agent       -+
    +-- plumbing_agent    |
    +-- electrical_agent  +- (parallel fan-out)
    +-- appliance_agent   |
    +-- general_agent    -+
            |
      hitl_briefing        <- graph pauses here (interrupt_before synthesizer)
            |
       [human input]       <- approve / defer / notes
            |
       synthesizer         <- resumes, filters deferred items, builds final report
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
homebase_poc/
+-- app.py                        # Streamlit demo UI (Phase 4)
+-- main.py                       # CLI runner (interactive + non-interactive)
+-- pyproject.toml                # Package definition and dependencies
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
|   +-- subagent_tools.py         # Domain recommendation functions + router
|
+-- tests/
    +-- test_registry_tools.py    # 34 tests  -  classification, stale detection, boundary cases
    +-- test_orchestrator.py      # 20 tests  -  orchestrator node, report builder
    +-- test_graph.py             # 10 integration tests  -  graph structure and execution
    +-- test_subagent_tools.py    # 33 tests  -  domain recommendations, router
    +-- test_subagents.py         # 19 tests  -  subagent nodes, category filtering
    +-- test_hitl.py              # 31 tests  -  HITL briefing, deferral logic, interrupt behavior
```

---

## Setup

**Requirements:** Python 3.11+

```bash
# Clone / navigate to project root
cd homebase_poc

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

# Install with dev dependencies
pip install -e ".[dev]"
```

---

## Running

### Streamlit UI (recommended for demos)

```bash
streamlit run app.py
```

The UI provides:

- **Prompt library sidebar**  -  10 pre-built trigger phrases, one-click load
- **Live agent log**  -  color-coded by agent type, streams in real time
- **Quadrant summary**  -  metric counters + full classification table
- **Recommendation cards**  -  tabbed HU/HI and LU/HI views with action/effort/cost
- **HITL checkpoint panel**  -  approve, defer specific items by ID, add notes
- **Final report**  -  synthesized output with HITL decision summary appended

### CLI  -  Interactive HITL mode

```bash
python main.py
python main.py --trigger "morning briefing"
```

Pauses after subagent delegation, prompts for approval and optional item deferrals, then resumes to produce the final report.

### CLI  -  Non-interactive mode

```bash
python main.py --no-hitl
python main.py --no-hitl --trigger "plumbing audit"
```

Runs end-to-end without pausing. Auto-approves all HU/HI items. Useful for testing and scripted runs.

---

## Tests

```bash
# Run full suite
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_hitl.py -v
```

**131 tests across 6 files.** All tests use mocked tools  -  no external dependencies, no LLM calls required.

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

Pre-built triggers optimized for the demo:

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

The architecture patterns demonstrated here are domain-agnostic. The home registry is a stand-in for any structured dataset requiring triage, delegation, and human oversight:

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

## Phased Build Log

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Orchestrator + quadrant classification + registry tools | [DONE] Complete |
| 2 | 5 specialist subagents + parallel fan-out + synthesizer | [DONE] Complete |
| 3 | HITL checkpoint + `MemorySaver` + deferral logic | [DONE] Complete |
| 4 | Streamlit UI + prompt library + recommendation cards | [DONE] Complete |

---

## Dependencies

| Package | Purpose |
|---|---|
| `langgraph>=1.0.10` | Agent graph, state management, HITL checkpointing |
| `langchain-core>=0.3.0` | LangChain base primitives |
| `streamlit>=1.55.0` | Demo UI |
| `pytest>=9.0.2` | Test runner (dev) |

> **Note:** No LLM API calls are made in this POC. All agent logic is rule-based and deterministic. To add live LLM reasoning, add `langchain-anthropic>=0.3.0` to dependencies and replace the tool functions with Claude-backed chains.
