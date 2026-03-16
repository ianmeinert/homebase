# Architecture Overview

## Graph Topology

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
       synthesizer         <- Claude Sonnet or Groq generates narrative (runtime provider selection)
            |
           END
```

---

## LangGraph Features Demonstrated

| Feature | Where |
|---|---|
| `StateGraph` with typed shared state (`TypedDict`) | `graph/state.py`, `graph/graph.py` |
| `groq_api_key` and `anthropic_api_key` carried in graph state — survive HITL interrupt/resume | `graph/state.py` |
| Parallel node fan-out with `Annotated[list, merge_lists]` reducer | `graph/graph.py` |
| `MemorySaver` checkpointer for state persistence across interrupt | `graph/graph.py` |
| `interrupt_before` for HITL checkpoint | `graph/graph.py` |
| `graph.update_state()` + `graph.stream(None, config)` for resume | `agents/orchestrator.py` |

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
|   +-- registry.json             # Seed data — 30 items across 5 categories
|
+-- graph/
|   +-- state.py                  # HomebaseState TypedDict schema
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
|   +-- llm_providers.py          # Multi-provider abstraction (ChatAnthropic / ChatGroq selection, provider metadata)
|   +-- llm_tools.py              # Subagent recommendation functions + confidence scoring
|   +-- tracing.py                # LangSmith tracing init, status check, per-run metadata
|   +-- update_agent.py           # NL registry update agent + intent router
|   +-- chart_agent.py            # AI chart generation (two-tier: simple spec / complex figure dict)
|   +-- rca_agent.py              # Cross-item RCA agent with category scoping
|   +-- whys_agent.py             # 5 Whys causal chain agent (category-based)
|   +-- quadrant_preview.py       # Predictive quadrant classification from free-text description
|   +-- completeness_agent.py     # Completeness scorer + follow-up question generator
|   +-- intake_agent.py           # Document Intake Agent — Gemini multimodal
|   +-- analytics_agent.py        # Spreadsheet Analytics Agent — Gemini 2.5 Flash-Lite
|   +-- schema_agent.py           # Schema Metric Discovery Agent — Gemini 2.5 Flash-Lite
|   +-- subagent_tools.py         # Rule-based tools (reference/fallback)
|
+-- scripts/
|   +-- seed_run_history.py       # Inserts synthetic run history records
|
+-- tests/
    +-- conftest.py               # Global LLM mock + in-memory SQLite fixture
    +-- test_registry_tools.py
    +-- test_orchestrator.py
    +-- test_graph.py
    +-- test_subagent_tools.py
    +-- test_subagents.py
    +-- test_hitl.py
    +-- test_update_agent.py
    +-- test_chart_agent.py
    +-- test_rca_agent.py
    +-- test_whys_agent.py
    +-- test_quadrant_preview.py
    +-- test_completeness_agent.py
    +-- test_intake_agent.py
    +-- test_analytics_agent.py
    +-- test_schema_agent.py
```
