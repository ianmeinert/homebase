# Agent Reference

## LLM-Driven vs Rule-Based

Understanding where LLM reasoning is applied versus deterministic logic is important for
cost modeling, debugging, and enterprise justification.

### LLM / Agent-Driven

| Component | Module | Model | Calls per Invocation |
|---|---|---|---|
| Quadrant classification | `orchestrator.py` | Groq / Llama 3.3 70B | 1 per run |
| Specialist recommendations | `subagents.py` (×5 parallel) | Groq / Llama 3.3 70B | 1 per agent per run |
| Synthesis narrative | `orchestrator.py` — synthesizer node | Groq / Llama 3.3 70B | 1 per run |
| Confidence scoring | `llm_tools.py` — embedded in recommendation call | Groq / Llama 3.3 70B | Part of subagent call |
| Intent routing (ambiguous) | `update_agent.py` — `classify_input()` | Groq / Llama 3.3 70B | 0–1 per command |
| Chart complexity classification | `chart_agent.py` | Groq / Llama 3.3 70B | 1 per chart request |
| Chart spec / figure generation | `chart_agent.py` | Groq / Llama 3.3 70B | 1 per chart request |
| Registry command interpretation | `update_agent.py` | Groq / Llama 3.3 70B | 1–2 per command |
| Cross-item RCA | `rca_agent.py` | Groq / Llama 3.3 70B | 1 per RCA request |
| 5 Whys causal chain | `whys_agent.py` | Groq / Llama 3.3 70B | 1 per category analyzed |
| Predictive quadrant preview | `quadrant_preview.py` | Groq / Llama 3.3 70B | 1 per preview (deduped) |
| Completeness scoring | `completeness_agent.py` | Groq / Llama 3.3 70B | 1 per score (deduped) |
| Document intake | `intake_agent.py` | Gemini 2.5 Flash-Lite | 1 per uploaded document |
| Spreadsheet analytics + correlation | `analytics_agent.py` | Gemini 2.5 Flash-Lite | 2 per uploaded file |
| Schema metric discovery | `schema_agent.py` | Gemini 2.5 Flash-Lite | 1 per discovery run |

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

---

## Agent Pages

- [Core Agents](core.md) — Orchestrator, subagents, HITL briefing, synthesizer
- [Analytics Agents](analytics.md) — Document intake, spreadsheet analytics, schema discovery
- [Analysis Agents](rca.md) — RCA, 5 Whys, quadrant preview, completeness scorer
- [Chart Agent](chart.md) — AI chart generation, intent routing
