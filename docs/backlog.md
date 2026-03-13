# HOMEBASE — Feature Backlog

Tracked features, extensions, and research ideas for the HOMEBASE multi-agent POC.
Items are organized by priority tier and mapped to their enterprise analog where applicable.

---

## In Progress

_Nothing currently in active development._

---

## Next

_Nothing currently in active development._

---

> **Superseded by v1.15.0 Spreadsheet Analytics Agent.** Bulk import from tabular files remains a potential future extension if registry write-back from analytics findings needs to expand beyond notes.



## Architecture Notes

### Multi-Provider Strategy

HOMEBASE intentionally demonstrates a **multi-provider, multi-model agentic architecture:**

| Provider | Model | Role | Justification |
|---|---|---|---|
| Groq | Llama 3.3 70B | Orchestration, classification, RCA, registry commands | Low latency, high throughput for real-time agent workflows |
| Gemini | 2.5 Flash-Lite | Document intake, spreadsheet analytics, multimodal understanding | Native PDF/image support; strong extraction and data analysis performance |
| Anthropic | Claude Sonnet / Opus | Long-document reasoning, instruction-sensitive writing, complex agentic chains | Superior context recall at scale (200K tokens), strong instruction adherence, reduced off-script behavior in multi-step chains; well-suited for synthesis nodes and narrative generation |
| OpenAI | o3 / o3-mini | Structured reasoning over schemas, math-heavy scoring, competitive benchmarks | Chain-of-thought reasoning with explicit steps; strong performance on structured inference tasks (SWE-bench, HumanEval); natural fit for schema metric discovery and scoring rubric agents |

No single model is optimal for all tasks. LangGraph as the orchestration layer enables
provider-agnostic routing — the same graph topology works regardless of which LLM is
behind each node. The current active providers are Groq and Gemini; Anthropic (Claude)
and OpenAI (o3-series) are identified as future provider candidates based on their
respective strengths in long-context synthesis and structured chain-of-thought reasoning.

### Enterprise Analog Map

| HOMEBASE Feature | Enterprise Equivalent |
|---|---|
| Registry item | Risk register item / service ticket / compliance finding |
| Urgency × Impact quadrant | Likelihood × Impact scoring framework |
| HU/HI → HITL escalation | High-priority item requiring human approval |
| Specialist subagents | Domain SME agents (security, ops, finance, legal) |
| Stale item detection | SLA breach / aging ticket detection |
| Cross-item RCA | Systemic root cause analysis across work item categories |
| 5 Whys agent | Structured RCA interview workflow |
| Predictive quadrant preview | Ticket severity/routing prediction before submission |
| Completeness scorer | Classifier-informed work item creation assistant |
| Document intake agent | Attachment scraping and structured data extraction |
| Schema metric discovery | Schema-aware agent identifies computable metrics, gaps, and derived fields from CSV or ERD input |
| Confidence scoring | Model uncertainty quantification for stakeholder trust |
| LangSmith tracing | Audit trail of model reasoning for compliance validation |
| Multi-provider architecture | Provider-agnostic deployment pattern for constrained environments |

---

## Completed

| Version | Feature |
|---|---|
| v1.0.0 | Groq/Llama 3.3 70B integration, orchestrator, 5 subagents, HITL |
| v1.1.0 | Plotly charts, trigger filtering, confidence scoring |
| v1.2.0 | Run history tab |
| v1.3.0 | PDF export |
| v1.4.0 | SQLite backend |
| v1.5.0 | LangSmith tracing |
| v1.6.0 | Item detail drawer, NL item updates, API key in graph state |
| v1.7.0 | Stale items alert panel |
| v1.8.0 | Unified NL command field, hybrid intent router |
| v1.9.0 | AI chart generation agent (two-tier), unified command field chart routing |
| v1.10.0 | Cross-item RCA agent, category-scoped RCA, `updated_at` timestamp schema, registry seed expanded to 30 items, run history seed script |
| v1.11.0 | 5 Whys causal chain agent (category-based), RCA synthesis mode, safety keyword category resolution, stacked whys UI panels, sample documents PDF |
| v1.12.0 | Predictive Quadrant Preview (inline badge, confidence bar, Groq/Llama) |
| v1.13.0 | Completeness Scorer + Prompt Agent (per-category rubrics, follow-up questions, integrated into Predictive Quadrant Preview expander) |
| v1.14.0 | Document Intake Agent (Gemini 2.0 Flash multimodal, HITL registry updates, PDF + image support, Google API key integration) |
| v1.15.0 | Spreadsheet Analytics Agent (Gemini 2.5 Flash-Lite, pandas profiling, HITL registry correlation), chart generation from uploaded data (Option A + B), complex chart token fix, Streamlit deprecation fix, post-v1.14.0 bug fixes |
| v1.16.0 | Schema-Aware Metric Discovery Agent (Gemini 2.5 Flash-Lite, CSV + Mermaid ERD input, computable metrics, derived fields, quality observations, schema gaps), HOMEBASE ERD, POC disclaimer, dependency fixes |
