# HOMEBASE — Feature Backlog

Tracked features, extensions, and research ideas for the HOMEBASE multi-agent POC.
Items are organized by priority tier and mapped to their enterprise analog where applicable.

---

## In Progress

_Nothing currently in active development._

---

## Next

| Version | Feature | Description | Enterprise Analog |
|---|---|---|---|
| v1.20.0 | Registry Bulk Operations Agent | NL bulk updates across multiple registry items — "close all plumbing items", "mark all stale as in-progress"; extends current single-item update agent | Batch ticket processing, bulk status normalization |

---

## Backlog

| Version | Feature | Description | Enterprise Analog |
|---|---|---|---|
| v1.20.0 | Memory / Context Agent | Orchestrator references past run decisions from `run_history` — "this item has been HU/HI for 3 consecutive runs", "last time we saw this pattern we deferred it"; episodic memory layered on existing `run_history` table | Institutional memory for recurring risk patterns, time-to-resolution tracking |
| v1.20.0 | Export / Reporting Agent | Structured output beyond current PDF — markdown, JSON summary, email-ready HTML; trend analysis over run history (HU/HI frequency, deferral rates, stale item trajectory) | Executive reporting, compliance documentation, RMA trend dashboards |
| v1.20.0 | Scheduled / Triggered Runs | Background scheduler auto-runs orchestrator on configurable cadence (daily briefing, weekly review) without manual UI input; notification panel surfaces results; demonstrates agentic autonomy beyond reactive chat | Automated risk/compliance monitoring with scheduled escalation |
| future | OpenAI o3 Integration | Wire o3 / o3-mini as provider for schema metric discovery and scoring rubric agents; chain-of-thought reasoning for structured inference tasks | Provider-agnostic deployment; structured reasoning over schemas and scoring rubrics |

---

## Architecture Notes

### Multi-Provider Strategy

HOMEBASE intentionally demonstrates a **multi-provider, multi-model agentic architecture:**

| Provider | Model | Role | Justification |
|---|---|---|---|
| Groq | Llama 3.3 70B | Subagents (HVAC, Plumbing, Electrical, Appliance, General), orchestration, classification, RCA, registry commands | Low latency, high throughput for parallel batch recommendation calls |
| Anthropic | Claude Sonnet | Synthesizer node — final action plan narrative (activated by `ANTHROPIC_API_KEY`) | Superior instruction adherence and narrative quality for the high-visibility synthesis step; graceful fallback to Groq when key is absent |
| Gemini | 2.5 Flash-Lite | Document intake, spreadsheet analytics, schema discovery, multimodal understanding | Native PDF/image support; strong extraction and data analysis performance |
| OpenAI | o3 / o3-mini | Structured reasoning over schemas, math-heavy scoring *(future)* | Chain-of-thought reasoning with explicit steps; natural fit for schema metric discovery and scoring rubric agents |

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
| Schema metric discovery | Schema-aware agent for metric potential and gap analysis |
| TF-IDF duplicate detection | Deduplication pipeline for intake queues (RMA, SNOW, Jira) |
| Guided intake flow (Submit New Issue) | Structured ticket submission workflow with AI triage and HITL approval gate |
| Confidence scoring | Model uncertainty quantification for stakeholder trust |
| LangSmith tracing | Audit trail of model reasoning for compliance validation |
| Multi-provider architecture | Provider-agnostic deployment for constrained environments (FedRAMP, ATO) |

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
| v1.16.1 | MkDocs documentation site (Material theme, GitHub Pages, auto-deploy workflow), README cleanup, Node.js 24 Actions upgrade |
| v1.17.0 | Multi-provider LLM architecture (Claude Sonnet synthesizer, Groq subagents, runtime provider selection via `ANTHROPIC_API_KEY`, sidebar provider status, 29 new tests) |
| v1.18.0 | TF-IDF Duplicate Detection (`tools/duplicate_detector.py`, dual-channel TF-IDF, threshold 0.55, HITL warning UI, `execute_add` integration, 36 new tests) |
| v1.19.0 | Guided Intake Flow (`📋 Submit New Issue` expander) — 5-step HITL intake mirroring RMA checklist: Describe → Duplicate Check → Triage → Review & Approve → Done |
