# HOMEBASE — Feature Backlog

Tracked features, extensions, and research ideas for the HOMEBASE multi-agent POC.
Items are organized by priority tier and mapped to their enterprise analog where applicable.

---

## In Progress

_Nothing currently in active development._

---

## Next

### 5 Whys Follow-on Agent

**Provider:** Groq (Llama 3.3 70B)
**Effort:** Low

After an RCA run, user can select a specific pattern cluster and trigger an interactive
5 Whys drill-down. Agent asks a sequence of causal questions, builds a causal chain,
and outputs a structured root cause summary with corrective action recommendations.

**Home analog:** "Why is HVAC degrading repeatedly?" → deferred filter changes → budget
deprioritization → no scheduled maintenance cadence → no reminder system → root cause:
no preventive maintenance workflow.

**Enterprise analog:** Structured RCA interview workflow for recurring issue categories.

---

### Predictive Quadrant Preview

**Provider:** Groq (Llama 3.3 70B)
**Effort:** Low

Before committing to a full agent run, user types a rough description of an issue and
receives an LLM-predicted quadrant classification (HU/HI, HU/LI, LU/HI, LU/LI) with
a confidence score. Surfaces inline in the command field area as a preview badge.

**Home analog:** "My furnace is making a grinding noise and it's January" →
predicted HU/HI (0.91 confidence) before any run is triggered.

**Enterprise analog:** Predict ticket severity/routing before submission to reduce re-routing
and SME group misassignment.

---

## Medium

### Completeness Scorer + Prompt Agent

**Provider:** Groq (Llama 3.3 70B)
**Effort:** Medium

Agent monitors item creation in real time. Scores the in-progress description against
known high-value fields for the predicted category. If critical fields are missing or
underspecified, a chatbot surfaces targeted prompting questions before submission.

**Home analog:** User types "plumbing issue in bathroom" — agent detects missing
urgency signals, location specificity, and duration — prompts: "How long has this been
occurring?" and "Is there visible water damage?"

**Enterprise analog:** Classifier-informed ticket creation assistant. ML model predicts routing
category; agent detects missing features that cause re-routing; chatbot prompts user
to supply them before submission.

**Dependencies:** Requires a completeness rubric per category (rule-based or
LLM-derived). No training data required — LLM evaluates completeness against
category-specific criteria.

---

### Document Intake Agent

**Provider:** Gemini 1.5 Flash (multimodal)
**Effort:** Medium

User uploads warranty documents, contractor invoices, work receipts, or inspection
reports. Gemini extracts structured data (item ID, work performed, cost, date, contractor,
scope) and either creates a new registry item or updates an existing one. Supports PDF
and image input.

**Home analog:** Upload a contractor invoice for HVAC repair → agent extracts
date completed, cost, scope of work → closes HV-001, updates description with
completion notes.

**Enterprise analog:** Attachment scraping and classification. Extract structured information
from screenshots, PDFs, and uploaded documents associated with work items.

**Architecture note:** Introduces Gemini as a second LLM provider alongside Groq.
This demonstrates a **multi-provider agentic architecture** — Groq/Llama for real-time
orchestration and classification (speed-optimized), Gemini for document understanding
(multimodal-optimized). Each model doing what it does best, coordinated by LangGraph.
This is a key demo differentiator and directly supports provider-agnostic architecture
required in regulated or constrained deployment environments.

---

## Future / Research

### Schema-Aware Metric Discovery Agent

**Provider:** TBD (embedding model + vector store)
**Effort:** High

Upload one or more data schemas (field definitions, data types, table relationships)
into a RAG store. An agent reasons over the ingested schema to identify which fields
are meaningful for metrics, which are redundant or underutilized, what derived metrics
are computable from existing fields, and what schema gaps would need to be filled to
unlock additional analysis.

Output is a structured metric discovery report: computable metrics, recommended
derived fields, data quality observations, and suggested additions.

**Home analog:** Ingest the homebase SQLite schema (registry + run_history field
definitions) and have the agent surface what metrics are already derivable, what
fields are missing (e.g. cost, contractor, resolution time), and what additions
would improve RCA quality.

**Enterprise analog:** Ingest an operational data schema and have the agent analyze
fields for metric potential — identifying what can be measured today, what requires
additional instrumentation, and where data quality gaps exist.

**Dependencies:** Embedding model, vector store (e.g. ChromaDB or FAISS), schema
ingestion pipeline. LangGraph retrieval tool node pattern is well-suited for the
agent layer.

---

### File Ingest Agent (CSV / Excel / ODS)

**Provider:** Groq (Llama 3.3 70B)
**Effort:** Medium
**Origin:** Internal — deferred from v1.9.0

Bulk import registry items from tabular files. Streamlit file widget lifecycle issues
caused deferral. Recommended approach: file path input instead of `st.file_uploader`
to avoid stream reset race conditions.

---

## Architecture Notes

### Multi-Provider Strategy

HOMEBASE intentionally demonstrates a **multi-provider, multi-model agentic architecture:**

| Provider | Model | Role | Justification |
|---|---|---|---|
| Groq | Llama 3.3 70B | Orchestration, classification, RCA, registry commands | Low latency, high throughput for real-time agent workflows |
| Gemini | 1.5 Flash | Document intake, multimodal understanding | Native PDF/image support; strong extraction performance |

No single model is optimal for all tasks. LangGraph as the orchestration layer enables
provider-agnostic routing — the same graph topology works regardless of which LLM is
behind each node.

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
| Schema metric discovery | Schema-aware agent identifies computable metrics, gaps, and derived fields |
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
| v1.9.0 | AI chart generation agent (two-tier), cross-item RCA agent, category-scoped RCA, `updated_at` timestamp schema |
