# Enterprise Bridge

HOMEBASE uses home management as the problem domain. The architecture, patterns, and
agent behaviors are directly transferable to enterprise workloads.

---

## Concept Map

| HOMEBASE | Enterprise Equivalent |
|---|---|
| Home task registry | Risk register / service ticket backlog / compliance tracker |
| Urgency × Impact quadrant | Likelihood × Impact scoring framework |
| HU/HI → HITL escalation | High-priority item requiring human approval |
| Specialist subagents (HVAC, plumbing, etc.) | Domain SME agents (security, ops, finance, legal) |
| Stale item detection (`days_since_update >= 14`) | SLA breach / aging ticket detection |
| Deferral with notes | Documented risk acceptance / exception handling |
| `MemorySaver` checkpointer | Audit trail of human decisions |
| SQLite backend | Persistent, portable state store |
| LangSmith tracing | Audit trail of model reasoning for stakeholder/compliance validation |
| Cross-item RCA | Systemic root cause analysis across work item categories |
| Category-scoped RCA | Domain-targeted root cause analysis |
| 5 Whys causal chain | Structured RCA interview workflow |
| Confidence scoring | Model uncertainty quantification for stakeholder trust |
| Predictive quadrant preview | Ticket severity/routing prediction before submission |
| Completeness scorer | Classifier-informed work item creation assistant |
| Document intake agent | Attachment scraping and structured data extraction |
| Spreadsheet analytics agent | Tabular data analysis with registry correlation |
| Schema metric discovery | Schema-aware agent for metric potential and gap analysis |
| Multi-provider architecture | Provider-agnostic deployment for constrained environments (FedRAMP, ATO) |
| TF-IDF duplicate detection | Deduplication pipeline for intake queues (RMA, ServiceNow, Jira) |
| Guided intake flow | Structured ticket submission workflow with AI triage and HITL approval gate (mirrors VA RMA Submitter Checklist) |

---

## Transferable Patterns

### Orchestrator → Subagent Fan-Out

The parallel fan-out pattern maps directly to any domain where a triage step determines
which specialist agent(s) should handle an item. Examples:

- IT service desk: classify ticket → route to network, security, identity, or app support agent
- Compliance: classify finding → route to legal, audit, privacy, or controls agent
- Healthcare: classify case → route to billing, clinical, scheduling, or referral agent

### HITL Checkpoint

The `interrupt_before` + `MemorySaver` pattern provides a reusable, auditable human
approval gate. Any node in any graph can be interrupted before execution, with full
state preserved across the pause. The approval decision becomes part of the permanent
run record.

### Multi-Provider Routing

The same LangGraph topology supports any combination of LLM providers. This is directly
applicable to constrained environments (FedRAMP, government ATO, air-gapped) where
specific providers are authorized for specific data sensitivity tiers.
