# HOMEBASE

**Multi-Agent Home Management System — LangGraph + Groq + Gemini + Anthropic**

`v1.19.0`

!!! warning "Proof of Concept — Not Production Ready"
    HOMEBASE is a demonstration system built to illustrate multi-agent agentic AI architecture
    patterns. It has not undergone formal code review, security assessment, penetration testing,
    secrets management audit, or production hardening. It should not be deployed in a production
    environment, used to process real sensitive data, or presented as a production-grade system
    without a full security review, compliance evaluation, and architectural assessment appropriate
    to the target environment and regulatory context.

---

## What is HOMEBASE?

HOMEBASE is a multi-agent system built with LangGraph, Groq (Llama 3.3 70B), Gemini (2.5 Flash-Lite),
and Anthropic (Claude Sonnet) that demonstrates orchestrator/subagent delegation, parallel agent
execution, multi-provider LLM routing, human-in-the-loop (HITL) checkpoints, and state persistence.

The domain is **home management**. The architecture is **enterprise-transferable**.

HOMEBASE applies a quadrant classification framework to a home task registry, routes items to
specialist subagents backed by live LLM calls, and requires human approval before finalizing an
action plan. The same orchestration pattern — classification, delegation, escalation, HITL — maps
directly to risk management, service ticket triage, compliance tracking, and other enterprise
workloads.

---

## Quadrant Model

Items are scored on two dimensions (0.0–1.0 scale):

| Quadrant | Condition | Disposition |
|---|---|---|
| **HU/HI** | Urgency ≥ 0.6 AND Impact ≥ 0.6 | Immediate — HITL escalation |
| **HU/LI** | Urgency ≥ 0.6, Impact < 0.6 | Schedule soon |
| **LU/HI** | Urgency < 0.6, Impact ≥ 0.6 | Contingency plan |
| **LU/LI** | Both < 0.6 | Defer / accept |

Items with no status update in **14+ days** are flagged as stale regardless of quadrant.

---

## Quick Links

- [Setup & Installation](setup.md)
- [Running the App](running.md)
- [Architecture Overview](architecture/index.md)
- [Agent Reference](agents/index.md)
- [Enterprise Bridge](reference/enterprise.md)
- [Changelog](changelog.md)
