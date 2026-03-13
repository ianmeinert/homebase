# HOMEBASE — Entity Relationship Diagram

> **Proof of Concept Notice**
> This diagram and the HOMEBASE system it represents are provided as a proof of concept for
> demonstration purposes only. This codebase has not undergone formal code review, security
> assessment, penetration testing, or production hardening. It should not be deployed in a
> production environment or used to process real sensitive data without a full security review,
> compliance evaluation, and architectural assessment appropriate to the target environment.

```mermaid
erDiagram
    REGISTRY {
        TEXT id PK "e.g. HV-001, PLB-002"
        TEXT category "hvac | plumbing | electrical | appliance | general"
        TEXT title
        TEXT description
        REAL urgency "0.0 – 1.0"
        REAL impact "0.0 – 1.0"
        TEXT updated_at "ISO datetime — drives stale detection"
        TEXT status "open | in_progress | closed"
    }

    RUN_HISTORY {
        TEXT run_id PK "UUID"
        TEXT timestamp "ISO datetime"
        TEXT trigger "free-text phrase that initiated the run"
        TEXT category_filter "JSON array or NULL — scoped run"
        INTEGER item_count "total items evaluated"
        TEXT quadrant_summary "JSON — HU_HI / HU_LI / LU_HI / LU_LI counts"
        INTEGER stale_count "items with updated_at older than 14 days"
        INTEGER hitl_approved "0 | 1 boolean"
        TEXT hitl_notes "free-text notes added during HITL checkpoint"
        TEXT deferred_items "JSON array of item IDs deferred by human"
        TEXT summary_report "Groq-generated narrative for this run"
    }

    REGISTRY ||--o{ RUN_HISTORY : "item IDs referenced in deferred_items and quadrant_summary"
```
