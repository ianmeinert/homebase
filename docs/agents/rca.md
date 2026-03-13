# Analysis Agents

## Cross-Item RCA Agent

**Module:** `tools/rca_agent.py`  
**Model:** Groq / Llama 3.3 70B  
**Introduced:** v1.10.0

Performs root cause analysis across the full registry or scoped to a single category.
A single LLM call returns:

- **Pattern clusters** — groups of related items sharing a common cause
- **Systemic narrative** — prose explaining the underlying systemic issue
- **Prioritized recommendations** — ranked action items
- **Confidence score** — 0.0–1.0 with rationale

**Category scoping** — natural language category extraction (`extract_rca_category`) plus
a UI dropdown selector. Re-runs analysis on scope change.

**Invocation:** `rca [optional: category]` in the unified command field, or the dropdown.

**Enterprise analog:** Systemic root cause analysis across work item categories — identifies
common failure modes across a ticket backlog or risk register.

---

## 5 Whys Agent

**Module:** `tools/whys_agent.py`  
**Model:** Groq / Llama 3.3 70B  
**Introduced:** v1.11.0

Builds a structured 5-level causal chain from registry items for a given category.
Each "because" becomes the next "why" — producing a root cause statement, corrective
action, and confidence score with rationale.

### Data Flow

```
registry items → 5 Whys (per category) → whys_results[]
                                               ↓
                              (auto) RCA synthesis → rca_result
                              (triggers when 2+ valid whys results exist)
```

**Safety keyword resolution** — recognizes safety intent keywords (`fire`, `smoke`,
`carbon monoxide`, `hazard`, `risk`, etc.) and resolves to the highest-urgency open
category via DB query. Enables natural queries like `"5 whys on the fire safety cluster"`.

**Auto-category fallback** — `_highest_severity_category()` selects the category with
the highest average `urgency × impact` among open items when no category is specified.

**UI** — stacked panels per category; cascading indented chain cards; root cause callout
and corrective action side-by-side.

**Invocation:** `5 whys [optional: category]` in the unified command field.

---

## Predictive Quadrant Preview

**Module:** `tools/quadrant_preview.py`  
**Model:** Groq / Llama 3.3 70B  
**Introduced:** v1.12.0

Predicts the urgency × impact quadrant (HU/HI, HU/LI, LU/HI, LU/LI) from a free-text
issue description **before** any agent run is triggered.

Renders inline below the command field as:

- Predicted quadrant badge
- Confidence percentage bar (color-coded green/amber/red)
- One-sentence rationale

**Dedup guard** — LLM call is skipped if the input hasn't changed since the last
prediction (compares against `qp_input` in session state).

**Enterprise analog:** Ticket severity/routing prediction before submission — reduces
SME group misassignment in high-volume intake pipelines.

---

## Completeness Scorer

**Module:** `tools/completeness_agent.py`  
**Model:** Groq / Llama 3.3 70B  
**Introduced:** v1.13.0

Scores a free-text issue description against a per-category rubric. Returns:

- Completeness score (0.0–1.0)
- List of missing or underspecified fields
- Numbered follow-up questions targeting the gaps

### Per-Category Rubrics

Each of the five categories defines 5 high-value fields:

| Category | Key Rubric Fields |
|---|---|
| HVAC | Symptom, location, duration, temperature context, last service date |
| Plumbing | Symptom, location, duration, water damage extent, shut-off valve status |
| Electrical | Symptom, location, circuit/breaker status, intermittent vs persistent, safety risk |
| Appliance | Symptom, appliance model/age, error codes, last maintenance, warranty status |
| General | Symptom, location, duration, weather/seasonal context, previous attempts |

**Keyword-based category inference** — lightweight pre-LLM pass maps description to rubric
category. Appliance keywords checked before HVAC to prevent false matches
(e.g. `"dryer not heating"` → appliance, not HVAC).

**Integration** — fires automatically after quadrant preview resolves, using the same
description and inferred category. Renders as a completeness bar + numbered question list
below the quadrant badge.

**Enterprise analog:** Classifier-informed work item creation assistant — predicts routing
category, detects missing features that cause re-routing, prompts user to supply them
before submission.
