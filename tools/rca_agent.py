"""
tools/rca_agent.py  —  Cross-item Root Cause Analysis agent.

Takes the full registry + run history, performs a systemic cross-item analysis,
and returns:
  - pattern_clusters: items grouped by shared risk factors
  - narrative:        LLM-generated systemic root cause analysis (2-3 paragraphs)
  - recommendations:  prioritized corrective actions
  - confidence:       0.0-1.0 overall confidence score with per-cluster scores

Single LLM call — not a graph node. Triggered via unified command field.
"""

import json
from tools.llm_tools import get_model
from tools.db import get_conn

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_full_registry(category: str | None = None) -> list[dict]:
    """Load all registry items regardless of status. Optionally filter by category."""
    import datetime
    conn = get_conn()
    if category:
        rows = conn.execute(
            "SELECT id, category, title, description, urgency, impact, "
            "updated_at, status FROM registry WHERE category = ?",
            (category,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, category, title, description, urgency, impact, "
            "updated_at, status FROM registry"
        ).fetchall()
    keys = ["id", "category", "title", "description", "urgency",
            "impact", "updated_at", "status"]
    records = []
    for r in rows:
        d = dict(zip(keys, r))
        try:
            updated = datetime.datetime.fromisoformat(d["updated_at"])
            d["days_since_update"] = (datetime.datetime.now() - updated).days
        except (ValueError, TypeError):
            d["days_since_update"] = 0
        records.append(d)
    return records


def _load_run_history(limit: int = 10, category: str | None = None) -> list[dict]:
    """Load most recent run history records. Optionally filter to runs matching category."""
    conn = get_conn()
    if category:
        rows = conn.execute(
            "SELECT run_id, timestamp, trigger, item_count, stale_count, "
            "hitl_approved, quadrant_summary FROM run_history "
            "WHERE category_filter LIKE ? OR category_filter IS NULL "
            "ORDER BY timestamp DESC LIMIT ?",
            (f'%{category}%', limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT run_id, timestamp, trigger, item_count, stale_count, "
            "hitl_approved, quadrant_summary FROM run_history "
            "ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()
    keys = ["run_id", "timestamp", "trigger", "item_count",
            "stale_count", "hitl_approved", "quadrant_summary"]
    records = []
    for r in rows:
        d = dict(zip(keys, r))
        d["timestamp"] = d["timestamp"][:16] if d["timestamp"] else ""
        try:
            d["quadrant_summary"] = json.loads(d["quadrant_summary"] or "{}")
        except Exception:
            d["quadrant_summary"] = {}
        records.append(d)
    return records


# ---------------------------------------------------------------------------
# RCA prompt
# ---------------------------------------------------------------------------

RCA_SYSTEM_PROMPT = """You are an expert root cause analysis (RCA) agent performing a
cross-item systemic analysis of a home management registry.

Your task is to identify systemic patterns, underlying root causes, and prioritized
corrective actions across ALL items — not item-by-item analysis.

You will receive:
- Full registry: all items with urgency, impact, staleness, category, status
- Run history: recent analysis runs showing trends in item count, stale count, quadrant distribution

Return ONLY a JSON object with this exact structure:

{
  "confidence": <float 0.0-1.0, overall confidence in this analysis>,
  "confidence_rationale": "<one sentence explaining confidence level>",
  "pattern_clusters": [
    {
      "cluster_id": "<short slug, e.g. deferred_maintenance>",
      "label": "<human-readable cluster name>",
      "risk_factor": "<the shared systemic risk factor driving this cluster>",
      "item_ids": ["<id1>", "<id2>", ...],
      "severity": "critical" | "high" | "moderate" | "low",
      "confidence": <float 0.0-1.0, confidence for this specific cluster>
    }
  ],
  "narrative": "<2-3 paragraph systemic root cause analysis. Identify the underlying drivers, not just symptoms. Reference specific patterns and data points. Connect clusters to each other where causally related. Be specific — name categories, urgency levels, stale counts.>",
  "recommendations": [
    {
      "priority": 1,
      "action": "<specific corrective action>",
      "rationale": "<why this addresses root cause, not just symptoms>",
      "addresses_clusters": ["<cluster_id1>", ...],
      "urgency": "immediate" | "short_term" | "long_term"
    }
  ]
}

Rules:
- Clusters should group items by SHARED ROOT CAUSE, not just category
- Minimum 2, maximum 6 clusters
- Recommendations must address root causes, not just re-list items
- Minimum 3, maximum 6 recommendations, ordered by priority (1 = highest)
- Confidence score reflects data quality and pattern strength:
  - < 0.5: sparse data, few items, no run history
  - 0.5-0.7: moderate data, some patterns visible
  - 0.7-0.9: clear patterns, sufficient history
  - > 0.9: strong convergent evidence across multiple data sources
- Return ONLY valid JSON. No explanation, no markdown, no preamble."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_rca(instruction: str, category: str | None = None, api_key=None) -> dict:
    """
    Perform a cross-item root cause analysis.

    Returns a dict with keys:
      clusters, narrative, recommendations, confidence,
      confidence_rationale, error, item_count, run_count
    """
    registry = _load_full_registry(category=category)
    history  = _load_run_history(limit=10, category=category)

    if not registry:
        scope_msg = f" in the '{category}' category" if category else ""
        return {
            "clusters": [], "narrative": "", "recommendations": [],
            "confidence": 0.0, "confidence_rationale": "No registry data available.",
            "error": f"No items found{scope_msg} — add items before running RCA.",
            "item_count": 0, "run_count": 0,
            "category": category,
        }

    # Build context payload — trim descriptions to avoid token bloat
    registry_trimmed = []
    for item in registry:
        registry_trimmed.append({
            "id":               item["id"],
            "category":         item["category"],
            "title":            item["title"],
            "description":      (item["description"] or "")[:120],
            "urgency":          item["urgency"],
            "impact":           item["impact"],
            "days_since_update": item["days_since_update"],
            "updated_at":       item.get("updated_at", ""),
            "status":           item["status"],
        })

    scope_label = f"Category scope: {category}" if category else "Category scope: ALL categories"
    user_msg = (
        f"{scope_label}\n\n"
        f"Registry ({len(registry)} items):\n{json.dumps(registry_trimmed, indent=2)}\n\n"
        f"Run history ({len(history)} recent runs):\n{json.dumps(history, indent=2)}\n\n"
        f"User instruction: {instruction}"
    )

    try:
        model    = get_model(api_key=api_key)
        response = model.invoke([
            {"role": "system", "content": RCA_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ])
        raw = response.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        result = json.loads(raw)
        
        clusters = result.get("pattern_clusters", [])
        for cl in clusters:
            if "item_ids" in cl:
                cl["item_ids"] = [str(i) for i in cl["item_ids"]]

        return {
            "clusters":             result.get("pattern_clusters", []),
            "narrative":            result.get("narrative", ""),
            "recommendations":      result.get("recommendations", []),
            "confidence":           float(result.get("confidence", 0.0)),
            "confidence_rationale": result.get("confidence_rationale", ""),
            "error":                None,
            "item_count":           len(registry),
            "run_count":            len(history),
            "category":             category,
        }

    except json.JSONDecodeError as e:
        return {
            "clusters": [], "narrative": "", "recommendations": [],
            "confidence": 0.0, "confidence_rationale": "",
            "error": f"RCA parse error: {e}",
            "item_count": len(registry), "run_count": len(history),
            "category": category,
        }
    except Exception as e:
        return {
            "clusters": [], "narrative": "", "recommendations": [],
            "confidence": 0.0, "confidence_rationale": "",
            "error": f"RCA failed: {e}",
            "item_count": len(registry), "run_count": len(history),
            "category": category,
        }


# ---------------------------------------------------------------------------
# Synthesis mode — aggregate multiple 5 Whys results into cross-category RCA
# ---------------------------------------------------------------------------

RCA_SYNTHESIS_PROMPT = """You are an expert root cause analysis facilitator synthesizing
multiple 5 Whys analyses across different home maintenance categories into a unified
cross-category root cause analysis.

You will receive a list of completed 5 Whys results, each covering a different category.
Each result includes: category, problem statement, causal chain, root cause, and
corrective action.

Your task:
1. Identify systemic patterns that appear across multiple categories
2. Cluster the per-category root causes into cross-cutting themes
3. Write a unified narrative connecting the patterns
4. Produce prioritized cross-category recommendations

Return ONLY a JSON object with this exact structure:

{
  "confidence": <float 0.0-1.0>,
  "confidence_rationale": "<one sentence>",
  "pattern_clusters": [
    {
      "cluster_id": "<short slug>",
      "label": "<human-readable cluster name>",
      "risk_factor": "<shared systemic risk factor>",
      "item_ids": [],
      "categories": ["<cat1>", "<cat2>"],
      "severity": "critical" | "high" | "moderate" | "low",
      "confidence": <float 0.0-1.0>
    }
  ],
  "narrative": "<2-3 paragraph synthesis. Connect root causes across categories. Identify the meta-pattern driving issues across the home. Be specific.>",
  "recommendations": [
    {
      "priority": 1,
      "action": "<cross-category corrective action>",
      "rationale": "<why this addresses the systemic root cause>",
      "addresses_clusters": ["<cluster_id>"],
      "urgency": "immediate" | "short_term" | "long_term"
    }
  ]
}

Rules:
- Minimum 2, maximum 5 clusters
- Clusters must represent CROSS-CATEGORY patterns, not just re-state individual category findings
- Minimum 3, maximum 5 recommendations, ordered by priority
- Return ONLY valid JSON. No explanation, no markdown, no preamble."""


def run_rca_synthesis(whys_results: list[dict], api_key=None) -> dict:
    """
    Synthesize multiple 5 Whys results into a cross-category RCA.

    Args:
        whys_results: list of dicts from run_whys() — one per category
        api_key: optional Groq API key

    Returns same shape as run_rca() for drop-in compatibility with the UI.
    """
    _empty = {
        "clusters": [], "narrative": "", "recommendations": [],
        "confidence": 0.0, "confidence_rationale": "",
        "error": None, "item_count": 0, "run_count": 0,
        "category": None, "synthesized_from": [],
    }

    valid = [r for r in whys_results if not r.get("error") and r.get("root_cause")]
    if not valid:
        return {**_empty, "error": "No valid 5 Whys results to synthesize."}

    # Summarize each whys result for the prompt
    summaries = [
        {
            "category":          r["category"],
            "problem_statement": r.get("problem_statement", ""),
            "root_cause":        r["root_cause"],
            "corrective_action": r.get("corrective_action", ""),
            "confidence":        r.get("confidence", 0.0),
            "item_count":        r.get("item_count", 0),
        }
        for r in valid
    ]

    user_msg = (
        f"5 Whys results to synthesize ({len(summaries)} categories):\n"
        f"{json.dumps(summaries, indent=2)}"
    )

    try:
        model = get_model(api_key=api_key)
        response = model.invoke([
            {"role": "system", "content": RCA_SYNTHESIS_PROMPT},
            {"role": "user",   "content": user_msg},
        ])
        raw = response.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        result = json.loads(raw)
        
        clusters = result.get("pattern_clusters", [])
        for cl in clusters:
            if "item_ids" in cl:
                cl["item_ids"] = [str(i) for i in cl["item_ids"]]

        total_items = sum(r.get("item_count", 0) for r in valid)

        return {
            "clusters":             result.get("pattern_clusters", []),
            "narrative":            result.get("narrative", ""),
            "recommendations":      result.get("recommendations", []),
            "confidence":           float(result.get("confidence", 0.0)),
            "confidence_rationale": result.get("confidence_rationale", ""),
            "error":                None,
            "item_count":           total_items,
            "run_count":            len(valid),
            "category":             None,
            "synthesized_from":     [r["category"] for r in valid],
        }

    except json.JSONDecodeError as e:
        return {**_empty, "error": f"RCA synthesis parse error: {e}",
                "synthesized_from": [r["category"] for r in valid]}
    except Exception as e:
        return {**_empty, "error": f"RCA synthesis failed: {e}",
                "synthesized_from": [r["category"] for r in valid]}