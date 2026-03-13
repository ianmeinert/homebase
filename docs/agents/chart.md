# Chart Agent

**Module:** `tools/chart_agent.py`  
**Model:** Groq / Llama 3.3 70B  
**Introduced:** v1.9.0

Generates interactive Plotly charts from plain language requests via the unified command field.

---

## Two-Tier Pipeline

```
User input → intent router → chart_agent.generate_chart()
                                         |
                               classify complexity
                                    /         \
                             simple            complex
                                |                 |
                         _build_from_spec()   _build_complex()
                                |                 |
                         structured spec      full Plotly
                         → Plotly figure      figure dict
                                |                 |
                              render            render
```

### Simple Charts

The LLM returns a structured spec (chart type, x/y fields, filters, title). The spec is
built deterministically into a Plotly figure — no Plotly code is executed from the LLM response.

### Complex Charts

For multi-series, filtered, or comparative requests, the LLM returns a complete Plotly
figure dict. Hydrated via `go.Figure()`. Row limit capped at 50 rows, x/y arrays capped
at 20 points to prevent token overflow.

---

## Intent Routing

The unified command field uses a hybrid router — heuristic first, LLM fallback for ambiguous input.

`chart` keyword set triggers chart classification before run/registry heuristics:

```
"chart urgency by category"        → chart agent (simple)
"plot item count over time"        → chart agent (complex)
"chart this data"                  → chart agent using analytics DataFrame
```

---

## Analytics Data Integration

When a spreadsheet file has been uploaded and analyzed, the analytics DataFrame is
available to the chart agent via two paths:

- **Option A** — `📈 Chart this data` button within the Spreadsheet Analytics expander
- **Option B** — Unified command field; column name matching routes analytics data automatically
  when column names from the uploaded file appear in the instruction

---

## Rule-Based Charts

The following charts are generated deterministically from graph state — no LLM involved:

| Chart | Data Source |
|---|---|
| Scatter plot (urgency × impact) | Registry items post-classification |
| Category bar chart | Item count per category |
| Stale items donut | Stale vs current item count |
| Score distribution | Urgency and impact score histograms |
| Run history trend | Quadrant counts per run (x-axis: `run_label`) |
