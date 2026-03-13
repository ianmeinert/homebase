# Multi-Provider Strategy

HOMEBASE intentionally demonstrates a **multi-provider, multi-model agentic architecture**.
No single model is optimal for all tasks. LangGraph as the orchestration layer enables
provider-agnostic routing — the same graph topology works regardless of which LLM backs each node.

---

## Provider Map

| Provider | Model | Role | Justification |
|---|---|---|---|
| **Groq** | Llama 3.3 70B | Orchestration, classification, RCA, registry commands | Low latency, high throughput for real-time agent workflows |
| **Gemini** | 2.5 Flash-Lite | Document intake, spreadsheet analytics, schema discovery, multimodal understanding | Native PDF/image support; strong extraction and data analysis performance |
| **Anthropic** *(future)* | Claude Sonnet / Opus | Long-document reasoning, instruction-sensitive writing, complex agentic chains | Superior context recall at scale (200K tokens), strong instruction adherence, reduced off-script behavior in multi-step chains; well-suited for synthesis nodes and narrative generation |
| **OpenAI** *(future)* | o3 / o3-mini | Structured reasoning over schemas, math-heavy scoring | Chain-of-thought reasoning with explicit steps; strong performance on structured inference tasks; natural fit for schema metric discovery and scoring rubric agents |

The current active providers are **Groq** and **Gemini**. Anthropic (Claude) and OpenAI (o3-series)
are identified as future provider candidates based on their respective strengths.

---

## Provider-Agnostic Routing

Adding a new provider requires only a new **node-level model binding** — the graph topology,
state schema, HITL flow, and tool layer are unchanged. This is the core architectural claim
HOMEBASE is designed to demonstrate.

```python
# Groq node binding (current)
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=state["groq_api_key"])

# Swapping to Claude requires only:
llm = ChatAnthropic(model="claude-opus-4", api_key=state["anthropic_api_key"])
```

---

## LLM Call Budget

**Typical calls per full run:** 7–8

| Source | Calls |
|---|---|
| Orchestrator (classification) | 1 |
| Specialist subagents (×5 parallel) | 5 |
| Synthesizer (post-HITL narrative) | 1 |
| Command routing (ambiguous input only) | 0–1 |

Dashboard agents (intake, analytics, schema discovery, RCA, chart, etc.) each add 1–2 calls
per invocation but are triggered independently of the main run flow.
