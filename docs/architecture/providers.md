# Multi-Provider Strategy

HOMEBASE intentionally demonstrates a **multi-provider, multi-model agentic architecture**.
No single model is optimal for all tasks. LangGraph as the orchestration layer enables
provider-agnostic routing — the same graph topology works regardless of which LLM backs each node.

---

## Provider Map

| Provider | Model | Role | Justification |
|---|---|---|---|
| **Groq** | Llama 3.3 70B | Subagents (HVAC, Plumbing, Electrical, Appliance, General), orchestration, classification, RCA, registry commands | Low latency, high throughput for parallel batch recommendation calls |
| **Anthropic** | Claude Sonnet | Synthesizer node — final action plan narrative (activated by `ANTHROPIC_API_KEY`) | Superior instruction adherence and narrative quality for the high-visibility synthesis step; graceful fallback to Groq when key is absent |
| **Gemini** | 2.5 Flash-Lite | Document intake, spreadsheet analytics, schema discovery, multimodal understanding | Native PDF/image support; strong extraction and data analysis performance |
| **OpenAI** *(future)* | o3 / o3-mini | Structured reasoning over schemas, math-heavy scoring | Chain-of-thought reasoning with explicit steps; natural fit for schema metric discovery and scoring rubric agents |

The active providers are **Groq**, **Anthropic (Claude)**, and **Gemini**.

---

## Synthesizer Provider Selection

The synthesizer node uses a runtime provider abstraction (`tools/llm_providers.py`) that
selects between Claude Sonnet and Groq/Llama based on key availability:

```python
# tools/llm_providers.py
def get_synthesizer_model(groq_api_key=None, anthropic_api_key=None):
    anthropic_key = _resolve_anthropic_key(anthropic_api_key)
    if anthropic_key:
        return ChatAnthropic(model="claude-sonnet-4-20250514", ...)
    return ChatGroq(model="llama-3.3-70b-versatile", ...)
```

The sidebar SYSTEM panel shows the active synthesizer provider in real time.
Every synthesized report includes a `[Synthesized by <Provider>]` attribution footer.

**Why subagents stay on Groq:** The 5 specialist subagents run in parallel and each
make a single batch LLM call. Groq's throughput advantage matters here. Claude is
reserved for the synthesizer — the single high-visibility narrative generation step
where instruction quality and writing style have the most impact.

---

## Provider-Agnostic Routing

Adding a new provider requires only a new **node-level model binding** — the graph topology,
state schema, HITL flow, and tool layer are unchanged. This is the core architectural claim
HOMEBASE is designed to demonstrate.

```python
# Swapping the synthesizer to a different provider requires only:
llm = ChatAnthropic(model="claude-sonnet-4-20250514", api_key=state["anthropic_api_key"])
# or
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=state["groq_api_key"])
```

---

## LLM Call Budget

**Typical calls per full run:** 7–8

| Source | Provider | Calls |
|---|---|---|
| Orchestrator (classification) | Groq | 1 |
| Specialist subagents (×5 parallel) | Groq | 5 |
| Synthesizer (post-HITL narrative) | Claude or Groq | 1 |
| Command routing (ambiguous input only) | Groq | 0–1 |

Dashboard agents (intake, analytics, schema discovery, RCA, chart, etc.) each add 1–2 calls
per invocation but are triggered independently of the main run flow.
