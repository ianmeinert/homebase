# Core Agents

## Orchestrator

**Module:** `agents/orchestrator.py`  
**Model:** Groq / Llama 3.3 70B

The orchestrator node is the entry point of every run. It:

1. Accepts the trigger phrase from the command field or CLI
2. Applies a keyword-to-category filter to scope which registry items are considered
3. Classifies each scoped item into a quadrant (HU/HI, HU/LI, LU/HI, LU/LI) via a single Groq call
4. Fans out to specialist subagents in parallel

**HU/HI-only mode** — triggers containing keywords like `immediate`, `urgent`, or `critical`
automatically filter to HU/HI items only before classification.

**Category filter keyword map** (examples):

| Trigger keyword | Categories passed to subagents |
|---|---|
| `plumbing` | plumbing |
| `electrical` / `fire` | electrical |
| `hvac` / `seasonal` | hvac |
| `appliance` | appliance |
| `full` / `weekly` / `morning` | all five categories |

---

## Specialist Subagents

**Module:** `agents/subagents.py`  
**Model:** Groq / Llama 3.3 70B  
**Pattern:** Parallel fan-out via LangGraph `Annotated[list, merge_lists]` reducer

Five specialist subagent nodes execute simultaneously, one per category:

- `hvac_agent`
- `plumbing_agent`
- `electrical_agent`
- `appliance_agent`
- `general_agent`

Each agent receives only its category's items, makes a single Groq call, and returns
a recommendation with a confidence score (0.0–1.0) and rationale. Results merge into
shared graph state before the HITL checkpoint.

**Rule-based fallback:** `tools/subagent_tools.py` contains the original deterministic
recommendation logic and is retained as a reference and fallback.

---

## HITL Briefing

**Module:** `agents/orchestrator.py` — `hitl_briefing` node  
**Pattern:** `interrupt_before` — graph pauses before the synthesizer node

The HITL briefing node:

1. Formats classified items into a structured briefing panel (UI) or terminal prompt (CLI)
2. Surfaces HU/HI and LU/HI items requiring human decision
3. Pauses graph execution via LangGraph's `interrupt_before` mechanism

**In the UI**, the HITL checkpoint panel presents each HU/HI and LU/HI item with:

- Approve / Defer radio buttons
- Free-text notes field
- Submit to resume graph execution

**In the CLI**, the terminal presents items interactively and accepts keyboard input.

**State persistence** — `MemorySaver` checkpointer persists the entire graph state
across the interrupt, including the `groq_api_key` carried in state (not from env vars).

---

## Synthesizer

**Module:** `agents/orchestrator.py` — `synthesizer` node  
**Model:** Groq / Llama 3.3 70B

After HITL decisions are submitted, the synthesizer node:

1. Receives the merged subagent recommendations and HITL decisions
2. Makes a single Groq call to generate the final narrative report
3. Appends a structured HITL decision record (approved/deferred/noted)
4. Writes the completed run to `run_history` in SQLite
5. Terminates the graph

The synthesizer narrative renders in the UI with highlighted item IDs and is available
for PDF export via reportlab.
