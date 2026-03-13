# Setup & Installation

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — package and virtual environment manager

---

## Installation

```bash
git clone https://github.com/ianmeinert/homebase.git
cd homebase
uv sync --dev
cp .env.example .env
```

Edit `.env` and add your Groq API key:

```
GROQ_API_KEY=gsk_...
```

Get a key at: [https://console.groq.com](https://console.groq.com)

---

## Google API Key (Gemini Agents)

The Document Intake, Spreadsheet Analytics, and Schema Metric Discovery agents require a Google
API key for Gemini 2.5 Flash-Lite. Enter it in the Streamlit sidebar when prompted (`AIza...`).

Get a key at: [https://aistudio.google.com](https://aistudio.google.com)

The Groq-backed features (orchestrator, RCA, 5 Whys, charts, etc.) work without it.

---

## Database

`data/homebase.db` is created and seeded automatically on first run. No migration step required.

The registry seeds with 30 items across 5 categories (HVAC, plumbing, electrical, appliance,
general). All reads and writes go through SQLite after first run — `data/registry.json` is only
read once during initial seeding.

---

## Demo Data

For meaningful RCA confidence scores and trend analysis, seed synthetic run history:

```bash
uv run python scripts/seed_run_history.py
```

This inserts 100 synthetic run history records spanning 90 days with realistic quadrant
distributions, staleness trends, and HITL decisions.

| Flag | Description |
|---|---|
| `--clear` | Clear existing history before seeding |
| `--count N` | Control number of records inserted (default: 100) |

---

## LangSmith Tracing (Optional)

Add to `.env` to activate distributed tracing:

```
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=homebase
```

Get a key at: [https://smith.langchain.com](https://smith.langchain.com) (free tier available).

Tracing status appears in the sidebar. Each run produces a full trace with node timing, LLM calls
(prompt/response/tokens/latency), HITL state, and searchable tags.

---

## Dependencies

| Package | Purpose |
|---|---|
| `langgraph>=1.0.10` | Agent graph, state management, HITL checkpointing |
| `langchain-core>=0.3.0` | LangChain base primitives |
| `langchain-groq>=0.2.0` | Groq/Llama model integration |
| `google-genai>=1.0.0` | Google Generative AI SDK (Gemini) |
| `openpyxl>=3.1.0` | XLSX read/write support for pandas |
| `odfpy>=1.4.0` | ODS support for pandas |
| `plotly>=5.0.0` | Interactive charts |
| `reportlab>=4.0.0` | PDF report generation |
| `python-dotenv>=1.0.0` | Environment variable loading |
| `streamlit>=1.55.0` | Demo UI |
| `pytest>=9.0.2` | Test runner (dev) |

---

## Notes

- The free Groq tier has per-minute rate limits. If you see a `429` error, wait 60 seconds and
  retry. Check usage at: [https://console.groq.com](https://console.groq.com)
- `days_since_update` is computed on read from the `updated_at` timestamp column. Every registry
  write sets `updated_at = datetime('now')`.
- `tools/subagent_tools.py` contains the original rule-based recommendation logic and is retained
  as a reference/fallback.
