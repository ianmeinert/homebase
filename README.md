# HOMEBASE

### Multi-Agent Home Management System — LangGraph + Groq

### v1.16.0 | [📖 Documentation](https://ianmeinert.github.io/homebase/)

> **⚠ Proof of Concept — Not Production Ready**
> HOMEBASE is a demonstration system built to illustrate multi-agent agentic AI architecture
> patterns. It has not undergone formal code review, security assessment, penetration testing,
> secrets management audit, or production hardening. It should not be deployed in a production
> environment, used to process real sensitive data, or presented as a production-grade system
> without a full security review, compliance evaluation, and architectural assessment appropriate
> to the target environment and regulatory context.

A multi-agent system built with LangGraph and Groq (Llama 3.3 70B) demonstrating
orchestrator/subagent delegation, parallel agent execution, live LLM reasoning,
human-in-the-loop (HITL) checkpoints, and state persistence. The domain is home
management; the architecture is enterprise-transferable.

---

## Quickstart

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/ianmeinert/homebase.git
cd homebase
uv sync --dev
cp .env.example .env
# Add GROQ_API_KEY to .env — get one at https://console.groq.com
uv run streamlit run app.py
```

For CLI usage, demo data seeding, LangSmith tracing setup, and Google API key configuration
see the [Setup guide](https://ianmeinert.github.io/homebase/setup/).

---

## Architecture

```
orchestrator  (trigger-based category filter + optional HU/HI-only mode)
    +-- hvac_agent        -+
    +-- plumbing_agent     |
    +-- electrical_agent   +- (parallel fan-out, one Groq call per agent)
    +-- appliance_agent    |
    +-- general_agent     -+
            |
      hitl_briefing        <- graph pauses here (interrupt_before synthesizer)
            |
       [human input]       <- approve / defer HU/HI + LU/HI items / add notes
            |
       synthesizer         <- Groq generates narrative, appends HITL decision record
            |
           END
```

Full architecture docs: [Architecture Overview](https://ianmeinert.github.io/homebase/architecture/) | [Multi-Provider Strategy](https://ianmeinert.github.io/homebase/architecture/providers/) | [Data Model](https://ianmeinert.github.io/homebase/architecture/data-model/)

---

## Tests

```bash
uv run pytest
```

No API key required — all LLM calls are mocked. **554 passing tests** across 16 files.

---

## Documentation

| Section | Description |
|---|---|
| [Setup](https://ianmeinert.github.io/homebase/setup/) | Install, env vars, demo data, LangSmith |
| [Running](https://ianmeinert.github.io/homebase/running/) | UI features, CLI, prompt library, test table |
| [Architecture](https://ianmeinert.github.io/homebase/architecture/) | Graph topology, project structure |
| [Agents](https://ianmeinert.github.io/homebase/agents/) | Agent reference, LLM vs rule-based breakdown |
| [Enterprise Bridge](https://ianmeinert.github.io/homebase/reference/enterprise/) | Stakeholder concept map |
| [Changelog](https://ianmeinert.github.io/homebase/changelog/) | Version history |
| [Backlog](https://ianmeinert.github.io/homebase/backlog/) | Feature backlog |
