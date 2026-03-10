"""
app.py  -  HOMEBASE Streamlit UI (v1.0  -  LLM-backed)

Run with:
    streamlit run app.py

Requires:
    ANTHROPIC_API_KEY in .env or entered via sidebar

Features:
  - Live agent message streaming
  - Gemini-generated recommendations per subagent (batched)
  - Gemini-generated synthesis narrative
  - Quadrant classification table
  - Subagent recommendation cards
  - HITL checkpoint panel
  - Prompt library sidebar
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import uuid
import os

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# -- Page config --------------------------------------------------------------
st.set_page_config(
    page_title="HOMEBASE",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Styling -------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dark command-center theme */
.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * {
    color: #e6edf3 !important;
}

/* Headers */
h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: -0.5px;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 12px;
}

/* Quadrant badge styling */
.badge-hu-hi { background:#3d0f0f; color:#ff6b6b; border:1px solid #7f1d1d; padding:2px 8px; border-radius:4px; font-family:'IBM Plex Mono',monospace; font-size:12px; font-weight:600; }
.badge-hu-li { background:#3d2e0f; color:#fbbf24; border:1px solid #78350f; padding:2px 8px; border-radius:4px; font-family:'IBM Plex Mono',monospace; font-size:12px; font-weight:600; }
.badge-lu-hi { background:#1f2d1f; color:#6ee7b7; border:1px solid #064e3b; padding:2px 8px; border-radius:4px; font-family:'IBM Plex Mono',monospace; font-size:12px; font-weight:600; }
.badge-lu-li { background:#1e2530; color:#93c5fd; border:1px solid #1e3a5f; padding:2px 8px; border-radius:4px; font-family:'IBM Plex Mono',monospace; font-size:12px; font-weight:600; }
.badge-stale { background:#2a1f00; color:#f59e0b; border:1px solid #92400e; padding:2px 6px; border-radius:4px; font-family:'IBM Plex Mono',monospace; font-size:11px; }

/* Rec card */
.rec-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 3px solid #ff6b6b;
    border-radius: 6px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.rec-card-contingency {
    border-left-color: #6ee7b7;
}
.rec-card-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 2px;
}
.rec-card-value {
    font-size: 13px;
    color: #e6edf3;
}

/* Log line */
.log-line {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #8b949e;
    padding: 2px 0;
    border-bottom: 1px solid #161b22;
}
.log-line.orchestrator { color: #79c0ff; }
.log-line.agent        { color: #56d364; }
.log-line.hitl         { color: #f78166; }
.log-line.synth        { color: #d2a8ff; }

/* HITL panel */
.hitl-panel {
    background: #1a0a0a;
    border: 1px solid #7f1d1d;
    border-radius: 8px;
    padding: 20px;
    margin: 12px 0;
}

/* Prompt pill */
.prompt-pill {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 12px;
    color: #c9d1d9;
    margin: 3px 2px;
    display: inline-block;
    cursor: pointer;
}

/* Divider */
hr { border-color: #21262d !important; }

/* Button overrides */
.stButton > button {
    background: #21262d;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    transition: all 0.15s;
}
.stButton > button:hover {
    background: #30363d;
    border-color: #8b949e;
}

/* Primary button */
div[data-testid="column"]:first-child .stButton > button {
    background: #238636;
    border-color: #2ea043;
    color: white;
}
div[data-testid="column"]:first-child .stButton > button:hover {
    background: #2ea043;
}
</style>
""", unsafe_allow_html=True)


# -- Session state init --------------------------------------------------------
def init_state():
    defaults = {
        "phase": "idle",           # idle | running | hitl_wait | complete
        "messages": [],
        "classified_items": [],
        "subagent_results": [],
        "hu_hi": [],
        "hitl_graph": None,
        "thread_config": None,
        "summary_report": "",
        "trigger": "weekly home review",
        "api_key": os.environ.get("GEMINI_API_KEY", ""),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# -- Helpers -------------------------------------------------------------------

QUADRANT_META = {
    "HU/HI": ("badge-hu-hi", "HU/HI", "Immediate"),
    "HU/LI": ("badge-hu-li", "HU/LI", "Schedule Soon"),
    "LU/HI": ("badge-lu-hi", "LU/HI", "Contingency"),
    "LU/LI": ("badge-lu-li", "LU/LI", "Defer"),
}

def quadrant_badge(q: str) -> str:
    cls, icon, label = QUADRANT_META.get(q, ("badge-lu-li", q, q))
    return f'<span class="{cls}">{q}</span>'

def stale_badge() -> str:
    return '<span class="badge-stale">STALE</span>'

def log_class(msg: str) -> str:
    m = msg.lower()
    if "orchestrator" in m: return "orchestrator"
    if "agent" in m and "hitl" not in m: return "agent"
    if "hitl" in m: return "hitl"
    if "synth" in m: return "synth"
    return ""

def iter_stream_chunks(stream):
    """
    Normalize LangGraph stream output.
    Skips __interrupt__ chunks (HITL pause signal) and
    handles both dict {node_name: output} and tuple formats.
    """
    for chunk in stream:
        if isinstance(chunk, tuple):
            node_name, node_output = chunk
            if node_name == "__interrupt__":
                continue
            if isinstance(node_output, dict):
                yield node_name, node_output
        elif isinstance(chunk, dict):
            for node_name, node_output in chunk.items():
                if node_name == "__interrupt__":
                    continue
                if isinstance(node_output, dict):
                    yield node_name, node_output


def get_initial_state(trigger: str) -> dict:
    return {
        "trigger": trigger,
        "raw_registry": [], "classified_items": [],
        "hu_hi": [], "hu_li": [], "lu_hi": [], "lu_li": [],
        "stale_items": [], "delegated_items": [], "subagent_results": [],
        "hitl_approved": False, "hitl_notes": "", "deferred_items": [],
        "summary_report": "", "messages": [],
    }


# -- Sidebar  -  Prompt Library --------------------------------------------------
with st.sidebar:
    st.markdown("## HOMEBASE")
    st.markdown("<p style='color:#8b949e;font-size:13px;margin-top:-8px;'>Multi-Agent Home Management POC</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("#### PROMPT LIBRARY")
    st.markdown("<p style='color:#8b949e;font-size:12px;'>Click to load a trigger</p>", unsafe_allow_html=True)

    prompts = {
        "What needs immediate attention?": "what needs immediate attention",
        "Weekly home review": "weekly home review",
        "Morning briefing": "morning briefing",
        "Fire & safety check": "fire and safety inspection",
        "Plumbing audit": "plumbing systems audit",
        "Electrical inspection": "electrical systems inspection",
        "HVAC seasonal check": "hvac seasonal maintenance check",
        "Appliance status": "appliance status review",
        "Exterior walkthrough": "exterior and grounds walkthrough",
        "Full home assessment": "full home assessment",
    }

    for label, trigger in prompts.items():
        if st.button(label, key=f"prompt_{trigger}", use_container_width=True):
            st.session_state.trigger = trigger
            st.session_state.phase = "idle"
            st.session_state.messages = []
            st.session_state.classified_items = []
            st.session_state.subagent_results = []
            st.session_state.hu_hi = []
            st.session_state.summary_report = ""
            st.rerun()

    st.divider()

    # API Key
    st.markdown("#### API KEY")
    api_key_input = st.text_input(
        "Gemini API Key",
        value=st.session_state.api_key,
        type="password",
        label_visibility="collapsed",
        placeholder="AIza...",
    )
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input

    key_set = bool(st.session_state.api_key.strip())
    if key_set:
        st.markdown("<p style='font-size:11px;color:#56d364;font-family:IBM Plex Mono,monospace;'>OK API key set</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='font-size:11px;color:#f78166;font-family:IBM Plex Mono,monospace;'>NO API key required</p>", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### SYSTEM")
    st.markdown(
        "<p style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#8b949e;'>"
        "Framework: LangGraph<br>"
        "Model: gemini-2.0-flash<br>"
        "Agents: Orchestrator + 5 Specialists<br>"
        "Checkpoint: MemorySaver<br>"
        "HITL: interrupt_before synthesizer"
        "</p>",
        unsafe_allow_html=True,
    )


# -- Main layout ---------------------------------------------------------------
st.markdown("# HOMEBASE")
st.markdown(f"<p style='color:#8b949e;margin-top:-12px;font-size:14px;'>Home Management Multi-Agent System &nbsp;-&nbsp; <span style='font-family:IBM Plex Mono,monospace;color:#79c0ff;'>{st.session_state.trigger}</span></p>", unsafe_allow_html=True)
st.divider()

# -- Trigger row ---------------------------------------------------------------
col_input, col_run = st.columns([4, 1])
with col_input:
    trigger_input = st.text_input(
        "Trigger",
        value=st.session_state.trigger,
        label_visibility="collapsed",
        placeholder="Enter trigger phrase...",
    )
    st.session_state.trigger = trigger_input

with col_run:
    run_disabled = st.session_state.phase in ("running", "hitl_wait") or not st.session_state.api_key.strip()
    if st.button(">  RUN", disabled=run_disabled, use_container_width=True):
        st.session_state.phase = "running"
        st.session_state.messages = []
        st.session_state.classified_items = []
        st.session_state.subagent_results = []
        st.session_state.hu_hi = []
        st.session_state.summary_report = ""
        st.rerun()

st.divider()

# -- Main content columns ------------------------------------------------------
left_col, right_col = st.columns([3, 2])

# -- LEFT: Agent log + HITL ----------------------------------------------------
with left_col:

    # -- EXECUTE phase --------------------------------------------------------
    if st.session_state.phase == "running":
        from graph.graph import build_interactive_graph

        # Inject API key into environment for LLM calls
        os.environ["GEMINI_API_KEY"] = st.session_state.api_key.strip()

        g = build_interactive_graph()
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        st.session_state.hitl_graph = g
        st.session_state.thread_config = config

        st.markdown("#### AGENT LOG")
        log_placeholder = st.empty()
        log_lines = []

        with st.spinner("Agents running  -  calling Gemini..."):
            for node_name, node_output in iter_stream_chunks(g.stream(get_initial_state(st.session_state.trigger), config=config)):
                    for msg in node_output.get("messages", []):
                        cls = log_class(msg)
                        log_lines.append(f'<div class="log-line {cls}">{msg}</div>')
                        log_placeholder.markdown(
                            f'<div style="background:#0d1117;border:1px solid #21262d;border-radius:6px;padding:12px;max-height:300px;overflow-y:auto;">{"".join(log_lines)}</div>',
                            unsafe_allow_html=True
                        )
                    # Capture classified items from orchestrator
                    if "classified_items" in node_output:
                        st.session_state.classified_items = node_output["classified_items"]
                    if "hu_hi" in node_output:
                        st.session_state.hu_hi = node_output["hu_hi"]
                    # Accumulate subagent results
                    if "subagent_results" in node_output:
                        st.session_state.subagent_results += node_output["subagent_results"]

        st.session_state.messages = log_lines
        st.session_state.phase = "hitl_wait"
        st.rerun()

    # -- HITL WAIT phase -------------------------------------------------------
    elif st.session_state.phase == "hitl_wait":
        st.markdown("#### AGENT LOG")
        if st.session_state.messages:
            st.markdown(
                f'<div style="background:#0d1117;border:1px solid #21262d;border-radius:6px;padding:12px;max-height:300px;overflow-y:auto;">{"".join(st.session_state.messages)}</div>',
                unsafe_allow_html=True
            )

        st.divider()

        # HITL checkpoint panel
        st.markdown("""
<div class="hitl-panel">
<p style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#ff6b6b;font-weight:600;margin:0 0 6px 0;">HUMAN-IN-THE-LOOP CHECKPOINT</p>
<p style="font-size:13px;color:#8b949e;margin:0;">The agent has paused. Review HU/HI items and approve or defer before execution resumes.</p>
</div>
""", unsafe_allow_html=True)

        hu_hi_ids = [item["id"] for item in st.session_state.hu_hi]

        with st.form("hitl_form"):
            approved = st.radio(
                "Approve action plan?",
                ["Yes  -  proceed with all approved items", "No  -  defer everything"],
                index=0,
            )

            defer_ids = st.multiselect(
                "Defer specific items (optional)",
                options=hu_hi_ids,
                help="Selected items will be excluded from the final action plan",
            )

            notes = st.text_area(
                "Notes (optional)",
                placeholder="Add context for the record...",
                height=80,
            )

            submitted = st.form_submit_button("OK  Submit Decision & Resume", use_container_width=True)

        if submitted:
            hitl_approved = "Yes" in approved
            g = st.session_state.hitl_graph
            config = st.session_state.thread_config

            # Ensure API key is set for synthesizer Gemini call
            os.environ["GEMINI_API_KEY"] = st.session_state.api_key.strip()

            g.update_state(config, {
                "hitl_approved": hitl_approved,
                "hitl_notes": notes,
                "deferred_items": defer_ids,
            })

            resume_logs = list(st.session_state.messages)
            log_placeholder = st.empty()

            for node_name, node_output in iter_stream_chunks(g.stream(None, config=config)):
                    for msg in node_output.get("messages", []):
                        cls = log_class(msg)
                        resume_logs.append(f'<div class="log-line {cls}">{msg}</div>')

            final_state = g.get_state(config)
            st.session_state.summary_report = final_state.values.get("summary_report", "")
            st.session_state.messages = resume_logs
            st.session_state.phase = "complete"
            st.rerun()

    # -- COMPLETE phase --------------------------------------------------------
    elif st.session_state.phase == "complete":
        st.markdown("#### AGENT LOG")
        if st.session_state.messages:
            st.markdown(
                f'<div style="background:#0d1117;border:1px solid #21262d;border-radius:6px;padding:12px;max-height:300px;overflow-y:auto;">{"".join(st.session_state.messages)}</div>',
                unsafe_allow_html=True
            )

        st.divider()
        st.markdown("#### FINAL REPORT")
        st.code(st.session_state.summary_report, language=None)

        if st.button("Reset  New Run", use_container_width=True):
            for k in ["phase", "messages", "classified_items", "subagent_results",
                      "hu_hi", "hitl_graph", "thread_config", "summary_report"]:
                st.session_state[k] = [] if isinstance(st.session_state[k], list) else \
                                       None if k in ("hitl_graph", "thread_config") else \
                                       "idle" if k == "phase" else ""
            st.rerun()

    # -- IDLE ------------------------------------------------------------------
    else:
        st.markdown("""
<div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:32px;text-align:center;">
<p style="font-family:'IBM Plex Mono',monospace;font-size:28px;margin:0;color:#30363d;">></p>
<p style="color:#8b949e;font-size:14px;margin:8px 0 0 0;">Select a prompt from the sidebar or enter a trigger above, then press RUN.</p>
</div>
""", unsafe_allow_html=True)


# -- RIGHT: Quadrant table + Rec cards ----------------------------------------
with right_col:

    # -- Quadrant summary metrics ----------------------------------------------
    if st.session_state.classified_items:
        st.markdown("#### QUADRANT SUMMARY")

        from collections import Counter
        q_counts = Counter(i["quadrant"] for i in st.session_state.classified_items)
        stale_count = sum(1 for i in st.session_state.classified_items if i["days_since_update"] >= 14)
        total = len(st.session_state.classified_items)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("HU/HI", q_counts.get("HU/HI", 0))
        m2.metric("HU/LI", q_counts.get("HU/LI", 0))
        m3.metric("LU/HI", q_counts.get("LU/HI", 0))
        m4.metric("LU/LI", q_counts.get("LU/LI", 0))

        st.markdown(f"<p style='font-size:12px;color:#8b949e;margin:4px 0 12px 0;'>{total} total items &nbsp;-&nbsp; {stale_count} stale (14+ days)</p>", unsafe_allow_html=True)

        # Classification table
        st.markdown("#### CLASSIFICATION TABLE")
        rows = ""
        for item in sorted(st.session_state.classified_items,
                           key=lambda x: (["HU/HI","HU/LI","LU/HI","LU/LI"].index(x["quadrant"]), -(x["urgency"]+x["impact"]))):
            stale = stale_badge() if item["days_since_update"] >= 14 else ""
            badge = quadrant_badge(item["quadrant"])
            rows += f"""
<tr style="border-bottom:1px solid #21262d;">
  <td style="padding:6px 8px;font-size:12px;font-family:'IBM Plex Mono',monospace;color:#8b949e;">{item['id']}</td>
  <td style="padding:6px 8px;font-size:12px;color:#e6edf3;">{item['title']} {stale}</td>
  <td style="padding:6px 8px;">{badge}</td>
  <td style="padding:6px 8px;font-size:12px;color:#8b949e;text-align:center;">{item['urgency']:.1f} / {item['impact']:.1f}</td>
</tr>"""

        st.markdown(f"""
<div style="overflow-y:auto;max-height:300px;border:1px solid #21262d;border-radius:6px;">
<table style="width:100%;border-collapse:collapse;">
<thead>
<tr style="background:#161b22;border-bottom:1px solid #30363d;">
  <th style="padding:8px;font-family:'IBM Plex Mono',monospace;font-size:11px;color:#8b949e;text-align:left;">ID</th>
  <th style="padding:8px;font-family:'IBM Plex Mono',monospace;font-size:11px;color:#8b949e;text-align:left;">ITEM</th>
  <th style="padding:8px;font-family:'IBM Plex Mono',monospace;font-size:11px;color:#8b949e;text-align:left;">QUADRANT</th>
  <th style="padding:8px;font-family:'IBM Plex Mono',monospace;font-size:11px;color:#8b949e;text-align:center;">U / I</th>
</tr>
</thead>
<tbody>{rows}</tbody>
</table>
</div>
""", unsafe_allow_html=True)

    # -- Subagent recommendation cards -----------------------------------------
    if st.session_state.subagent_results:
        st.divider()
        st.markdown("#### SUBAGENT RECOMMENDATIONS")

        tabs = st.tabs(["HU/HI", "LU/HI"])

        with tabs[0]:
            hu_hi_recs = [r for r in st.session_state.subagent_results if r["item"]["quadrant"] == "HU/HI"]
            if hu_hi_recs:
                for r in sorted(hu_hi_recs, key=lambda x: -(x["item"]["urgency"] + x["item"]["impact"])):
                    item = r["item"]
                    rec = r["recommendation"]
                    stale = "STALE &nbsp;" if item["days_since_update"] >= 14 else ""
                    st.markdown(f"""
<div class="rec-card">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
    <span style="font-family:'IBM Plex Mono',monospace;font-size:12px;font-weight:600;color:#ff6b6b;">[{item['id']}]</span>
    <span style="font-size:11px;color:#f59e0b;">{stale}</span>
    <span style="font-size:11px;color:#8b949e;">{rec['agent']}</span>
  </div>
  <p style="font-size:13px;font-weight:600;color:#e6edf3;margin:0 0 8px 0;">{item['title']}</p>
  <div class="rec-card-label">ACTION</div>
  <div class="rec-card-value" style="margin-bottom:6px;">{rec['action']}</div>
  <div style="display:flex;gap:16px;margin-top:8px;">
    <div><div class="rec-card-label">EFFORT</div><div class="rec-card-value">{rec['estimated_effort']}</div></div>
    <div><div class="rec-card-label">COST EST.</div><div class="rec-card-value">{rec['estimated_cost']}</div></div>
  </div>
  <div style="margin-top:8px;padding:6px 8px;background:#0d1117;border-radius:4px;font-size:11px;color:#8b949e;border-left:2px solid #ff6b6b;">{rec['priority_note']}</div>
</div>
""", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color:#8b949e;font-size:13px;'>No HU/HI items.</p>", unsafe_allow_html=True)

        with tabs[1]:
            lu_hi_recs = [r for r in st.session_state.subagent_results if r["item"]["quadrant"] == "LU/HI"]
            if lu_hi_recs:
                for r in sorted(lu_hi_recs, key=lambda x: -x["item"]["impact"]):
                    item = r["item"]
                    rec = r["recommendation"]
                    st.markdown(f"""
<div class="rec-card rec-card-contingency">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
    <span style="font-family:'IBM Plex Mono',monospace;font-size:12px;font-weight:600;color:#6ee7b7;">[{item['id']}]</span>
    <span style="font-size:11px;color:#8b949e;">{rec['agent']}</span>
  </div>
  <p style="font-size:13px;font-weight:600;color:#e6edf3;margin:0 0 8px 0;">{item['title']}</p>
  <div class="rec-card-label">ACTION</div>
  <div class="rec-card-value" style="margin-bottom:6px;">{rec['action']}</div>
  <div style="display:flex;gap:16px;margin-top:8px;">
    <div><div class="rec-card-label">EFFORT</div><div class="rec-card-value">{rec['estimated_effort']}</div></div>
    <div><div class="rec-card-label">COST EST.</div><div class="rec-card-value">{rec['estimated_cost']}</div></div>
  </div>
  <div style="margin-top:8px;padding:6px 8px;background:#0d1117;border-radius:4px;font-size:11px;color:#8b949e;border-left:2px solid #6ee7b7;">{rec['priority_note']}</div>
</div>
""", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color:#8b949e;font-size:13px;'>No LU/HI items.</p>", unsafe_allow_html=True)

    # -- Idle placeholder ------------------------------------------------------
    if not st.session_state.classified_items and st.session_state.phase == "idle":
        st.markdown("""
<div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:24px;text-align:center;margin-top:8px;">
<p style="color:#30363d;font-size:24px;margin:0;">*</p>
<p style="color:#8b949e;font-size:13px;margin:8px 0 0 0;">Results will appear here after a run.</p>
</div>
""", unsafe_allow_html=True)