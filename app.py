"""
app.py  -  HOMEBASE Streamlit UI (v1.0  -  LLM-backed)

Run with:
    streamlit run app.py

Requires:
    ANTHROPIC_API_KEY in .env or entered via sidebar

Features:
  - Live agent message streaming
  - Groq-generated recommendations per subagent (batched)
  - Groq-generated synthesis narrative
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

# Load .env from project root (same directory as app.py)
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(dotenv_path=_env_path)
except ImportError:
    pass  # python-dotenv not installed -- set GROQ_API_KEY in environment or sidebar

# -- LangSmith tracing (activates if LANGCHAIN_API_KEY is set in .env) --------
from tools.tracing import init_tracing, is_tracing_enabled, get_project_name, get_run_metadata
init_tracing()

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

/* Download button */
.stDownloadButton > button {
    background: #161b22 !important;
    color: #79c0ff !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    transition: all 0.15s !important;
}
.stDownloadButton > button:hover {
    background: #1f3a5f !important;
    border-color: #79c0ff !important;
    color: #cae8ff !important;
}

/* HITL form submit button */
.stFormSubmitButton > button {
    background: #1f4a1f !important;
    color: #56d364 !important;
    border: 1px solid #238636 !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    transition: all 0.15s !important;
}
.stFormSubmitButton > button:hover {
    background: #238636 !important;
    border-color: #2ea043 !important;
    color: white !important;
}

/* Expander overrides */
div[data-testid="stExpander"] {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
}
div[data-testid="stExpander"] summary {
    background: #161b22 !important;
    color: #e6edf3 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
}
div[data-testid="stExpander"] summary:hover {
    background: #21262d !important;
}
div[data-testid="stExpander"] summary svg {
    fill: #8b949e !important;
}
div[data-testid="stExpander"] > div[data-testid="stExpanderDetails"] {
    background: #161b22 !important;
    border-top: 1px solid #21262d !important;
    padding: 12px 14px !important;
}
/* File uploader — compact dark styling */
[data-testid="stFileUploader"] {
    background: #161b22 !important;
    border: 1px dashed #30363d !important;
    border-radius: 6px !important;
    padding: 0 !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #8b949e !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    padding: 8px 12px !important;
    min-height: unset !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    display: none !important;
}
[data-testid="stFileUploaderDropzone"] > div {
    gap: 6px !important;
    flex-direction: row !important;
    align-items: center !important;
}
[data-testid="stFileUploaderDropzone"] svg {
    width: 14px !important;
    height: 14px !important;
    color: #8b949e !important;
    flex-shrink: 0 !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: #21262d !important;
    color: #8b949e !important;
    border: 1px solid #30363d !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    padding: 3px 10px !important;
    min-height: unset !important;
    height: 26px !important;
    white-space: nowrap !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: #30363d !important;
    border-color: #8b949e !important;
    color: #e6edf3 !important;
}
[data-testid="stFileUploaderDropzone"] small {
    font-size: 10px !important;
    color: #484f58 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
/* Uploaded file pill */
[data-testid="stFileUploaderFile"] {
    background: #0d1f0d !important;
    border: 1px solid #238636 !important;
    border-radius: 4px !important;
    padding: 4px 10px !important;
    font-size: 11px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    color: #56d364 !important;
}
[data-testid="stFileUploaderFile"] button {
    color: #56d364 !important;
    background: transparent !important;
    border: none !important;
}

</style>
""", unsafe_allow_html=True)


# -- Session state init --------------------------------------------------------
def init_state():
    defaults = {
        "phase": "idle",           # idle | running | hitl_wait | complete
        "active_tab": 0,           # 0=dashboard, 1=registry, 2=history
        "messages": [],
        "classified_items": [],
        "subagent_results": [],
        "hu_hi": [],
        "hitl_graph": None,
        "thread_config": None,
        "summary_report": "",
        "trigger": "",
        "pending_input": "",       # set by prompt library, consumed by command field
        "api_key": os.environ.get("GROQ_API_KEY", ""),
        "chart_result":  None,        # {fig, error, instruction} or None
        "rca_result":    None,        # synthesis RCA result from multiple 5 Whys
        "rca_category":  None,        # unused — kept for backward compat
        "whys_results":  [],          # list of 5 Whys results, one per category run
        "qp_result":     None,        # QuadrantPreview result dict or None
        "qp_input":      "",          # last input that was previewed (dedup)
        "cs_result":     None,        # CompletenessResult dict or None
        "cs_input":      "",          # last input+category that was scored (dedup key)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# Sync GROQ key from session state on every rerun
# (Streamlit reruns are stateless; env vars set in one pass may not persist)
if st.session_state.get("api_key", "").strip():
    os.environ["GROQ_API_KEY"] = st.session_state["api_key"].strip()


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

def confidence_bar(score: float) -> str:
    """Render a confidence score as a small colored progress bar + label."""
    pct = int(score * 100)
    if score >= 0.8:
        color = "#56d364"
        label = "HIGH"
    elif score >= 0.5:
        color = "#fbbf24"
        label = "MED"
    else:
        color = "#f78166"
        label = "LOW"
    return (
        f"<div style='display:flex;align-items:center;gap:8px;margin-top:6px;'>"
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:10px;color:#8b949e;width:32px;'>CONF</div>"
        f"<div style='flex:1;background:#21262d;border-radius:4px;height:6px;'>"
        f"<div style='width:{pct}%;background:{color};border-radius:4px;height:6px;'></div></div>"
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:10px;color:{color};width:36px;'>"
        f"{label} {pct}%</div></div>"
    )


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


def get_initial_state(trigger: str, api_key: str = "") -> dict:
    return {
        "trigger": trigger,
        "groq_api_key": api_key,
        "raw_registry": [], "classified_items": [],
        "hu_hi": [], "hu_li": [], "lu_hi": [], "lu_li": [],
        "stale_items": [], "delegated_items": [], "subagent_results": [],
        "hitl_approved": False, "hitl_notes": "", "deferred_items": [], "active_items": [],
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
        if st.button(label, key=f"prompt_{trigger}", width="stretch"):
            st.session_state.pending_input = trigger
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
        "Groq API Key",
        value=st.session_state.api_key,
        type="password",
        label_visibility="collapsed",
        placeholder="gsk_...",
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
        "Model: llama-3.3-70b-versatile<br>"
        "Agents: Orchestrator + 5 Specialists<br>"
        "Checkpoint: MemorySaver<br>"
        "HITL: interrupt_before synthesizer"
        "</p>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("#### LANGSMITH")
    if is_tracing_enabled():
        project = get_project_name()
        st.markdown(
            f"<p style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#56d364;'>"
            f"TRACING ACTIVE<br>"
            f"<span style='color:#8b949e;'>Project: {project}</span><br>"
            f"<a href='https://smith.langchain.com' target='_blank' "
            f"style='color:#79c0ff;text-decoration:none;'>smith.langchain.com</a>"
            f"</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<p style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#8b949e;'>"
            "TRACING OFF<br>"
            "<span style='font-size:10px;'>Set LANGCHAIN_API_KEY<br>in .env to enable</span>"
            "</p>",
            unsafe_allow_html=True,
        )


# -- Main layout ---------------------------------------------------------------
st.markdown("# HOMEBASE")
st.markdown(f"<p style='color:#8b949e;margin-top:-12px;font-size:14px;'>Home Management Multi-Agent System &nbsp;-&nbsp; <span style='font-family:IBM Plex Mono,monospace;color:#79c0ff;'>{st.session_state.trigger}</span></p>", unsafe_allow_html=True)
st.divider()

main_tabs = st.tabs(["DASHBOARD", "RUN HISTORY"])

with main_tabs[0]:

    # -- Unified command input + file upload ---------------------------------------
    from tools.update_agent import classify_input, execute_command as _execute_command, extract_rca_category
    from tools.registry_tools import get_registry as _get_registry
    from tools.chart_agent import generate_chart as _generate_chart
    from tools.rca_agent import run_rca as _run_rca, run_rca_synthesis as _run_rca_synthesis

    _cmd_disabled = st.session_state.phase in ("running", "hitl_wait")
    _no_key       = not st.session_state.api_key.strip()

    with st.form("unified_cmd_form", clear_on_submit=True):
        _f_col, _b_col = st.columns([5, 1])
        with _f_col:
            unified_input = st.text_input(
                "Command",
                value=st.session_state.get("pending_input", ""),
                label_visibility="collapsed",
                placeholder='Run · Registry · Chart · RCA  —  e.g. "weekly review"  ·  "mark APP-001 in progress"  ·  "chart urgency by category"  ·  "root cause analysis"',
                disabled=_cmd_disabled,
            )
        with _b_col:
            submitted = st.form_submit_button(
                ">  GO",
                disabled=_cmd_disabled or _no_key,
                width="stretch",
            )

    # -- Predictive Quadrant Preview + Completeness Scorer ------------------------
    # Single expander handles both:
    #   1. Quadrant prediction (predict_quadrant) — fires on every input change
    #   2. Completeness scoring (score_completeness) — fires after quadrant resolves,
    #      using the predicted category as the rubric key
    # Both use on_change dedup to avoid redundant API calls.
    from tools.quadrant_preview    import predict_quadrant    as _predict_quadrant
    from tools.completeness_agent  import score_completeness  as _score_completeness

    def _run_preview():
        _desc = st.session_state.get("qp_live_input", "").strip()
        if len(_desc) < 8:
            st.session_state.qp_result = None
            st.session_state.qp_input  = ""
            st.session_state.cs_result = None
            st.session_state.cs_input  = ""
            return
        _api_key_qp = st.session_state.api_key.strip() or None

        # -- Quadrant prediction (deduped) --
        if _desc != st.session_state.get("qp_input", ""):
            st.session_state.qp_result = _predict_quadrant(_desc, api_key=_api_key_qp)
            st.session_state.qp_input  = _desc

        # -- Completeness scoring (deduped on desc + category) --
        _qp_res = st.session_state.qp_result
        if _qp_res and not _qp_res.get("error") and _qp_res.get("quadrant"):
            # Derive category from quadrant: map HU/HI → use the predicted category
            # The quadrant preview doesn't return a category, so we ask the completeness
            # agent to infer it from the description directly (category="general" triggers
            # fuzzy matching inside score_completeness; we pass the full description so
            # the LLM can use category-aware rubric selection via the system prompt).
            # For a richer signal, we pass the rationale as a category hint.
            _cat_hint = _qp_res.get("rationale", "")
            _cs_key   = f"{_desc}|{_qp_res['quadrant']}"
            if _cs_key != st.session_state.get("cs_input", ""):
                # Infer category from description via a lightweight keyword pass,
                # falling back to the completeness agent's own normalization.
                _inferred_cat = _infer_category_from_description(_desc)
                st.session_state.cs_result = _score_completeness(
                    _desc, _inferred_cat, api_key=_api_key_qp
                )
                st.session_state.cs_input  = _cs_key
        else:
            st.session_state.cs_result = None
            st.session_state.cs_input  = ""

    def _infer_category_from_description(desc: str) -> str:
        """Lightweight keyword-based category inference for completeness rubric selection."""
        d = desc.lower()
        # Appliance checked first — "dryer not heating" contains "heating" but is not HVAC
        if any(k in d for k in ("washer", "dryer", "dishwasher", "refrigerator", "fridge", "oven", "stove", "microwave", "appliance")):
            return "appliance"
        if any(k in d for k in ("furnace", "hvac", "ac ", "air condition", "vent", "duct", "filter", "thermostat", "cooling", "heating", "heat pump")):
            return "hvac"
        # "heat" alone checked after appliance to avoid false matches (e.g. "not heating" on dryer)
        if "heat" in d and not any(k in d for k in ("washer", "dryer", "dishwasher", "refrigerator", "fridge", "oven", "stove", "microwave")):
            return "hvac"
        if any(k in d for k in ("pipe", "drain", "leak", "plumb", "toilet", "faucet", "sink", "water heater", "shower", "hose", "clog", "flood")):
            return "plumbing"
        if any(k in d for k in ("outlet", "circuit", "breaker", "electrical", "wiring", "panel", "gfci", "light", "switch", "spark", "power")):
            return "electrical"
        return "general"

    with st.expander("⬡  Predictive Quadrant Preview", expanded=False):
        st.caption("Type an issue description to predict its quadrant and score description completeness.")
        _qp_disabled = _cmd_disabled or _no_key
        st.text_input(
            "Issue description",
            key="qp_live_input",
            placeholder='e.g. "furnace making grinding noise and it\'s January"',
            disabled=_qp_disabled,
            label_visibility="collapsed",
            on_change=_run_preview,
        )

        _qp = st.session_state.get("qp_result")

        # -- Quadrant badge --
        if _qp and not _qp.get("error") and _qp.get("quadrant"):
            _q        = _qp["quadrant"]
            _conf     = _qp["confidence"]
            _rat      = _qp.get("rationale", "")
            _conf_pct = int(_conf * 100)

            _badge_class = {
                "HU/HI": "badge-hu-hi",
                "HU/LI": "badge-hu-li",
                "LU/HI": "badge-lu-hi",
                "LU/LI": "badge-lu-li",
            }.get(_q, "badge-lu-li")

            _bar_color = (
                "#56d364" if _conf >= 0.80 else
                "#fbbf24" if _conf >= 0.60 else
                "#ff6b6b"
            )

            st.markdown(
                f"""
                <div style='display:flex;align-items:center;gap:14px;margin:8px 0 4px 0;'>
                    <span class='{_badge_class}'>{_q}</span>
                    <div style='flex:1;background:#21262d;border-radius:4px;height:6px;overflow:hidden;'>
                        <div style='width:{_conf_pct}%;background:{_bar_color};height:100%;border-radius:4px;
                                    transition:width 0.3s ease;'></div>
                    </div>
                    <span style='font-family:IBM Plex Mono,monospace;font-size:12px;color:{_bar_color};
                                 min-width:36px;text-align:right;'>{_conf_pct}%</span>
                </div>
                <p style='color:#8b949e;font-size:12px;margin:4px 0 8px 0;font-style:italic;'>{_rat}</p>
                """,
                unsafe_allow_html=True,
            )

            # -- Completeness scorer --
            _cs = st.session_state.get("cs_result")
            if _cs and not _cs.get("error"):
                _cs_score     = _cs["score"]
                _cs_pct       = int(_cs_score * 100)
                _cs_questions = _cs.get("questions", [])
                _cs_missing   = _cs.get("missing_fields", [])
                _cs_category  = _cs.get("category", "")

                _cs_bar_color = (
                    "#56d364" if _cs_score >= 0.80 else
                    "#fbbf24" if _cs_score >= 0.50 else
                    "#ff6b6b"
                )

                st.markdown(
                    f"""
                    <div style='border-top:1px solid #21262d;margin:4px 0 8px 0;padding-top:10px;'>
                      <div style='display:flex;align-items:center;gap:10px;margin-bottom:6px;'>
                        <span style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#8b949e;
                                     text-transform:uppercase;letter-spacing:0.5px;'>Completeness</span>
                        <span style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#484f58;'>
                          {_cs_category}</span>
                        <div style='flex:1;background:#21262d;border-radius:4px;height:5px;overflow:hidden;'>
                          <div style='width:{_cs_pct}%;background:{_cs_bar_color};height:100%;border-radius:4px;'></div>
                        </div>
                        <span style='font-family:IBM Plex Mono,monospace;font-size:12px;
                                     color:{_cs_bar_color};min-width:36px;text-align:right;'>{_cs_pct}%</span>
                      </div>
                    """,
                    unsafe_allow_html=True,
                )

                if _cs_questions:
                    st.markdown(
                        "<p style='color:#8b949e;font-size:12px;margin:0 0 6px 0;'>"
                        "To improve this description, consider answering:</p>",
                        unsafe_allow_html=True,
                    )
                    for i, q in enumerate(_cs_questions, 1):
                        st.markdown(
                            f"<p style='color:#c9d1d9;font-size:13px;margin:0 0 4px 0;'>"
                            f"<span style='color:#484f58;font-family:IBM Plex Mono,monospace;'>{i}.</span>"
                            f"&nbsp;{q}</p>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        "<p style='color:#56d364;font-size:12px;margin:0;font-style:italic;'>"
                        "✓ Description looks complete.</p>",
                        unsafe_allow_html=True,
                    )

                st.markdown("</div>", unsafe_allow_html=True)

            elif _cs and _cs.get("error"):
                st.markdown(
                    f"<span style='color:#484f58;font-size:11px;font-family:IBM Plex Mono,monospace;'>"
                    f"Completeness check unavailable.</span>",
                    unsafe_allow_html=True,
                )

        elif _qp and _qp.get("error") and st.session_state.get("qp_input"):
            st.markdown(
                f"<span style='color:#ff6b6b;font-size:12px;font-family:IBM Plex Mono,monospace;'>"
                f"Preview unavailable: {_qp['error']}</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span style='color:#484f58;font-size:12px;font-family:IBM Plex Mono,monospace;'>"
                "Awaiting input…</span>",
                unsafe_allow_html=True,
            )
    # -----------------------------------------------------------------------------

    if submitted and unified_input.strip():
        _input        = unified_input.strip()
        st.session_state.pending_input = ""
        _api_key      = st.session_state.api_key.strip() or None
        _intent_class = classify_input(_input, api_key=_api_key)

        if _intent_class == "run":
            st.session_state.trigger           = _input
            st.session_state.phase             = "running"
            st.session_state.messages          = []
            st.session_state.classified_items  = []
            st.session_state.subagent_results  = []
            st.session_state.hu_hi             = []
            st.session_state.summary_report    = ""
            st.rerun()

        elif _intent_class == "chart":
            # AI chart generation — runs inline, result stored for right column
            with st.spinner("Chart agent building visualization..."):
                _fig, _chart_err = _generate_chart(_input, api_key=_api_key)
            st.session_state.chart_result = {
                "fig":         _fig,
                "error":       _chart_err,
                "instruction": _input,
            }
            st.rerun()

        elif _intent_class == "rca":
            # RCA synthesis — use existing whys_results if available, else run direct RCA
            _valid_whys = [r for r in st.session_state.whys_results if not r.get("error")]
            if _valid_whys:
                with st.spinner(f"RCA synthesizing across {len(_valid_whys)} categories..."):
                    _rca = _run_rca_synthesis(_valid_whys, api_key=_api_key)
            else:
                with st.spinner("RCA agent analyzing patterns..."):
                    _rca = _run_rca(_input, category=None, api_key=_api_key)
            st.session_state.rca_result = _rca
            st.rerun()

        elif _intent_class == "whys":
            from tools.whys_agent import run_whys
            # Extract category from instruction — defaults to highest-severity if not specified
            _whys_cat = extract_rca_category(_input) or None
            _cat_label = f"'{_whys_cat}'" if _whys_cat else "highest-severity category"
            with st.spinner(f"5 Whys agent analyzing {_cat_label}..."):
                _whys = run_whys(_input, category=_whys_cat, api_key=_api_key)
            # Append to whys_results, replacing any prior result for the same category
            _existing = [r for r in st.session_state.whys_results
                         if r.get("category") != _whys.get("category")]
            st.session_state.whys_results = _existing + [_whys]
            # Auto-trigger RCA synthesis when 2+ valid whys results exist
            _valid_whys = [r for r in st.session_state.whys_results if not r.get("error")]
            if len(_valid_whys) >= 2:
                with st.spinner(f"RCA synthesizing across {len(_valid_whys)} categories..."):
                    _rca_synth = _run_rca_synthesis(_valid_whys, api_key=_api_key)
                st.session_state.rca_result = _rca_synth
            st.rerun()

        else:
            # Registry command
            with st.spinner("Agent processing..."):
                _result = _execute_command(_input, api_key=_api_key)
            if _result["error"]:
                st.markdown(
                    f"<div style='background:#1a0a0a;border:1px solid #ff6b6b;border-radius:6px;"
                    f"padding:10px 14px;font-size:12px;color:#ff6b6b;font-family:IBM Plex Mono,monospace;'>"
                    f"{_result['error']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                _intent   = _result["intent"].upper()
                _item_id  = _result["item_id"] or ""
                _changes  = _result["changes"]
                _border   = {"ADD": "#238636", "UPDATE": "#1f6feb", "CLOSE": "#8b949e"}.get(_intent, "#238636")
                _label_c  = {"ADD": "#56d364", "UPDATE": "#79c0ff", "CLOSE": "#8b949e"}.get(_intent, "#56d364")
                _header   = f"{'ADDED' if _intent == 'ADD' else 'CLOSED' if _intent == 'CLOSE' else 'UPDATED'} {_item_id}"
                _detail   = "".join(
                    f"<div style='margin:2px 0;'><span style='color:#8b949e;'>{k}&nbsp;</span>"
                    f"<span style='color:{_label_c};font-weight:600;'>{v}</span></div>"
                    for k, v in _changes.items() if k != "_error"
                ) or "<div style='color:#8b949e;'>No changes recorded.</div>"
                st.markdown(
                    f"<div style='background:#0d1117;border:1px solid {_border};border-radius:6px;"
                    f"padding:12px 16px;font-family:IBM Plex Mono,monospace;font-size:12px;'>"
                    f"<div style='color:{_label_c};margin-bottom:8px;font-weight:600;'>{_header}</div>"
                    f"{_detail}</div>",
                    unsafe_allow_html=True,
                )
                if st.session_state.classified_items:
                    _fresh = {i["id"]: i for i in _get_registry()}
                    st.session_state.classified_items = [
                        _fresh.get(i["id"], i)
                        for i in st.session_state.classified_items
                        if i["id"] in _fresh
                    ]
    elif submitted and _no_key:
        st.warning("Enter your Groq API key in the sidebar first.")

    st.divider()

    # -- RCA panel (full width, rendered when rca_result is set) -------------------
    if st.session_state.rca_result:
        _rca = st.session_state.rca_result
        _rca_cat = st.session_state.get("rca_category")

        # Header row
        _rca_h, _rca_clear = st.columns([5, 1])
        with _rca_h:
            _synth_from = _rca.get("synthesized_from", [])
            if _synth_from:
                _synth_label = "  ·  ".join(c.upper() for c in _synth_from)
                st.markdown(f"### Root Cause Analysis  —  synthesized from {_synth_label}")
            else:
                st.markdown("### Root Cause Analysis")
        with _rca_clear:
            if st.button("✕  Clear", key="clear_rca"):
                st.session_state.rca_result = None
                st.session_state.rca_category = None
                st.rerun()

        if _rca.get("error"):
            st.error(_rca["error"])
        else:
            # Overall confidence badge
            _conf     = _rca.get("confidence", 0.0)
            _conf_pct = int(_conf * 100)
            _conf_color = (
                "#ff6b6b" if _conf < 0.5 else
                "#f5a623" if _conf < 0.7 else
                "#7ecb35" if _conf < 0.9 else
                "#00d4aa"
            )
            _conf_rationale = _rca.get("confidence_rationale", "")
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:6px;'>"
                f"<span style='font-size:12px;color:#888;font-family:IBM Plex Mono,monospace;'>ANALYSIS CONFIDENCE</span>"
                f"<span style='font-size:18px;font-weight:700;color:{_conf_color};'>{_conf_pct}%</span>"
                f"<span style='font-size:12px;color:#aaa;font-style:italic;'>{_conf_rationale}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Meta line
            _synth_cats = _rca.get("synthesized_from", [])
            _meta_source = f"synthesized from {len(_synth_cats)} 5 Whys analyses" if _synth_cats else f"{_rca.get('run_count',0)} historical runs"
            st.markdown(
                f"<div style='font-size:11px;color:#666;font-family:IBM Plex Mono,monospace;"
                f"margin-bottom:16px;'>Analyzed {_rca.get('item_count',0)} items · "
                f"{_meta_source} · "
                f"{len(_rca.get('clusters',[]))} pattern clusters identified</div>",
                unsafe_allow_html=True,
            )

            # Pattern clusters
            _clusters = _rca.get("clusters", [])
            if _clusters:
                st.markdown("**Pattern Clusters**")
                _sev_color = {
                    "critical": "#ff4444", "high": "#ff6b6b",
                    "moderate": "#f5a623", "low": "#7ecb35",
                }
                _cluster_cols = st.columns(min(len(_clusters), 3))
                for _i, _cl in enumerate(_clusters):
                    with _cluster_cols[_i % 3]:
                        _sev    = _cl.get("severity", "moderate")
                        _sc     = _sev_color.get(_sev, "#f5a623")
                        _cl_conf = int(float(_cl.get("confidence", 0.0)) * 100)
                        _ids    = ", ".join(_cl.get("item_ids", []))
                        st.markdown(
                            f"<div style='background:#161b22;border:1px solid {_sc};"
                            f"border-radius:8px;padding:12px 14px;margin-bottom:10px;'>"
                            f"<div style='font-size:13px;font-weight:700;color:{_sc};"
                            f"margin-bottom:4px;'>{_cl.get('label','')}</div>"
                            f"<div style='font-size:11px;color:#aaa;margin-bottom:6px;'>"
                            f"{_cl.get('risk_factor','')}</div>"
                            f"<div style='font-size:11px;color:#666;font-family:IBM Plex Mono,monospace;'>"
                            f"Items: {_ids}</div>"
                            f"<div style='display:flex;justify-content:space-between;"
                            f"align-items:center;margin-top:8px;'>"
                            f"<span style='font-size:10px;color:{_sc};text-transform:uppercase;"
                            f"letter-spacing:1px;'>{_sev}</span>"
                            f"<span style='font-size:11px;color:#888;'>confidence: {_cl_conf}%</span>"
                            f"</div></div>",
                            unsafe_allow_html=True,
                        )

            # Narrative
            _narrative = _rca.get("narrative", "")
            if _narrative:
                st.markdown("**Systemic Root Cause Narrative**")
                st.markdown(
                    f"<div style='background:#0d1117;border-left:3px solid #4a9eff;"
                    f"padding:14px 18px;border-radius:0 6px 6px 0;margin-bottom:16px;"
                    f"font-size:13px;color:#cdd9e5;line-height:1.7;'>{_narrative}</div>",
                    unsafe_allow_html=True,
                )

            # Recommendations
            _recs = _rca.get("recommendations", [])
            if _recs:
                st.markdown("**Corrective Actions**")
                _urg_color = {
                    "immediate": "#ff6b6b", "short_term": "#f5a623", "long_term": "#7ecb35"
                }
                _urg_label = {
                    "immediate": "IMMEDIATE", "short_term": "SHORT-TERM", "long_term": "LONG-TERM"
                }
                for _rec in _recs:
                    _urg   = _rec.get("urgency", "short_term")
                    _uc    = _urg_color.get(_urg, "#f5a623")
                    _ul    = _urg_label.get(_urg, _urg.upper())
                    _addrs = ", ".join(_rec.get("addresses_clusters", []))
                    st.markdown(
                        f"<div style='background:#161b22;border:1px solid #30363d;"
                        f"border-radius:6px;padding:12px 14px;margin-bottom:8px;"
                        f"display:flex;gap:12px;align-items:flex-start;'>"
                        f"<div style='min-width:28px;height:28px;background:{_uc}22;"
                        f"border:1px solid {_uc};border-radius:50%;display:flex;"
                        f"align-items:center;justify-content:center;font-size:12px;"
                        f"font-weight:700;color:{_uc};'>{_rec.get('priority','')}</div>"
                        f"<div style='flex:1;'>"
                        f"<div style='font-size:13px;font-weight:600;color:#e6edf3;"
                        f"margin-bottom:4px;'>{_rec.get('action','')}</div>"
                        f"<div style='font-size:12px;color:#aaa;margin-bottom:6px;'>"
                        f"{_rec.get('rationale','')}</div>"
                        f"<div style='display:flex;gap:10px;'>"
                        f"<span style='font-size:10px;color:{_uc};text-transform:uppercase;"
                        f"letter-spacing:1px;border:1px solid {_uc}33;padding:1px 6px;"
                        f"border-radius:3px;'>{_ul}</span>"
                        f"{'<span style=\"font-size:10px;color:#555;font-family:IBM Plex Mono,monospace;\">Addresses: ' + _addrs + '</span>' if _addrs else ''}"
                        f"</div></div></div>",
                        unsafe_allow_html=True,
                    )

        st.divider()

    # -- 5 Whys panels (one per category run, stacked) ----------------------------
    if st.session_state.whys_results:
        _whys_header, _whys_clear = st.columns([5, 1])
        with _whys_header:
            _cats_done = [r.get("category","").upper() for r in st.session_state.whys_results]
            st.markdown(f"### 5 Whys  —  {'  ·  '.join(_cats_done)}")
        with _whys_clear:
            if st.button("✕  Clear All", key="clear_whys"):
                st.session_state.whys_results = []
                st.session_state.rca_result = None
                st.rerun()

        for _wi, _whys in enumerate(st.session_state.whys_results):
            _cat_title = _whys.get("category", "").upper()
            _item_ct   = _whys.get("item_count", 0)
            st.markdown(
                f"<div style='font-size:13px;font-weight:700;color:#4a9eff;"
                f"font-family:IBM Plex Mono,monospace;margin:14px 0 6px;'>"
                f"{_cat_title}  <span style='color:#555;font-weight:400;font-size:11px;'>"
                f"({_item_ct} items analyzed)</span></div>",
                unsafe_allow_html=True,
            )

            if _whys.get("error"):
                st.error(_whys["error"])
                continue

            # Problem statement
            _prob = _whys.get("problem_statement", "")
            if _prob:
                st.markdown(
                    f"<div style='font-size:12px;color:#aaa;font-style:italic;"
                    f"margin-bottom:10px;'>{_prob}</div>",
                    unsafe_allow_html=True,
                )

            # Confidence badge
            _wconf     = _whys.get("confidence", 0.0)
            _wconf_pct = int(_wconf * 100)
            _wconf_color = (
                "#ff6b6b" if _wconf < 0.5 else
                "#f5a623" if _wconf < 0.7 else
                "#7ecb35" if _wconf < 0.9 else
                "#00d4aa"
            )
            _wconf_rationale = _whys.get('confidence_rationale', '')
            st.markdown(
                f"<div style='margin-bottom:12px;'>"
                f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:3px;'>"
                f"<span style='font-size:11px;color:#555;font-family:IBM Plex Mono,monospace;'>CHAIN CONFIDENCE</span>"
                f"<span style='font-size:16px;font-weight:700;color:{_wconf_color};'>{_wconf_pct}%</span>"
                f"</div>"
                f"<div style='font-size:11px;color:#666;font-style:italic;line-height:1.4;'>{_wconf_rationale}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Causal chain
            _chain = _whys.get("causal_chain", [])
            if _chain:
                for _step in _chain:
                    _lvl      = _step.get("level", 0)
                    _why      = _step.get("why", "")
                    _because  = _step.get("because", "")
                    _is_root  = _lvl == 5
                    _indent   = (_lvl - 1) * 16
                    _bc       = "#ff6b6b" if _is_root else "#30363d"
                    _lc       = "#ff6b6b" if _is_root else "#4a9eff"
                    st.markdown(
                        f"<div style='margin-left:{_indent}px;margin-bottom:5px;'>"
                        f"<div style='background:#161b22;border:1px solid {_bc};border-radius:6px;padding:8px 12px;'>"
                        f"<div style='font-size:9px;color:{_lc};font-family:IBM Plex Mono,monospace;"
                        f"text-transform:uppercase;letter-spacing:1px;margin-bottom:3px;'>"
                        f"{'ROOT CAUSE' if _is_root else f'WHY {_lvl}'}</div>"
                        f"<div style='font-size:11px;color:#666;margin-bottom:4px;font-style:italic;'>{_why}</div>"
                        f"<div style='font-size:12px;color:#e6edf3;'>"
                        f"<span style='color:{_lc};margin-right:5px;'>→</span>{_because}</div>"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )

            # Root cause + corrective action side by side
            _rc_col, _ca_col = st.columns(2)
            with _rc_col:
                _root_cause = _whys.get("root_cause", "")
                if _root_cause:
                    st.markdown(
                        f"<div style='background:#1a0a0a;border:2px solid #ff6b6b;"
                        f"border-radius:8px;padding:12px 16px;margin-top:10px;'>"
                        f"<div style='font-size:10px;color:#ff6b6b;font-family:IBM Plex Mono,monospace;"
                        f"text-transform:uppercase;letter-spacing:1px;margin-bottom:5px;'>Root Cause</div>"
                        f"<div style='font-size:12px;color:#ffa0a0;font-weight:600;line-height:1.5;'>"
                        f"{_root_cause}</div></div>",
                        unsafe_allow_html=True,
                    )
            with _ca_col:
                _corrective = _whys.get("corrective_action", "")
                if _corrective:
                    st.markdown(
                        f"<div style='background:#0a1a0a;border:2px solid #7ecb35;"
                        f"border-radius:8px;padding:12px 16px;margin-top:10px;'>"
                        f"<div style='font-size:10px;color:#7ecb35;font-family:IBM Plex Mono,monospace;"
                        f"text-transform:uppercase;letter-spacing:1px;margin-bottom:5px;'>Corrective Action</div>"
                        f"<div style='font-size:12px;color:#a8e06e;line-height:1.6;'>{_corrective}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            if _wi < len(st.session_state.whys_results) - 1:
                st.markdown("<hr style='border-color:#21262d;margin:16px 0;'>", unsafe_allow_html=True)

        st.divider()

    # -- Main content columns ------------------------------------------------------
    left_col, right_col = st.columns([3, 2])

    # -- LEFT: Agent log + HITL ----------------------------------------------------
    with left_col:

        # -- EXECUTE phase --------------------------------------------------------
        if st.session_state.phase == "running":
            from graph.graph import build_interactive_graph

            # Inject API key into environment for LLM calls
            os.environ["GROQ_API_KEY"] = st.session_state.api_key.strip()

            from agents.orchestrator import _resolve_category_filter
            _category_filter, _ = _resolve_category_filter(st.session_state.trigger)
            st.session_state.last_category_filter = _category_filter

            g = build_interactive_graph()
            thread_id = str(uuid.uuid4())
            config = {
                "configurable": {"thread_id": thread_id},
                **get_run_metadata(st.session_state.trigger, _category_filter),
            }
            st.session_state.hitl_graph = g
            st.session_state.thread_config = config

            st.markdown("#### AGENT LOG")
            log_placeholder = st.empty()
            log_lines = []

            with st.spinner("Agents running  -  calling Groq..."):
                for node_name, node_output in iter_stream_chunks(g.stream(get_initial_state(st.session_state.trigger, api_key=st.session_state.api_key.strip()), config=config)):
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
            lu_hi_ids = [r["item"]["id"] for r in st.session_state.subagent_results
                         if r["item"]["quadrant"] == "LU/HI"]
            all_deferrable = hu_hi_ids + [i for i in lu_hi_ids if i not in hu_hi_ids]

            with st.form("hitl_form"):
                approved = st.radio(
                    "Approve action plan?",
                    ["Yes  -  proceed with all approved items", "No  -  defer everything"],
                    index=0,
                )

                defer_ids = st.multiselect(
                    "Defer specific items (optional)",
                    options=all_deferrable,
                    help="HU/HI and LU/HI items can be deferred. Deferred items are excluded from the final action plan.",
                )

                notes = st.text_area(
                    "Notes (optional)",
                    placeholder="Add context for the record...",
                    height=80,
                )

                submitted = st.form_submit_button("OK  Submit Decision & Resume", width="stretch")

            if submitted:
                hitl_approved = "Yes" in approved
                g = st.session_state.hitl_graph
                config = st.session_state.thread_config

                # Ensure API key is set for synthesizer Groq call
                os.environ["GROQ_API_KEY"] = st.session_state.api_key.strip()

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
                deferred = set(final_state.values.get("deferred_items", []))
                st.session_state.deferred_items = list(deferred)
                st.session_state.active_items = [
                    i for i in st.session_state.classified_items
                    if i["id"] not in deferred
                ]
                st.session_state.messages = resume_logs

                # Persist run to history
                from tools.history_tools import save_run
                save_run(
                    trigger=st.session_state.trigger,
                    classified_items=st.session_state.classified_items,
                    deferred_items=list(deferred),
                    hitl_approved=final_state.values.get("hitl_approved", False),
                    hitl_notes=final_state.values.get("hitl_notes", ""),
                    summary_report=st.session_state.summary_report,
                    category_filter=st.session_state.get("last_category_filter"),
                )

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

            # Format report: styled prose panel with word wrap
            import re as _re
            report_lines = st.session_state.summary_report.splitlines()
            formatted = []
            for line in report_lines:
                stripped = line.strip()
                if stripped.startswith("---"):
                    formatted.append("<hr style='border:none;border-top:1px solid #30363d;margin:12px 0;'>")
                elif stripped.startswith("HITL DECISION SUMMARY"):
                    formatted.append(f"<p style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#8b949e;letter-spacing:1px;margin:8px 0 4px 0;'>{stripped}</p>")
                elif stripped and stripped[0].isdigit() and "." in stripped[:4]:
                    line_html = _re.sub(r"(\[[\w-]+\])", r"<span style='color:#79c0ff;font-family:IBM Plex Mono,monospace;font-size:11px;'>\1</span>", stripped)
                    formatted.append(f"<p style='margin:6px 0;font-size:13px;color:#e6edf3;line-height:1.6;'>{line_html}</p>")
                elif stripped.startswith("Approved") or stripped.startswith("All HU/HI"):
                    formatted.append(f"<p style='margin:4px 0;font-size:13px;color:#56d364;'>{stripped}</p>")
                elif stripped:
                    formatted.append(f"<p style='margin:8px 0;font-size:13px;color:#e6edf3;line-height:1.7;'>{stripped}</p>")

            report_html = "\n".join(formatted)
            st.markdown(f"""
    <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;
                padding:20px 24px;word-wrap:break-word;overflow-wrap:break-word;">
    {report_html}
    </div>
    """, unsafe_allow_html=True)

            # Export buttons
            from datetime import datetime as _dt
            from tools.history_tools import build_report_pdf
            _ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            _trigger_slug = st.session_state.trigger.lower().replace(" ", "_")[:30]
            _filename_base = f"homebase_{_trigger_slug}_{_ts}"

            _pdf_bytes = build_report_pdf(
                st.session_state.trigger,
                st.session_state.summary_report,
            )

            st.download_button(
                label="Export Report  (.pdf)",
                data=_pdf_bytes,
                file_name=f"{_filename_base}.pdf",
                mime="application/pdf",
                width="stretch",
            )

            st.markdown("")
            if st.button("Reset  New Run", width="stretch"):
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

        # -- Stale items alert panel -----------------------------------------------
        if st.session_state.classified_items:
            stale_items = [
                i for i in st.session_state.classified_items
                if i.get("days_since_update", 0) >= 14
            ]
            if stale_items:
                stale_sorted = sorted(stale_items, key=lambda x: -x["days_since_update"])
                Q_STALE_COLORS = {"HU/HI": "#ff6b6b", "HU/LI": "#fbbf24", "LU/HI": "#6ee7b7", "LU/LI": "#93c5fd"}
                rows_html = ""
                for item in stale_sorted:
                    q_color = Q_STALE_COLORS.get(item["quadrant"], "#8b949e")
                    rows_html += (
                        f"<div style='display:flex;justify-content:space-between;align-items:center;"
                        f"padding:5px 0;border-bottom:1px solid #21262d;'>"
                        f"<span style='font-family:IBM Plex Mono,monospace;font-size:11px;color:{q_color};width:76px;flex-shrink:0;'>[{item['id']}]</span>"
                        f"<span style='font-size:11px;color:#e6edf3;flex:1;margin:0 8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{item['title']}</span>"
                        f"<span style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#fbbf24;flex-shrink:0;'>{item['days_since_update']}d</span>"
                        f"</div>"
                    )
                st.markdown(
                    f"""<div style='background:#1a1200;border:1px solid #7d4e00;border-left:3px solid #fbbf24;
                        border-radius:6px;padding:12px 14px;margin-bottom:16px;'>
                      <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'>
                        <span style='font-family:IBM Plex Mono,monospace;font-size:11px;font-weight:600;
                          color:#fbbf24;letter-spacing:1px;'>STALE ITEMS ALERT</span>
                        <span style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#7d4e00;'>
                          {len(stale_items)} item{"s" if len(stale_items) != 1 else ""} &nbsp;·&nbsp; 14+ days without update</span>
                      </div>
                      {rows_html}
                    </div>""",
                    unsafe_allow_html=True,
                )

        # -- AI-generated chart ---------------------------------------------------
        if st.session_state.chart_result:
            _cr = st.session_state.chart_result
            st.markdown("#### AI CHART")
            st.markdown(
                f"<p style='font-family:IBM Plex Mono,monospace;font-size:11px;"
                f"color:#8b949e;margin:-8px 0 8px 0;'>↳ {_cr['instruction']}</p>",
                unsafe_allow_html=True,
            )
            if _cr.get("error"):
                st.error(f"Chart error: {_cr['error']}")
            elif _cr.get("fig"):
                st.plotly_chart(_cr["fig"], width="stretch")
            _cc1, _cc2 = st.columns([3, 1])
            with _cc2:
                if st.button("✕ Clear chart", key="clear_chart"):
                    st.session_state.chart_result = None
                    st.rerun()
            st.divider()

        # -- Quadrant summary metrics ----------------------------------------------
        if st.session_state.classified_items:
            st.markdown("#### QUADRANT SUMMARY")

            from collections import Counter

            # Post-run: use active_items (deferred excluded); pre-HITL: use all classified
            chart_items = (
                st.session_state.active_items
                if st.session_state.phase == "complete" and st.session_state.active_items
                else st.session_state.classified_items
            )
            deferred_set = set(st.session_state.get("deferred_items", []))
            is_post_run = st.session_state.phase == "complete" and bool(deferred_set)

            q_counts = Counter(i["quadrant"] for i in chart_items)
            stale_count = sum(1 for i in chart_items if i["days_since_update"] >= 14)
            total = len(chart_items)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("HU/HI", q_counts.get("HU/HI", 0))
            m2.metric("HU/LI", q_counts.get("HU/LI", 0))
            m3.metric("LU/HI", q_counts.get("LU/HI", 0))
            m4.metric("LU/LI", q_counts.get("LU/LI", 0))

            summary_suffix = f"&nbsp;-&nbsp; {len(deferred_set)} deferred" if is_post_run else ""
            st.markdown(f"<p style='font-size:12px;color:#8b949e;margin:4px 0 12px 0;'>{total} active items &nbsp;-&nbsp; {stale_count} stale (14+ days){summary_suffix}</p>", unsafe_allow_html=True)

            # -- Charts ---------------------------------------------------------------
            import plotly.graph_objects as go
            import plotly.express as px

            chart_tabs = st.tabs(["Scatter", "Categories", "Stale", "Distribution"])

            # Quadrant colors
            Q_COLORS = {
                "HU/HI": "#ff6b6b",
                "HU/LI": "#fbbf24",
                "LU/HI": "#6ee7b7",
                "LU/LI": "#93c5fd",
            }

            items = chart_items

            # Tab 1 -- Urgency vs Impact scatter
            with chart_tabs[0]:
                fig = go.Figure()

                # Quadrant shading
                fig.add_shape(type="rect", x0=0.6, x1=1.05, y0=0.6, y1=1.05,
                    fillcolor="rgba(255,107,107,0.06)", line=dict(width=0))
                fig.add_shape(type="rect", x0=0.6, x1=1.05, y0=-0.05, y1=0.6,
                    fillcolor="rgba(251,191,36,0.06)", line=dict(width=0))
                fig.add_shape(type="rect", x0=-0.05, x1=0.6, y0=0.6, y1=1.05,
                    fillcolor="rgba(110,231,183,0.06)", line=dict(width=0))
                fig.add_shape(type="rect", x0=-0.05, x1=0.6, y0=-0.05, y1=0.6,
                    fillcolor="rgba(147,197,253,0.06)", line=dict(width=0))

                # Threshold lines
                fig.add_shape(type="line", x0=0.6, x1=0.6, y0=0, y1=1,
                    line=dict(color="#30363d", width=1, dash="dash"))
                fig.add_shape(type="line", x0=0, x1=1, y0=0.6, y1=0.6,
                    line=dict(color="#30363d", width=1, dash="dash"))

                # Quadrant labels
                for label, x, y in [("HU/HI", 0.82, 0.95), ("HU/LI", 0.82, 0.3),
                                      ("LU/HI", 0.3, 0.95), ("LU/LI", 0.3, 0.3)]:
                    fig.add_annotation(x=x, y=y, text=label, showarrow=False,
                        font=dict(color=Q_COLORS[label], size=10, family="IBM Plex Mono"),
                        opacity=0.5)

                # Plot items by quadrant
                for q, color in Q_COLORS.items():
                    q_items = [i for i in items if i["quadrant"] == q]
                    if q_items:
                        fig.add_trace(go.Scatter(
                            x=[i["urgency"] for i in q_items],
                            y=[i["impact"] for i in q_items],
                            mode="markers+text",
                            name=q,
                            marker=dict(color=color, size=12, line=dict(width=1.5, color="#0d1117")),
                            text=[i["id"] for i in q_items],
                            textposition="top center",
                            textfont=dict(size=9, color="#8b949e"),
                            hovertemplate="<b>%{customdata}</b><br>Urgency: %{x:.2f}<br>Impact: %{y:.2f}<extra></extra>",
                            customdata=[i["title"] for i in q_items],
                        ))

                fig.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    font=dict(color="#8b949e", family="IBM Plex Sans"),
                    xaxis=dict(title="Urgency", range=[-0.05, 1.05],
                        gridcolor="#21262d", zerolinecolor="#21262d",
                        tickfont=dict(size=10)),
                    yaxis=dict(title="Impact", range=[-0.05, 1.05],
                        gridcolor="#21262d", zerolinecolor="#21262d",
                        tickfont=dict(size=10)),
                    legend=dict(
                        bgcolor="#161b22",
                        bordercolor="#30363d",
                        borderwidth=1,
                        font=dict(size=12, color="#e6edf3", family="IBM Plex Sans"),
                        itemsizing="constant",
                        itemwidth=40,
                    ),
                    margin=dict(l=40, r=120, t=20, b=40),
                    height=300,
                )
                st.plotly_chart(fig, width="stretch")

            # Tab 2 -- Category breakdown bar
            with chart_tabs[1]:
                from collections import Counter
                cat_counts = Counter(i["category"] for i in items)
                cats = sorted(cat_counts.keys())
                cat_colors = {
                    "hvac": "#79c0ff", "plumbing": "#6ee7b7",
                    "electrical": "#fbbf24", "appliance": "#d2a8ff", "general": "#8b949e"
                }
                fig2 = go.Figure(go.Bar(
                    x=cats,
                    y=[cat_counts[c] for c in cats],
                    marker_color=[cat_colors.get(c, "#8b949e") for c in cats],
                    text=[cat_counts[c] for c in cats],
                    textposition="outside",
                    textfont=dict(color="#e6edf3", size=11),
                ))
                fig2.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    font=dict(color="#8b949e", family="IBM Plex Sans"),
                    xaxis=dict(gridcolor="#21262d", tickfont=dict(size=10)),
                    yaxis=dict(gridcolor="#21262d", tickfont=dict(size=10), dtick=1),
                    margin=dict(l=40, r=20, t=20, b=40),
                    height=300,
                    showlegend=False,
                )
                st.plotly_chart(fig2, width="stretch")

            # Tab 3 -- Stale vs current donut
            with chart_tabs[2]:
                current_count = total - stale_count
                fig3 = go.Figure(go.Pie(
                    labels=["Current", "Stale (14+ days)"],
                    values=[current_count, stale_count],
                    hole=0.6,
                    marker=dict(colors=["#56d364", "#f59e0b"],
                        line=dict(color="#0d1117", width=2)),
                    textinfo="label+percent",
                    textfont=dict(size=11, color="#e6edf3"),
                    hovertemplate="%{label}: %{value} items<extra></extra>",
                ))
                fig3.add_annotation(
                    text=f"{stale_count}<br>stale",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=14, color="#f59e0b", family="IBM Plex Mono"),
                )
                fig3.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    font=dict(color="#8b949e", family="IBM Plex Sans"),
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=300,
                    showlegend=True,
                    legend=dict(
                        bgcolor="#161b22",
                        bordercolor="#30363d",
                        borderwidth=1,
                        font=dict(size=12, color="#e6edf3", family="IBM Plex Sans"),
                        itemsizing="constant",
                        itemwidth=40,
                    ),
                )
                st.plotly_chart(fig3, width="stretch")

            # Tab 4 -- Urgency + Impact distribution
            with chart_tabs[3]:
                fig4 = go.Figure()
                fig4.add_trace(go.Histogram(
                    x=[i["urgency"] for i in items],
                    name="Urgency",
                    marker_color="#ff6b6b",
                    opacity=0.75,
                    xbins=dict(start=0, end=1, size=0.1),
                ))
                fig4.add_trace(go.Histogram(
                    x=[i["impact"] for i in items],
                    name="Impact",
                    marker_color="#79c0ff",
                    opacity=0.75,
                    xbins=dict(start=0, end=1, size=0.1),
                ))
                fig4.update_layout(
                    barmode="overlay",
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    font=dict(color="#8b949e", family="IBM Plex Sans"),
                    xaxis=dict(title="Score", gridcolor="#21262d", tickfont=dict(size=10)),
                    yaxis=dict(title="Items", gridcolor="#21262d", tickfont=dict(size=10), dtick=1),
                    legend=dict(
                        bgcolor="#161b22",
                        bordercolor="#30363d",
                        borderwidth=1,
                        font=dict(size=12, color="#e6edf3", family="IBM Plex Sans"),
                        itemsizing="constant",
                        itemwidth=40,
                    ),
                    margin=dict(l=40, r=120, t=20, b=40),
                    height=300,
                )
                st.plotly_chart(fig4, width="stretch")

            st.divider()

            # Classification table with inline detail drawer
            st.markdown("#### CLASSIFICATION TABLE")
            st.markdown(
                "<p style='font-size:11px;color:#8b949e;margin:-8px 0 10px 0;font-family:IBM Plex Mono,monospace;'>"
                "Click any row to expand full details</p>",
                unsafe_allow_html=True,
            )

            # Column header row
            st.markdown("""
    <div style="display:grid;grid-template-columns:90px 1fr 90px 70px;gap:0;
                padding:6px 8px;background:#161b22;border:1px solid #30363d;
                border-radius:6px 6px 0 0;margin-bottom:0;">
      <span style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#8b949e;">ID</span>
      <span style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#8b949e;">ITEM</span>
      <span style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#8b949e;">QUADRANT</span>
      <span style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#8b949e;text-align:center;">U / I</span>
    </div>""", unsafe_allow_html=True)

            sorted_items = sorted(
                st.session_state.classified_items,
                key=lambda x: (["HU/HI","HU/LI","LU/HI","LU/LI"].index(x["quadrant"]), -(x["urgency"]+x["impact"]))
            )

            Q_COLORS_MAP = {"HU/HI": "#ff6b6b", "HU/LI": "#fbbf24", "LU/HI": "#6ee7b7", "LU/LI": "#93c5fd"}
            CAT_ICONS    = {"hvac": "HVAC", "plumbing": "PLM", "electrical": "ELC", "appliance": "APP", "general": "GEN"}

            for item in sorted_items:
                is_deferred = item["id"] in deferred_set
                is_stale    = item["days_since_update"] >= 14
                q_color     = Q_COLORS_MAP.get(item["quadrant"], "#8b949e")
                opacity     = "0.35" if is_deferred else "1"

                # Expander label — plain text only (Streamlit does not render HTML in labels)
                stale_marker = "  [STALE]" if is_stale else ""
                defer_marker = "  [deferred]" if is_deferred else ""
                label = (
                    f"[{item['id']}]  {item['title']}{stale_marker}{defer_marker}"
                    f"  ·  {item['quadrant']}  ·  U:{item['urgency']:.1f} I:{item['impact']:.1f}"
                )

                with st.expander(label, expanded=False):
                    d1, d2, d3 = st.columns([1, 1, 1])

                    with d1:
                        st.markdown(f"""
    <div style="padding:10px;background:#0d1117;border-radius:6px;border:1px solid #21262d;">
      <div style="font-family:IBM Plex Mono,monospace;font-size:10px;color:#8b949e;margin-bottom:4px;">ITEM</div>
      <div style="font-family:IBM Plex Mono,monospace;font-size:13px;font-weight:600;color:{q_color};">{item['id']}</div>
      <div style="font-size:13px;color:#e6edf3;margin:4px 0 8px 0;">{item['title']}</div>
      <div style="font-size:11px;color:#8b949e;line-height:1.5;">{item['description']}</div>
    </div>""", unsafe_allow_html=True)

                    with d2:
                        status_color = "#56d364" if item.get("status") == "open" else "#8b949e"
                        stale_color  = "#fbbf24" if is_stale else "#8b949e"
                        stale_text   = f"STALE ({item['days_since_update']}d)" if is_stale else f"{item['days_since_update']}d ago"
                        st.markdown(f"""
    <div style="padding:10px;background:#0d1117;border-radius:6px;border:1px solid #21262d;">
      <div style="font-family:IBM Plex Mono,monospace;font-size:10px;color:#8b949e;margin-bottom:8px;">SCORES</div>
      <div style="display:flex;flex-direction:column;gap:6px;">
        <div>
          <div style="font-size:10px;color:#8b949e;margin-bottom:2px;">URGENCY</div>
          <div style="background:#21262d;border-radius:4px;height:8px;">
            <div style="width:{int(item['urgency']*100)}%;background:#ff6b6b;border-radius:4px;height:8px;"></div>
          </div>
          <div style="font-size:11px;color:#e6edf3;margin-top:2px;">{item['urgency']:.2f}</div>
        </div>
        <div>
          <div style="font-size:10px;color:#8b949e;margin-bottom:2px;">IMPACT</div>
          <div style="background:#21262d;border-radius:4px;height:8px;">
            <div style="width:{int(item['impact']*100)}%;background:#6ee7b7;border-radius:4px;height:8px;"></div>
          </div>
          <div style="font-size:11px;color:#e6edf3;margin-top:2px;">{item['impact']:.2f}</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

                    with d3:
                        cat_label = item.get("category", "general").upper()
                        st.markdown(f"""
    <div style="padding:10px;background:#0d1117;border-radius:6px;border:1px solid #21262d;">
      <div style="font-family:IBM Plex Mono,monospace;font-size:10px;color:#8b949e;margin-bottom:8px;">STATUS</div>
      <div style="margin-bottom:6px;">
        <span style="font-size:10px;color:#8b949e;">QUADRANT&nbsp;</span>
        <span style="font-family:IBM Plex Mono,monospace;font-size:12px;font-weight:600;color:{q_color};">{item['quadrant']}</span>
      </div>
      <div style="margin-bottom:6px;">
        <span style="font-size:10px;color:#8b949e;">CATEGORY&nbsp;</span>
        <span style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#79c0ff;">{cat_label}</span>
      </div>
      <div style="margin-bottom:6px;">
        <span style="font-size:10px;color:#8b949e;">LAST UPDATE&nbsp;</span>
        <span style="font-family:IBM Plex Mono,monospace;font-size:11px;color:{stale_color};">{stale_text}</span>
      </div>
      <div>
        <span style="font-size:10px;color:#8b949e;">STATUS&nbsp;</span>
        <span style="font-family:IBM Plex Mono,monospace;font-size:11px;color:{status_color};">{item.get('status','open').upper()}</span>
      </div>
    </div>""", unsafe_allow_html=True)

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
      {confidence_bar(rec.get('confidence', 0.7))}
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
      {confidence_bar(rec.get('confidence', 0.7))}
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


# -- RUN HISTORY tab -----------------------------------------------------------
with main_tabs[1]:
    from tools.history_tools import get_history, delete_run, clear_history

    st.markdown("#### RUN HISTORY")
    st.markdown("<p style='color:#8b949e;font-size:13px;margin-top:-8px;'>Audit trail of completed agent runs and HITL decisions.</p>", unsafe_allow_html=True)

    history = get_history()

    if not history:
        st.markdown("""
<div style="background:#161b22;border:1px solid #21262d;border-radius:8px;
            padding:32px;text-align:center;margin-top:8px;">
<p style="color:#30363d;font-size:24px;margin:0;">*</p>
<p style="color:#8b949e;font-size:13px;margin:8px 0 0 0;">No runs recorded yet. Complete a run on the Dashboard tab.</p>
</div>
""", unsafe_allow_html=True)
    else:
        # Summary bar
        h_col1, h_col2, h_col3, h_col4 = st.columns(4)
        h_col1.metric("Total Runs", len(history))
        h_col2.metric("Approved", sum(1 for r in history if r["hitl_approved"]))
        h_col3.metric("With Deferrals", sum(1 for r in history if r["deferred_items"]))
        h_col4.metric("Items Reviewed", sum(r["item_count"] for r in history))

        st.markdown("")

        # Run cards
        for rec in history:
            ts = rec["timestamp"].replace("T", "  ")
            approved_color = "#56d364" if rec["hitl_approved"] else "#f78166"
            approved_label = "APPROVED" if rec["hitl_approved"] else "REJECTED"
            deferred_label = f"{len(rec['deferred_items'])} deferred" if rec["deferred_items"] else "none deferred"
            filter_label   = ", ".join(rec["category_filter"]) if rec["category_filter"] else "full registry"
            q = rec["quadrant_summary"]

            with st.expander(f"{ts}   |   {rec['trigger']}   |   {rec['item_count']} items   |   {approved_label}", expanded=False):
                # Meta row
                meta_col1, meta_col2, meta_col3 = st.columns(3)
                with meta_col1:
                    st.markdown(f"""
<p style="font-size:11px;color:#8b949e;font-family:'IBM Plex Mono',monospace;margin:0;">FILTER</p>
<p style="font-size:13px;color:#e6edf3;margin:0;">{filter_label}</p>
""", unsafe_allow_html=True)
                with meta_col2:
                    st.markdown(f"""
<p style="font-size:11px;color:#8b949e;font-family:'IBM Plex Mono',monospace;margin:0;">HITL DECISION</p>
<p style="font-size:13px;color:{approved_color};margin:0;">{approved_label} &nbsp;—&nbsp; {deferred_label}</p>
""", unsafe_allow_html=True)
                with meta_col3:
                    st.markdown(f"""
<p style="font-size:11px;color:#8b949e;font-family:'IBM Plex Mono',monospace;margin:0;">STALE AT RUN TIME</p>
<p style="font-size:13px;color:#e6edf3;margin:0;">{rec.get('stale_count', 0)} items</p>
""", unsafe_allow_html=True)

                # Quadrant breakdown
                st.markdown("")
                q_cols = st.columns(4)
                q_defs = [("HU/HI","#ff6b6b"), ("HU/LI","#fbbf24"), ("LU/HI","#6ee7b7"), ("LU/LI","#93c5fd")]
                for col, (qname, qcolor) in zip(q_cols, q_defs):
                    col.markdown(f"""
<div style="background:#161b22;border:1px solid #21262d;border-radius:6px;
            padding:8px 12px;text-align:center;">
  <p style="font-size:10px;font-family:'IBM Plex Mono',monospace;color:{qcolor};margin:0;">{qname}</p>
  <p style="font-size:20px;font-weight:600;color:#e6edf3;margin:2px 0 0 0;">{q.get(qname, 0)}</p>
</div>
""", unsafe_allow_html=True)

                # Deferred items list
                if rec["deferred_items"]:
                    st.markdown("")
                    st.markdown(f"<p style='font-size:11px;color:#8b949e;font-family:IBM Plex Mono,monospace;margin:0 0 4px 0;'>DEFERRED ITEMS</p>", unsafe_allow_html=True)
                    deferred_str = "  ".join(
                        f"<span style='background:#21262d;color:#8b949e;padding:2px 7px;border-radius:3px;font-family:IBM Plex Mono,monospace;font-size:11px;'>{d}</span>"
                        for d in rec["deferred_items"]
                    )
                    st.markdown(deferred_str, unsafe_allow_html=True)

                # HITL notes
                if rec.get("hitl_notes"):
                    st.markdown("")
                    st.markdown(f"""
<div style="background:#161b22;border-left:2px solid #30363d;padding:8px 12px;border-radius:0 4px 4px 0;">
<p style="font-size:11px;color:#8b949e;font-family:'IBM Plex Mono',monospace;margin:0 0 2px 0;">NOTES</p>
<p style="font-size:12px;color:#e6edf3;margin:0;">{rec['hitl_notes']}</p>
</div>
""", unsafe_allow_html=True)

                # Report toggle
                st.markdown("")
                if rec.get("summary_report"):
                    with st.expander("View full report", expanded=False):
                        import re as _re2
                        report_lines2 = rec["summary_report"].splitlines()
                        formatted2 = []
                        for line in report_lines2:
                            stripped = line.strip()
                            if stripped.startswith("---"):
                                formatted2.append("<hr style='border:none;border-top:1px solid #30363d;margin:12px 0;'>")
                            elif stripped.startswith("HITL"):
                                formatted2.append(f"<p style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#8b949e;margin:8px 0 4px 0;'>{stripped}</p>")
                            elif stripped and stripped[0].isdigit() and "." in stripped[:4]:
                                line_html = _re2.sub(r"(\[[\w-]+\])", r"<span style='color:#79c0ff;font-family:IBM Plex Mono,monospace;font-size:11px;'>\1</span>", stripped)
                                formatted2.append(f"<p style='margin:6px 0;font-size:13px;color:#e6edf3;line-height:1.6;'>{line_html}</p>")
                            elif stripped.startswith("Approved") or stripped.startswith("All HU/HI"):
                                formatted2.append(f"<p style='margin:4px 0;font-size:13px;color:#56d364;'>{stripped}</p>")
                            elif stripped:
                                formatted2.append(f"<p style='margin:8px 0;font-size:13px;color:#e6edf3;line-height:1.7;'>{stripped}</p>")
                        st.markdown(f"""
<div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;
            padding:16px 20px;word-wrap:break-word;overflow-wrap:break-word;">
{"".join(formatted2)}
</div>
""", unsafe_allow_html=True)

                # Actions row
                st.markdown("")
                if st.button("Delete", key=f"del_{rec['run_id']}"):
                    delete_run(rec["run_id"])
                    st.rerun()

        # Clear all
        st.divider()
        if st.button("Clear All History", width="stretch"):
            count = clear_history()
            st.success(f"Cleared {count} run records.")
            st.rerun()