"""
tools/history_tools.py  -  Run history persistence.

Each completed run is appended to data/run_history.json.
Records include trigger, classification summary, HITL decisions,
and the full synthesized report for audit trail purposes.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

HISTORY_PATH = Path(__file__).parent.parent / "data" / "run_history.json"


def _load_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    with open(HISTORY_PATH) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_history(records: list[dict]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(records, f, indent=4)


def save_run(
    trigger: str,
    classified_items: list[dict],
    deferred_items: list[str],
    hitl_approved: bool,
    hitl_notes: str,
    summary_report: str,
    category_filter: list[str] | None = None,
) -> str:
    """
    Persist a completed run to history.
    Returns the run_id of the saved record.
    """
    from collections import Counter

    q_counts = Counter(i.get("quadrant", "unknown") for i in classified_items)
    stale_count = sum(1 for i in classified_items if i.get("days_since_update", 0) >= 14)

    record = {
        "run_id":          str(uuid.uuid4()),
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
        "trigger":         trigger,
        "category_filter": category_filter,
        "item_count":      len(classified_items),
        "quadrant_summary": {
            "HU/HI": q_counts.get("HU/HI", 0),
            "HU/LI": q_counts.get("HU/LI", 0),
            "LU/HI": q_counts.get("LU/HI", 0),
            "LU/LI": q_counts.get("LU/LI", 0),
        },
        "stale_count":    stale_count,
        "hitl_approved":  hitl_approved,
        "hitl_notes":     hitl_notes,
        "deferred_items": deferred_items,
        "summary_report": summary_report,
    }

    history = _load_history()
    history.append(record)
    _save_history(history)
    return record["run_id"]


def get_history(limit: int | None = None) -> list[dict]:
    """
    Return run history newest-first.
    Optionally limit to the most recent N records.
    """
    history = _load_history()
    history = sorted(history, key=lambda r: r["timestamp"], reverse=True)
    if limit:
        return history[:limit]
    return history


def delete_run(run_id: str) -> bool:
    """Delete a single run record by run_id. Returns True if found."""
    history = _load_history()
    updated = [r for r in history if r["run_id"] != run_id]
    if len(updated) == len(history):
        return False
    _save_history(updated)
    return True


def clear_history() -> int:
    """Delete all run history. Returns count of deleted records."""
    history = _load_history()
    count = len(history)
    _save_history([])
    return count


# -- PDF export ----------------------------------------------------------------


def build_report_pdf(trigger: str, summary_report: str) -> bytes:
    """
    Render a completed run report as a print-ready PDF (light background).
    Returns raw PDF bytes suitable for st.download_button.
    """
    import io
    import re
    from datetime import datetime
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable

    # Print-safe color palette
    DARK_GREY  = colors.HexColor("#222222")
    MID_GREY   = colors.HexColor("#444444")
    LIGHT_GREY = colors.HexColor("#888888")
    RULE       = colors.HexColor("#cccccc")
    ACCENT     = colors.HexColor("#1a5276")
    GREEN      = colors.HexColor("#1e7e34")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.85 * inch,
    )

    s_title = ParagraphStyle("title",
        fontName="Helvetica-Bold", fontSize=22,
        textColor=ACCENT, spaceAfter=2, leading=26)
    s_subtitle = ParagraphStyle("subtitle",
        fontName="Helvetica", fontSize=11,
        textColor=MID_GREY, spaceAfter=2)
    s_datestamp = ParagraphStyle("datestamp",
        fontName="Helvetica", fontSize=9,
        textColor=LIGHT_GREY, spaceAfter=14)
    s_section = ParagraphStyle("section",
        fontName="Helvetica-Bold", fontSize=9,
        textColor=LIGHT_GREY, spaceBefore=14, spaceAfter=6)
    s_body = ParagraphStyle("body",
        fontName="Helvetica", fontSize=10,
        textColor=DARK_GREY, leading=16, spaceAfter=8)
    s_item = ParagraphStyle("item",
        fontName="Helvetica", fontSize=10,
        textColor=DARK_GREY, leading=16, spaceAfter=2,
        leftIndent=12)
    s_approved = ParagraphStyle("approved",
        fontName="Helvetica-Bold", fontSize=10,
        textColor=GREEN, spaceAfter=3)
    s_footer = ParagraphStyle("footer",
        fontName="Helvetica", fontSize=8,
        textColor=LIGHT_GREY, alignment=1)

    def _clean(text):
        return re.sub(r"[^\x00-\x7F]", "", text)

    story = []

    # Header
    ts = datetime.now().strftime("%B %d, %Y  %H:%M")
    story.append(Paragraph("HOMEBASE", s_title))
    story.append(Paragraph(f"Home Management Report  |  {trigger}", s_subtitle))
    story.append(Paragraph(f"Generated: {ts}", s_datestamp))
    story.append(HRFlowable(width="100%", thickness=1.5, color=ACCENT,
                             spaceAfter=14, spaceBefore=2))

    # Body
    for line in summary_report.splitlines():
        stripped = line.strip()

        if not stripped:
            story.append(Spacer(1, 5))

        elif stripped.startswith("---"):
            story.append(HRFlowable(width="100%", thickness=0.5, color=RULE,
                                     spaceBefore=10, spaceAfter=10))

        elif stripped.startswith("HITL DECISION SUMMARY"):
            story.append(Paragraph("HITL DECISION SUMMARY", s_section))

        elif stripped.startswith("Approved") or stripped.startswith("All HU/HI"):
            story.append(Paragraph(_clean(stripped), s_approved))

        elif stripped and stripped[0].isdigit() and "." in stripped[:4]:
            clean_line = _clean(stripped)
            highlighted = re.sub(
                r"(\[\w+-\d+\])",
                r'<font color="#1a5276"><b>\1</b></font>',
                clean_line,
            )
            story.append(Paragraph(highlighted, s_item))

        else:
            story.append(Paragraph(_clean(stripped), s_body))

    # Footer
    story.append(Spacer(1, 24))
    story.append(HRFlowable(width="100%", thickness=0.5, color=RULE, spaceAfter=6))
    story.append(Paragraph(
        "HOMEBASE  |  Multi-Agent Home Management System  |  LangGraph + Groq",
        s_footer,
    ))

    doc.build(story)
    return buf.getvalue()