"""
state.py  -  Shared state schema for the HOMEBASE multi-agent POC.
All agents read from and write to this state object as it flows through the graph.
"""

from typing import TypedDict, Annotated
import operator


def merge_lists(a: list, b: list) -> list:
    """Reducer: merge two lists  -  used for parallel subagent writes."""
    return a + b


class TaskItem(TypedDict):
    id: str
    category: str
    title: str
    description: str
    urgency: float
    impact: float
    days_since_update: int
    status: str
    quadrant: str          # Set by classifier agent


class HombaseState(TypedDict):
    # Input
    trigger: str                                                    # What initiated this run

    # Registry data
    raw_registry: list[dict]                                        # Raw items from registry tool

    # Classified items
    classified_items: list[TaskItem]                                # Items after quadrant classification

    # Quadrant buckets (populated after classification)
    hu_hi: list[TaskItem]                                           # High Urgency / High Impact
    hu_li: list[TaskItem]                                           # High Urgency / Low Impact
    lu_hi: list[TaskItem]                                           # Low Urgency / High Impact
    lu_li: list[TaskItem]                                           # Low Urgency / Low Impact
    stale_items: list[TaskItem]                                     # 14+ days without update

    # Phase 2  -  Delegation
    delegated_items: list[dict]                                     # Items routed to subagents
    subagent_results: Annotated[list[dict], merge_lists]            # Parallel subagent writes

    # Phase 3  -  HITL
    hitl_approved: bool                                             # Human approval decision
    hitl_notes: str                                                 # Optional human notes
    deferred_items: list[str]                                       # Item IDs human chose to defer

    # Output
    summary_report: str                                             # Final human-readable report
    messages: Annotated[list[str], operator.add]                    # Agent message log (append-only)