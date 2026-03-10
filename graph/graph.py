"""
graph.py  -  LangGraph graph definition for HOMEBASE POC.
Phase 1: Single orchestrator node.
Phase 2: Orchestrator -> parallel subagents -> synthesizer.
Phase 3: Adds HITL briefing node + interrupt_before synthesizer.
         Execution pauses after briefing; resumes after human approval.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from graph.state import HombaseState
from agents.orchestrator import orchestrator_node, hitl_briefing_node, synthesizer_node
from agents.subagents import (
    hvac_agent_node,
    plumbing_agent_node,
    electrical_agent_node,
    appliance_agent_node,
    general_agent_node,
)


def build_graph(checkpointer=None, interrupt=True):
    """
    Build the HOMEBASE graph.

    Args:
        checkpointer: LangGraph checkpointer instance (MemorySaver for HITL).
                      Pass None to build without persistence (used in tests).
        interrupt: If True, adds interrupt_before synthesizer for HITL.
                   Set False for non-interactive / test runs.
    """
    builder = StateGraph(HombaseState)

    # Nodes
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("hvac_agent", hvac_agent_node)
    builder.add_node("plumbing_agent", plumbing_agent_node)
    builder.add_node("electrical_agent", electrical_agent_node)
    builder.add_node("appliance_agent", appliance_agent_node)
    builder.add_node("general_agent", general_agent_node)
    builder.add_node("hitl_briefing", hitl_briefing_node)
    builder.add_node("synthesizer", synthesizer_node)

    # Entry
    builder.set_entry_point("orchestrator")

    # Orchestrator fans out to all subagents in parallel
    for agent in ["hvac_agent", "plumbing_agent", "electrical_agent", "appliance_agent", "general_agent"]:
        builder.add_edge("orchestrator", agent)

    # All subagents converge on HITL briefing
    for agent in ["hvac_agent", "plumbing_agent", "electrical_agent", "appliance_agent", "general_agent"]:
        builder.add_edge(agent, "hitl_briefing")

    # HITL briefing -> synthesizer (interrupt fires here in interactive mode)
    builder.add_edge("hitl_briefing", "synthesizer")

    # Synthesizer -> END
    builder.add_edge("synthesizer", END)

    interrupt_nodes = ["synthesizer"] if interrupt else []
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_nodes,
    )


# Module-level graph  -  non-interactive (for tests and imports)
graph = build_graph(checkpointer=None, interrupt=False)


def build_interactive_graph():
    """Returns a graph with MemorySaver checkpointer and HITL interrupt enabled."""
    return build_graph(checkpointer=MemorySaver(), interrupt=True)