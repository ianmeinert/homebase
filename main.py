"""
main.py  -  CLI runner for the HOMEBASE multi-agent POC.

Modes:
    python main.py                        # Interactive HITL mode (default)
    python main.py --trigger "briefing"   # Custom trigger
    python main.py --no-hitl              # Non-interactive, skips HITL pause
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import uuid
from graph.graph import graph, build_interactive_graph


DEFAULT_TRIGGER = "weekly home review"


def get_initial_state(trigger: str) -> dict:
    return {
        "trigger": trigger,
        "raw_registry": [],
        "classified_items": [],
        "hu_hi": [], "hu_li": [], "lu_hi": [], "lu_li": [],
        "stale_items": [],
        "delegated_items": [],
        "subagent_results": [],
        "hitl_approved": False,
        "hitl_notes": "",
        "deferred_items": [],
        "summary_report": "",
        "messages": [],
    }


def print_message_log(messages: list[str]):
    print("\nAGENT MESSAGE LOG:")
    print("-" * 60)
    for msg in messages:
        print(f"  {msg}")


def run_noninteractive(trigger: str):
    """Non-interactive run  -  no HITL pause, auto-approves all items."""
    print(f"\nHOMEBASE - Non-interactive mode\n")
    state = get_initial_state(trigger)
    state["hitl_approved"] = True

    result = graph.invoke(state)
    print_message_log(result["messages"])
    print("\n")
    print(result["summary_report"])


def run_interactive(trigger: str):
    """
    Interactive HITL run.
    Graph pauses before synthesizer  -  human reviews HU/HI items,
    approves or defers, then execution resumes.
    """
    print(f"\nHOMEBASE - Interactive HITL mode\n")

    hitl_graph = build_interactive_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    state = get_initial_state(trigger)

    # -----------------------------------------------------------------------
    # Run 1: orchestrator -> subagents -> hitl_briefing -> INTERRUPT
    # -----------------------------------------------------------------------
    print("  Running: orchestrator -> subagents -> hitl briefing...\n")

    for chunk in hitl_graph.stream(state, config=config):
        for node_name, node_output in chunk.items():
            if "messages" in node_output:
                for msg in node_output["messages"]:
                    print(f"  {msg}")

    # Print the HITL briefing (stored in summary_report after briefing node)
    current_state = hitl_graph.get_state(config)
    print(current_state.values.get("summary_report", ""))

    # -----------------------------------------------------------------------
    # HITL interaction  -  collect human input
    # -----------------------------------------------------------------------
    hu_hi_items = current_state.values.get("hu_hi", [])
    hu_hi_ids = [item["id"] for item in hu_hi_items]

    print("\n" + "-" * 60)
    print("  YOUR DECISION")
    print("-" * 60)

    # Approval
    while True:
        approval = input("\n  Approve action plan? [yes/no]: ").strip().lower()
        if approval in ("yes", "y", "no", "n"):
            approved = approval in ("yes", "y")
            break
        print("  Please enter 'yes' or 'no'.")

    # Optional deferrals
    deferred = []
    if approved and hu_hi_ids:
        print(f"\n  HU/HI item IDs: {', '.join(hu_hi_ids)}")
        defer_input = input("  Defer any items? Enter IDs comma-separated, or press Enter to skip: ").strip()
        if defer_input:
            deferred = [d.strip().upper() for d in defer_input.split(",") if d.strip()]
            # Validate
            invalid = [d for d in deferred if d not in hu_hi_ids]
            if invalid:
                print(f"  [STALE]  Unrecognized IDs ignored: {invalid}")
                deferred = [d for d in deferred if d in hu_hi_ids]

    # Optional notes
    notes = input("  Add notes (optional, press Enter to skip): ").strip()

    print("\n  Decision recorded. Resuming execution...\n")
    print("-" * 60)

    # -----------------------------------------------------------------------
    # Run 2: resume -> synthesizer -> END
    # -----------------------------------------------------------------------
    hitl_graph.update_state(
        config,
        {
            "hitl_approved": approved,
            "hitl_notes": notes,
            "deferred_items": deferred,
        }
    )

    for chunk in hitl_graph.stream(None, config=config):
        for node_name, node_output in chunk.items():
            if "messages" in node_output:
                for msg in node_output["messages"]:
                    print(f"  {msg}")

    # Final report
    final_state = hitl_graph.get_state(config)
    print("\n")
    print(final_state.values.get("summary_report", "No report generated."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HOMEBASE Multi-Agent POC")
    parser.add_argument("--trigger", type=str, default=DEFAULT_TRIGGER)
    parser.add_argument("--no-hitl", action="store_true", help="Skip HITL pause")
    args = parser.parse_args()

    if args.no_hitl:
        run_noninteractive(args.trigger)
    else:
        run_interactive(args.trigger)