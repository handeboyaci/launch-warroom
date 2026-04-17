"""CLI Entry point for the Oncology War-Room."""

import argparse
import asyncio
import logging

from dotenv import load_dotenv

# Load environment variables
from warroom.graph.state import new_state
from warroom.graph.warroom_graph import build_warroom_graph

load_dotenv()

# Configure logging
logging.basicConfig(
  level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
  parser = argparse.ArgumentParser(
    description="Run the Oncology Launch War-Room simulation."
  )
  parser.add_argument(
    "--query", type=str, required=True, help="The strategic question to answer."
  )
  parser.add_argument(
    "--stream", action="store_true", help="Whether to stream the graph execution."
  )
  args = parser.parse_args()

  # Build the graph
  app = build_warroom_graph()

  # Initialize state
  state = new_state(args.query)

  logger.info("Starting War-Room for query: %s", args.query)

  try:
    final_state = await app.ainvoke(state, {"recursion_limit": 25})

    print("\n" + "=" * 80)
    print("FINAL STRATEGY BRIEF")
    print("=" * 80)
    print(final_state.get("strategy_brief", "No brief generated."))

    print("\n" + "=" * 80)
    print("RED TEAM COUNTER-PLAN")
    print("=" * 80)
    print(final_state.get("competitor_counter_plan", "No counter-plan generated."))

    print("\n" + "=" * 80)
    print("CITATIONS")
    print("=" * 80)
    for cit in final_state.get("citations", []):
      print(f"- {cit}")

  except Exception as e:
    logger.exception("Graph execution failed")
    print(f"\nERROR: {e}")


if __name__ == "__main__":
  asyncio.run(main())
