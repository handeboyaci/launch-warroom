import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

from dotenv import load_dotenv

from warroom.constants import DATA_DIR
from warroom.eval.llm_judge import evaluate_test_case
from warroom.graph.state import WarRoomState
from warroom.graph.warroom_graph import build_warroom_graph

load_dotenv()

logging.basicConfig(
  level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

EVAL_DIR = DATA_DIR / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_DIR = EVAL_DIR / "transcripts"
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.json"


def _unwrap_agent_output(output: Any) -> str:
  """Unwrap structured agent output (list/dict/blocks) into a clean string."""
  if not output:
    return ""

  if isinstance(output, list) and len(output) > 0:
    parts = []
    for item in output:
      if isinstance(item, dict) and "text" in item:
        parts.append(str(item["text"]))
      elif isinstance(item, str):
        parts.append(item)
    if parts:
      return "\n".join(parts)

  return str(output)


def save_vlm_transcript(test_case: dict, state: WarRoomState):
  """Save a detailed agent-by-agent transcript of the war-room run."""
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  case_id = test_case["id"]
  file_path = TRANSCRIPT_DIR / f"transcript_{timestamp}_{case_id}.md"

  content = [
    f"# War-Room Transcript: {case_id}",
    f"**Query:** {test_case['query']}",
    f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "\n## 🏛️ Strategic Context",
    "**The Player (We):** Acting as the Discovery & Strategy Lead for the user's organization (Benchmarking the landscape to inform internal discovery).",
    "**The Adversary (Red Team):** Acting as an established Top 10 Pharma rival with a mission to Pick apart our strategy and induce clinical inertia.",
    "\n---\n",
    "## 🏥 Clinical Intel Agent",
    _unwrap_agent_output(
      state.get("clinical_intel", "*No clinical intelligence was generated.*")
    ),
    "\n---\n",
    "## 📚 Medical Affairs Agent (Literature Review)",
    _unwrap_agent_output(
      state.get("literature_intel", "*No medical literature review was generated.*")
    ),
    f"\n**Verified Citations:** {', '.join(state.get('citations', []))}",
    "\n---\n",
    "## 🚀 Launch Strategist (Final Brief)",
    _unwrap_agent_output(
      state.get("strategy_brief", "*No strategy brief was generated.*")
    ),
    "\n---\n",
    "## 👺 Red Team (Adversarial Simulation)",
    _unwrap_agent_output(
      state.get(
        "competitor_counter_plan", "*No competitive counter-strategy was generated.*"
      )
    ),
    "\n---\n",
    "## 🛡️ Defense Strategist (Counter-Rebuttal)",
    _unwrap_agent_output(
      state.get("defense_rebuttal", "*No defense rebuttal was generated.*")
    ),
    "\n---\n",
  ]

  with open(file_path, "w") as f:
    f.write("\n\n".join(content))

  logger.info("Saved transcript to %s", file_path)


def run_single_eval(test_case: dict, graph) -> dict:
  """Run the graph on a single test case and evaluate it."""
  query = test_case["query"]
  logger.info("Evaluating [ID: %s]: %s", test_case["id"], query)

  # Sleep to avoid rate limits
  time.sleep(5)

  initial_state = WarRoomState(
    {
      "query": query,
      "clinical_intel": "",
      "literature_intel": "",
      "strategy_brief": "",
      "competitor_counter_plan": "",
      "defense_rebuttal": "",
      "citations": [],
      "compliance_warnings": [],
      "prc_iteration_count": 0,
    }
  )

  try:
    final_state = graph.invoke(initial_state)
    save_vlm_transcript(test_case, final_state)
  except Exception as e:
    logger.error("Graph execution failed for %s: %s", test_case["id"], e)
    strategy_brief = "ERROR IN GRAPH"
    is_temporal_error = "temporal sandbox" in str(e).lower()
    scores = {
      "temporal_integrity": {
        "score": 0 if is_temporal_error else 1,
        "justification": str(e) if is_temporal_error else "N/A",
      },
      "prc_compliance": {
        "score": 1 if not is_temporal_error else 5,
        "justification": "Deterministic PRC compliance guardrail failed."
        if not is_temporal_error
        else "N/A",
      },
      "strategic_utility": {"score": 1, "justification": "Failed to produce brief"},
      "citation_validity": {"score": 0, "justification": "Failed to produce brief"},
    }
    return {"test_case": test_case, "scores": scores, "state_summary": strategy_brief}

  scores = evaluate_test_case(test_case, final_state)
  return {"test_case": test_case, "scores": scores, "state_summary": "Success"}


def run_full_eval_suite(eval_cases: list, graph) -> None:
  """Run the full evaluation suite and generate a report."""
  results = []
  logger.info("Starting evaluation of %d cases...", len(eval_cases))

  with ThreadPoolExecutor(max_workers=1) as executor:
    future_to_case = {
      executor.submit(run_single_eval, tc, graph): tc for tc in eval_cases
    }

    for future in as_completed(future_to_case):
      tc = future_to_case[future]
      try:
        res = future.result()
        results.append(res)
        logger.info("Completed [ID: %s]", tc["id"])
      except Exception as e:
        logger.error("Test case %s generated an exception: %s", tc["id"], e)

  report_lines = [
    "# Oncology War-Room Evaluation Report",
    f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"**Total Cases:** {len(results)}",
    "",
    "## Aggregate Scores",
  ]

  if results:
    total_temporal = sum(
      1 for r in results if r["scores"]["temporal_integrity"]["score"] == 1
    )
    avg_prc = sum(r["scores"]["prc_compliance"]["score"] for r in results) / len(
      results
    )
    avg_utility = sum(r["scores"]["strategic_utility"]["score"] for r in results) / len(
      results
    )
    total_citation = sum(
      1 for r in results if r["scores"]["citation_validity"]["score"] == 1
    )

    temporal_pct = (total_temporal / len(results)) * 100
    citation_pct = (total_citation / len(results)) * 100

    report_lines.extend(
      [
        f"- **Temporal Integrity Coverage:** {total_temporal}/{len(results)} ({temporal_pct:.1f}%)",
        f"- **Citation Validity:** {total_citation}/{len(results)} ({citation_pct:.1f}%)",
        f"- **Average PRC Compliance:** {avg_prc:.2f}/5.0",
        f"- **Average Strategic Utility:** {avg_utility:.2f}/5.0",
        "",
      ]
    )

  report_lines.append("## Case Details\n")
  for r in results:
    tc = r["test_case"]
    scores = r["scores"]
    report_lines.append(f"### ID: {tc['id']}")
    report_lines.append(f"**Query:** {tc['query']}")
    report_lines.append(
      f"**Trap Type:** Temporal: {tc.get('temporal_trap', False)} | PRC: {tc.get('prc_trap', False)}"
    )
    report_lines.append(f"**State Details:** {r['state_summary']}")
    report_lines.append("**Scores:**")
    report_lines.append(
      f"- Temporal Integrity: {scores['temporal_integrity']['score']} - "
      f"{scores['temporal_integrity']['justification']}"
    )
    report_lines.append(
      f"- PRC Compliance: {scores['prc_compliance']['score']}/5 - "
      f"{scores['prc_compliance']['justification']}"
    )
    report_lines.append(
      f"- Strategic Utility: {scores['strategic_utility']['score']}/5 - "
      f"{scores['strategic_utility']['justification']}"
    )
    report_lines.append(
      f"- Citation Validity: {scores['citation_validity']['score']} - "
      f"{scores['citation_validity']['justification']}"
    )
    report_lines.append("\n---\n")

  report_content = "\n".join(report_lines)
  report_path = EVAL_DIR / f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
  with open(report_path, "w") as f:
    f.write(report_content)
  logger.info("Report generated to %s", report_path)


def main():
  parser = argparse.ArgumentParser(description="Run Oncology War-Room Evaluation.")
  parser.add_argument(
    "--test-file",
    type=str,
    default=str(GROUND_TRUTH_PATH),
    help="Test cases JSON file.",
  )
  parser.add_argument("--limit", type=int, help="Limit number of cases.")
  args = parser.parse_args()

  try:
    with open(args.test_file, "r") as f:
      eval_cases = json.load(f)
  except Exception as e:
    logger.error("Failed to load test cases: %s", e)
    return

  if args.limit:
    eval_cases = eval_cases[: args.limit]

  graph = build_warroom_graph()
  run_full_eval_suite(eval_cases, graph)


if __name__ == "__main__":
  main()
