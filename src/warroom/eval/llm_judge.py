"""LLM-as-a-Judge orchestrator for evaluating War-Room outputs."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from warroom.constants import LLM_MODEL_JUDGE
from warroom.eval.rubrics import LLM_JUDGE_HUMAN_PROMPT, LLM_JUDGE_SYSTEM_PROMPT
from warroom.graph.state import WarRoomState

logger = logging.getLogger(__name__)


def parse_json_from_llm(output: Any) -> dict[str, Any]:
  """Extract and parse JSON from the LLM's raw text output."""
  # If output is a structured list (blocks), extract the first block's text
  if isinstance(output, list) and len(output) > 0:
    first_block = output[0]
    if isinstance(first_block, dict) and "text" in first_block:
      output = first_block["text"]

  if not isinstance(output, str):
    output = str(output)
  # Try to find content within ```json ... ``` blocks
  match = re.search(r"```json\s*(.*?)\s*```", output, re.DOTALL)
  if match:
    json_str = match.group(1).strip()
  else:
    # Fallback: Find the first { and last } to extract raw JSON
    start_idx = output.find("{")
    end_idx = output.rfind("}")
    if start_idx != -1 and end_idx != -1:
      json_str = output[start_idx : end_idx + 1].strip()
    else:
      json_str = output.strip()

  try:
    # Aggressively clean the string
    # Remove any thinking blocks or preamble if they didn't use markdown tags
    start_idx = output.find("{")
    end_idx = output.rfind("}")
    if start_idx == -1 or end_idx == -1:
      logger.error("No JSON braces found in LLM output: %s", output)
      return _get_parse_error_scores()

    json_str = output[start_idx : end_idx + 1].strip()

    # Repair common LLM JSON errors
    json_str = json_str.replace("\n", " ").replace("\r", " ")
    # Handle potentially unescaped quotes or trailing commas
    # This is a basic repair, for production consider 'json_repair' lib

    return json.loads(json_str)

  except json.JSONDecodeError as e:
    logger.error("Failed to parse LLM JSON: %s\nStripped content: %s", e, json_str)
    # Write to debug file for analysis
    with open("data/eval/failed_judge_response.txt", "w") as f:
      f.write(f"ERROR: {e}\n\nRAW OUTPUT:\n{output}\n\nSTRIPPED STR:\n{json_str}")

    return _get_parse_error_scores()


def _get_parse_error_scores() -> dict[str, Any]:
  """Return default scores for parse errors."""
  return {
    "temporal_integrity": {"score": 0, "justification": "JSON Parse Error"},
    "prc_compliance": {"score": 1, "justification": "JSON Parse Error"},
    "strategic_utility": {"score": 1, "justification": "JSON Parse Error"},
    "citation_validity": {"score": 0, "justification": "JSON Parse Error"},
  }


def evaluate_test_case(
  test_case: dict[str, Any], final_state: WarRoomState
) -> dict[str, Any]:
  """Evaluate a completed War-Room execution using the LLM Judge.

  Args:
    test_case: Dictionary containing the query, expected focus, and trap booleans.
    final_state: The final state of the LangGraph execution.

  Returns:
    A dictionary containing the scores and justifications.
  """
  try:
    llm = ChatGoogleGenerativeAI(
      model=LLM_MODEL_JUDGE,
      temperature=0.0,  # Deterministic scoring
    )

    # Format the state cleanly to avoid massive token bloat
    state_summary = {
      "clinical_intel": final_state.get("clinical_intel", ""),
      "literature_intel": final_state.get("literature_intel", ""),
      "strategy_brief": final_state.get("strategy_brief", ""),
      "competitor_counter_plan": final_state.get("competitor_counter_plan", ""),
      "compliance_warnings": final_state.get("compliance_warnings", []),
      "citations": final_state.get("citations", []),
      "iteration_count": final_state.get("iteration_count", 0),
      "prc_iteration_count": final_state.get("prc_iteration_count", 0),
    }

    formatted_state = json.dumps(state_summary, indent=2)

    human_prompt = LLM_JUDGE_HUMAN_PROMPT.format(
      query=test_case["query"],
      expected_focus=test_case["expected_focus"],
      temporal_trap=test_case["temporal_trap"],
      prc_trap=test_case["prc_trap"],
      system_state=formatted_state,
    )

    messages = [
      SystemMessage(content=LLM_JUDGE_SYSTEM_PROMPT),
      HumanMessage(content=human_prompt),
    ]

    result = llm.invoke(messages)
    return parse_json_from_llm(result.content)

  except Exception as e:
    logger.exception("Error during LLM evaluation")
    return {
      "error": str(e),
      "temporal_integrity": {"score": 0, "justification": "Eval Exception"},
      "prc_compliance": {"score": 1, "justification": "Eval Exception"},
      "strategic_utility": {"score": 1, "justification": "Eval Exception"},
      "citation_validity": {"score": 0, "justification": "Eval Exception"},
    }
