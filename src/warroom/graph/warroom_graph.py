"""LangGraph assembly for the Oncology War-Room.

Connects the Clinical Intel, Medical Affairs, and Launch Strategist
agents with temporal leak detection guardrails and retry loops.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from warroom.agents.clinical_intel import clinical_intel_node
from warroom.agents.launch_strategist import launch_strategist_node
from warroom.agents.medical_affairs import medical_affairs_node
from warroom.graph.state import WarRoomState
from warroom.validators.temporal_leak import scan_for_temporal_leaks

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


# ── Conditional Edge Functions ───────────────────────────────────────


def leak_check_clinical_intel(state: WarRoomState) -> str:
  """Conditional edge after Clinical Intel.

  Scans the clinical_intel output for temporal leaks.
  Routes to error on leak (CI has no retry — it's first in the
  chain so a leak here indicates a fundamental prompt issue).
  """
  text = state.get("clinical_intel", "")
  if not isinstance(text, str):
    text = str(text)
  result = scan_for_temporal_leaks(text)

  if result.clean:
    return "medical_affairs"

  logger.error("Temporal leak in Clinical Intel output: %s", result.leaks)
  return "error_node"


def leak_check_medical_affairs(state: WarRoomState) -> str:
  """Conditional edge after Medical Affairs.

  Scans the literature_intel for temporal leaks. If leaks are
  found, routes to the retry node which stores leak details in
  state for feedback, unless MAX_RETRIES is reached.
  """
  text = state.get("literature_intel", "")
  if not isinstance(text, str):
    text = str(text)
  result = scan_for_temporal_leaks(text)

  if result.clean:
    return "launch_strategist"

  iteration = state.get("iteration_count", 0)
  if iteration >= MAX_RETRIES:
    logger.error(
      "MAX RETRIES (%d) reached for temporal leaks. Aborting.",
      MAX_RETRIES,
    )
    return "error_node"

  logger.warning(
    "Temporal leak detected in MA output (attempt %d/%d). Retrying with feedback.",
    iteration + 1,
    MAX_RETRIES,
  )
  return "retry_medical_affairs"


def leak_check_strategist(state: WarRoomState) -> str:
  """Conditional edge after Launch Strategist.

  Verification of the strategy_brief. Routes to retry if leaks
  are found, or error if max retries reached.
  """
  text = state.get("strategy_brief", "")
  if not isinstance(text, str):
    text = str(text)
  result = scan_for_temporal_leaks(text)

  if result.clean:
    return "prc_agent"

  iteration = state.get("ls_iteration_count", 0)
  if iteration >= 2:  # Allow 2 retries for strategist
    logger.error("MAX RETRIES reached for Strategist temporal leaks. Aborting.")
    return "error_node"

  logger.warning(
    "Temporal leak in Strategist brief (attempt %d/2). Retrying.",
    iteration + 1,
  )
  return "retry_strategist_temporal"


def retry_strategist_temporal(state: WarRoomState) -> dict:
  """Store LS leak details and increment counter."""
  text = state.get("strategy_brief", "")
  result = scan_for_temporal_leaks(text)
  return {
    "ls_iteration_count": state.get("ls_iteration_count", 0) + 1,
    "warnings": result.leaks,
  }


def check_prc_compliance(state: WarRoomState) -> str:
  """Conditional edge after PRC Agent.

  Checks if the brief passed compliance. If not, retries
  the strategist to rewrite it based on warnings.
  """
  warnings = state.get("compliance_warnings", [])
  if not warnings:
    return END

  iteration = state.get("prc_iteration_count", 0)
  if iteration >= MAX_RETRIES:
    logger.error("MAX RETRIES (%d) reached for PRC compliance. Aborting.", MAX_RETRIES)
    return "prc_error_node"

  logger.warning(
    "PRC violations detected (attempt %d/%d). Retrying strategist.",
    iteration + 1,
    MAX_RETRIES,
  )
  return "retry_prc"


def leak_check_red_team(state: WarRoomState) -> str:
  """Conditional edge after Red Team Agent.

  Assures the adversarial plan did not leak post-cutoff data.
  """
  text = state.get("competitor_counter_plan", "")
  if not isinstance(text, str):
    text = str(text)
  result = scan_for_temporal_leaks(text)

  if result.clean:
    return "defense_strategist"

  logger.error("Temporal leak in Red Team counter plan: %s", result.leaks)
  return "error_node"


def leak_check_defense_strategist(state: WarRoomState) -> str:
  """Conditional edge after Defense Strategist Agent.

  Assures the defense rebuttal did not leak post-cutoff data.
  """
  text = state.get("defense_rebuttal", "")
  if not isinstance(text, str):
    text = str(text)
  result = scan_for_temporal_leaks(text)

  if result.clean:
    return END

  logger.error("Temporal leak in Defense Rebuttal: %s", result.leaks)
  return "error_node"


# ── Helper Nodes ─────────────────────────────────────────────────────


def retry_medical_affairs(state: WarRoomState) -> dict:
  """Store leak details in state and increment counter.

  This node runs between a failed MA leak check and the next
  MA attempt. It stores the leak descriptions in ``warnings``
  so the agent can receive feedback about what went wrong.
  """
  text = state.get("literature_intel", "")
  result = scan_for_temporal_leaks(text)

  return {
    "iteration_count": state.get("iteration_count", 0) + 1,
    "warnings": result.leaks,
  }


def _flatten_content(content) -> str:
  """Robustly flatten structured agent output into a string."""
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    parts = []
    for item in content:
      if isinstance(item, dict) and "text" in item:
        parts.append(str(item["text"]))
      else:
        parts.append(str(item))
    return "\n".join(parts)
  return str(content)


def _sanitize_strategic_text(text: str) -> str:
  """Universal temporal scrubber for strategic fields."""
  if not text:
    return text
  import re

  def pmid_scrubber(match):
    pmid_v = int(match.group(1))
    if pmid_v > 33450000:
      return "[REDACTED: FUTURE CITATION]"
    return match.group(0)

  # Scrape PMIDs > 33450000 (roughly Jan 15, 2021)
  sanitized = re.sub(
    r"\bPMID\s*[:\s]*(\d{8,})\b", pmid_scrubber, text, flags=re.IGNORECASE
  )

  # Scrape Years > 2021
  sanitized = re.sub(
    r"(?<![-/])\b(202[2-9]|20[3-9]\d)\b(?![-/])", "[REDACTED: FUTURE YEAR]", sanitized
  )

  # Scrape Future Phrasing
  future_phrases = [
    (
      r"(?:went on to|would later|subsequently|eventually)\s+"
      r"(?:show|demonstrate|receive|gain|achieve)"
    ),
    r"(?:ultimately|in the end|as we now know)",
    r"(?:has since|have since)\s+(?:been|shown|demonstrated)",
  ]
  for phrase in future_phrases:
    sanitized = re.sub(
      phrase, "[REDACTED: FUTURE KNOWLEDGE]", sanitized, flags=re.IGNORECASE
    )

  return sanitized


def _sanitize_citations(citations: list[str]) -> list[str]:
  """Filter out PMIDs > 33450000 from citation lists."""
  if not citations:
    return []
  import re

  sanitized_list = []
  for cit in citations:
    match = re.search(r"PMID\s*[:\s]*(\d+)", cit, re.IGNORECASE)
    if match:
      if int(match.group(1)) <= 33450000:
        sanitized_list.append(cit)
    else:
      # Keep NCT IDs or other non-PMID citations
      sanitized_list.append(cit)
  return sanitized_list


def _wrap_node_with_sanitation(node_fn, output_key: str):
  """Wrap a node function to automatically sanitize its output before state update."""

  def wrapped_node(state: WarRoomState, config: RunnableConfig) -> dict:
    result = node_fn(state, config)
    if isinstance(result, dict):
      if output_key in result:
        result[output_key] = _sanitize_strategic_text(
          _flatten_content(result[output_key])
        )
      if "citations" in result:
        result["citations"] = _sanitize_citations(result.get("citations", []))
    return result

  return wrapped_node


def error_node(state: WarRoomState) -> dict:
  """Sanitize and recover from unrecoverable temporal leaks.

  Instead of aborting, we strip the flagged patterns from ALL strategic
  fields as a last resort to ensure the user gets safe, readable output.
  If the brief is empty (model refusal), we synthesize a baseline summary.
  """
  logger.warning("Triggering Universal Temporal Sanitizer...")

  raw_brief = _flatten_content(state.get("strategy_brief", ""))

  # Synthesis Fallback: if the brief is empty, baseline it from the IA/MA intel
  synthesized = False
  if not raw_brief or not raw_brief.strip():
    logger.warning("Strategist brief is empty. Synthesizing 'Safe Harbour' baseline.")
    synthesized = True
    raw_brief = (
      "# Baseline Strategic Summary (Safe Harbour)\n\n"
      "> [!CAUTION]\n"
      "> **AUTO-SYNTHESIS NOTE:** This brief was synthesized automatically "
      "from Clinical & Medical intelligence because the primary "
      "strategist draft was empty, non-compliant, or contained persistent "
      "temporal leaks.\n\n"
      "## Clinical Foundation\n"
      f"{state.get('clinical_intel', 'Data unavailable.')}\n\n"
      "## Medical Context\n"
      f"{state.get('literature_intel', 'Data unavailable.')}\n\n"
      "## Strategic Mandate\n"
      "Focus on Medical Affairs-led Disease State Education (DSE) "
      "regarding the therapeutic landscape, prioritizing safety management "
      "over promotional claims."
    )

  brief = _sanitize_strategic_text(raw_brief)
  red_team = _sanitize_strategic_text(
    _flatten_content(state.get("competitor_counter_plan", ""))
  )
  defense = _sanitize_strategic_text(
    _flatten_content(state.get("defense_rebuttal", ""))
  )

  # Avoid double cautions if already synthesized
  if not synthesized:
    brief += (
      "\n\n> [!CAUTION]\n"
      "> This brief has been automatically sanitized to remove potential "
      "temporal leaks (post-2021 data) that persisted after multiple retries."
    )

  update = {"strategy_brief": brief}

  if red_team:
    update["competitor_counter_plan"] = red_team
  if defense:
    update["defense_rebuttal"] = defense

  return update


def retry_prc(state: WarRoomState) -> dict:
  """Store iteration for PRC retry."""
  return {"prc_iteration_count": state.get("prc_iteration_count", 0) + 1}


def prc_error_node(state: WarRoomState) -> dict:
  """Sanitize and recover from unrecoverable PRC compliance failures.

  Presents the strategist's last attempt, sanitized for temporal leaks,
  but with a critical warning about non-compliance.
  """
  logger.warning("Triggering Universal PRC Sanitizer...")

  brief = _sanitize_strategic_text(_flatten_content(state.get("strategy_brief", "")))
  red_team = _sanitize_strategic_text(
    _flatten_content(state.get("competitor_counter_plan", ""))
  )
  defense = _sanitize_strategic_text(
    _flatten_content(state.get("defense_rebuttal", ""))
  )

  warnings = "\n".join([f"- {w}" for w in state.get("compliance_warnings", [])])

  update = {
    "strategy_brief": (
      brief + "\n\n> [!CAUTION]\n"
      "> **PRC COMPLIANCE WARNING:** This brief could not be fully "
      "reconciled with compliance guidelines. It may contain unverified "
      "promotional claims. Use with extreme caution.\n"
      "> \n"
      "> **Violations Found:**\n" + warnings
    )
  }

  if red_team:
    update["competitor_counter_plan"] = red_team
  if defense:
    update["defense_rebuttal"] = defense

  return update


def route_after_sanitization(state: WarRoomState) -> str:
  """Intelligently route to the next phase after sanitization."""
  if not state.get("competitor_counter_plan"):
    return "red_team"
  if not state.get("defense_rebuttal"):
    return "defense_strategist"
  return END


# ── Graph Builder ────────────────────────────────────────────────────


def build_warroom_graph() -> Any:
  """Build and compile the multi-agent state machine.

  Flow:
    START
      → clinical_intel
      → [leak check CI] → error_node | medical_affairs
      → [leak check MA] → error_node | retry | launch_strategist
      → [leak check LS] → error_node | retry | prc_agent
      → [check PRC] → prc_error_node | retry_prc | red_team
      → [leak check RT] → error_node | defense_strategist
      → [leak check DS] → error_node | END

    Sanitizer Successor:
      error_node → route_after_sanitization → red_team | defense_strategist | END
  """
  from warroom.agents.defense_strategist import defense_strategist_node
  from warroom.agents.prc_agent import prc_agent_node
  from warroom.agents.red_team import red_team_node

  builder = StateGraph(WarRoomState)  # type: ignore[arg-type]

  # Add nodes with universal sanitization wrappers.
  builder.add_node(
    "clinical_intel", _wrap_node_with_sanitation(clinical_intel_node, "clinical_intel")
  )
  builder.add_node(
    "medical_affairs",
    _wrap_node_with_sanitation(medical_affairs_node, "literature_intel"),
  )
  builder.add_node(
    "launch_strategist",
    _wrap_node_with_sanitation(launch_strategist_node, "strategy_brief"),
  )
  builder.add_node("retry_medical_affairs", retry_medical_affairs)
  builder.add_node("error_node", error_node)
  builder.add_node("prc_agent", prc_agent_node)
  builder.add_node("retry_prc", retry_prc)
  builder.add_node("prc_error_node", prc_error_node)
  builder.add_node(
    "red_team", _wrap_node_with_sanitation(red_team_node, "competitor_counter_plan")
  )
  builder.add_node(
    "defense_strategist",
    _wrap_node_with_sanitation(defense_strategist_node, "defense_rebuttal"),
  )
  builder.add_node("retry_strategist_temporal", retry_strategist_temporal)

  # Entry point.
  builder.set_entry_point("clinical_intel")

  # CI → leak check → MA or error.
  builder.add_conditional_edges(
    "clinical_intel",
    leak_check_clinical_intel,
    {
      "medical_affairs": "medical_affairs",
      "error_node": "error_node",
    },
  )

  # MA → leak check → LS, retry, or error.
  builder.add_conditional_edges(
    "medical_affairs",
    leak_check_medical_affairs,
    {
      "launch_strategist": "launch_strategist",
      "retry_medical_affairs": "retry_medical_affairs",
      "error_node": "error_node",
    },
  )

  # Retry node → back to MA.
  builder.add_edge("retry_medical_affairs", "medical_affairs")

  # LS → final leak check → prc_agent or error.
  builder.add_conditional_edges(
    "launch_strategist",
    leak_check_strategist,
    {
      "prc_agent": "prc_agent",
      "retry_strategist_temporal": "retry_strategist_temporal",
      "error_node": "error_node",
    },
  )

  # prc_agent → compliance check → red_team, retry_prc, or prc_error_node
  builder.add_conditional_edges(
    "prc_agent",
    check_prc_compliance,
    {
      END: "red_team",
      "retry_prc": "retry_prc",
      "prc_error_node": "prc_error_node",
    },
  )

  # red_team → leak_check_red_team → defense_strategist or error_node
  builder.add_conditional_edges(
    "red_team",
    leak_check_red_team,
    {
      "defense_strategist": "defense_strategist",
      "error_node": "error_node",
    },
  )

  # defense_strategist → leak_check_defense_strategist → END or error_node
  builder.add_conditional_edges(
    "defense_strategist",
    leak_check_defense_strategist,
    {
      END: END,
      "error_node": "error_node",
    },
  )

  # Retry nodes → back to start of their respective loops
  builder.add_edge("retry_prc", "launch_strategist")
  builder.add_edge("retry_strategist_temporal", "launch_strategist")

  # Fail-Forward Successor Routing
  builder.add_conditional_edges(
    "error_node",
    route_after_sanitization,
    {
      "red_team": "red_team",
      "defense_strategist": "defense_strategist",
      END: END,
    },
  )
  builder.add_conditional_edges(
    "prc_error_node",
    route_after_sanitization,
    {
      "red_team": "red_team",
      "defense_strategist": "defense_strategist",
      END: END,
    },
  )

  return builder.compile()
