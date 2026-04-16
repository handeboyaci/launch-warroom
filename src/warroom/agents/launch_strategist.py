"""Launch Strategist Agent Node.

Responsible for synthesizing the outputs of the Clinical Intel and
Medical Affairs agents into a final, actionable strategic brief.
This agent uses no tools.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI

from warroom.agents.prompts import LAUNCH_STRATEGIST_PROMPT
from warroom.constants import LLM_MODEL_AGENT
from warroom.graph.state import WarRoomState

logger = logging.getLogger(__name__)


def launch_strategist_node(state: WarRoomState, config: RunnableConfig) -> dict:
  """Execute the Launch Strategist Agent.

  Args:
    state: Current graph state (needs query, clinical_intel,
      and literature_intel).
    config: Runnable configuration for tracing.

  Returns:
    State delta with updated `strategy_brief`.
  """
  logger.info("Executing Launch Strategist Agent...")

  try:
    from langchain_google_genai import HarmBlockThreshold, HarmCategory

    llm = ChatGoogleGenerativeAI(
      model=LLM_MODEL_AGENT,
      temperature=0.3,
      safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
      },
    )

    human_prompt = (
      f"Strategic Query: {state['query']}\n\n"
      f"--- Clinical Intelligence Report ---\n"
      f"{state.get('clinical_intel', 'None available')}\n\n"
      f"--- Medical Affairs Report ---\n"
      f"{state.get('literature_intel', 'None available')}"
    )

    messages: list = [
      SystemMessage(content=LAUNCH_STRATEGIST_PROMPT),
      HumanMessage(content=human_prompt),
    ]

    # If this is a PRC retry, inject feedback from the Legal Team.
    prev_warnings = state.get("compliance_warnings", [])
    if prev_warnings:
      retry_feedback = (
        f"URGENT FEEDBACK FROM PROMOTIONAL REVIEW COMMITTEE (LEGAL):\n"
        f"Your previous draft was rejected for off-label promotion "
        f"or non-compliant claims.\n\n"
        f"--- YOUR PRIOR DRAFT (REJECTED) ---\n"
        f"{state.get('strategy_brief')}\n\n"
        f"--- END OF REJECTED DRAFT ---\n\n"
        f"You MUST rewrite the brief and explicitly REMOVE or CORRECT "
        f"the following violations:\n"
        + "\n".join(f"  - {w}" for w in prev_warnings)
        + "\n\nRewrite the Final Strategy Brief now. Ensure the Commercial "
        "action items are strictly constrained to the approved "
        "Medical Affairs facts. Do not promote experimental data."
      )
      messages.append(HumanMessage(content=retry_feedback))
      logger.warning(
        "Injecting PRC Legal feedback with %d violations.",
        len(prev_warnings),
      )

    # If this is a temporal retry, inject feedback about the leaks.
    temporal_warnings = state.get("warnings", [])
    if state.get("ls_iteration_count", 0) > 0 and temporal_warnings:
      retry_feedback = (
        "CRITICAL: Your previous draft was rejected for TEMPORAL LEAKS.\n"
        "You mentioned data or events that occur after January 15, 2021.\n\n"
        "--- VIOLATIONS DETECTED ---\n"
        + "\n".join(f"  - {w}" for w in temporal_warnings)
        + "\n\nYou MUST rewrite the brief using the RELATIVE TEMPORAL PROTOCOL.\n"
        "Do NOT use absolute years like 2022, 2023, etc. Use relative phrasing "
        "like 'within 18 months' or 'following the pivotal readout'."
      )
      messages.append(HumanMessage(content=retry_feedback))
      logger.warning(
        "Injecting Temporal feedback to strategist (attempt %d).",
        state.get("ls_iteration_count"),
      )

    result = llm.invoke(messages, config=config)
    final_output = result.content

    # Flatten list of content blocks if necessary (common with Gemini models)
    if isinstance(final_output, list):
      parts = []
      for item in final_output:
        if isinstance(item, dict) and "text" in item:
          parts.append(str(item["text"]))
        else:
          parts.append(str(item))
      final_output = "\n".join(parts)

    if not isinstance(final_output, str):
      final_output = str(final_output)

    logger.info(
      "Launch Strategist output generated. Length: %d chars", len(final_output)
    )
    if len(final_output) > 0:
      logger.debug("Output preview: %s...", final_output[:100].replace("\n", " "))
    else:
      logger.warning("Launch Strategist returned an EMPTY string.")

    return {"strategy_brief": final_output}

  except Exception as e:
    logger.exception("Error in launch_strategist_node")
    return {"strategy_brief": f"ERROR in Launch Strategist: {e}"}
