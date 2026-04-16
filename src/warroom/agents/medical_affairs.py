"""Medical Affairs Agent Node.

Responsible for querying the ChromaDB vector store using RAG tools
and synthesizing a report on published literature and FDA labels.
Also post-processes output to validate citations.
"""

from __future__ import annotations

import logging
from pprint import pformat

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from warroom.agents.prompts import MEDICAL_AFFAIRS_PROMPT
from warroom.constants import LLM_MODEL_AGENT
from warroom.graph.state import WarRoomState
from warroom.tools.rag_tools import RAG_TOOLS
from warroom.validators.citation_validator import (
  validate_citations,
)

logger = logging.getLogger(__name__)


def medical_affairs_node(state: WarRoomState, config: RunnableConfig) -> dict:
  """Execute the Medical Affairs Agent.

  On retry, includes feedback about detected temporal leaks
  from the previous attempt so the LLM can correct itself.

  Args:
    state: Current graph state (receives query, clinical_intel,
      and optionally warnings from a previous retry).
    config: Runnable configuration for tracing.

  Returns:
    State delta with updated `literature_intel` and `citations`.
  """
  logger.info("Executing Medical Affairs Agent...")

  try:
    from langchain_google_genai import HarmBlockThreshold, HarmCategory

    llm = ChatGoogleGenerativeAI(
      model=LLM_MODEL_AGENT,
      temperature=0.1,
      safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
      },
    )
    agent = create_react_agent(llm, tools=RAG_TOOLS)

    # Provide the system prompt, user's query, and the previous
    # agent's output as context.
    human_prompt = (
      f"Strategic Query: {state['query']}\n\n"
      f"Context from Clinical Intel Agent:\n"
      f"{state.get('clinical_intel', 'None available')}"
    )

    messages: list = [
      SystemMessage(content=MEDICAL_AFFAIRS_PROMPT),
      HumanMessage(content=human_prompt),
    ]

    # If this is a retry, inject feedback about previous leaks.
    prev_warnings = state.get("warnings", [])
    if prev_warnings:
      retry_feedback = (
        "SYSTEM FEEDBACK: Your previous response was rejected "
        "because it contained temporal leaks — references to "
        "events after January 15, 2021. You MUST NOT mention "
        "any of the following:\n"
        + "\n".join(f"  - {w}" for w in prev_warnings)
        + "\n\nPlease regenerate your report while strictly "
        "adhering to the January 2021 time constraint."
      )
      messages.append(HumanMessage(content=retry_feedback))
      logger.info(
        "Injecting retry feedback with %d warnings.",
        len(prev_warnings),
      )

    result = agent.invoke({"messages": messages}, config)
    final_output = result["messages"][-1].content

    # Flatten list of content blocks if necessary
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

    logger.debug(
      "Medical Affairs output:\n%s",
      pformat(final_output[:200]) + "...",
    )

    citation_res = validate_citations(final_output)

    if citation_res.unverified:
      logger.warning("Stripping hallucinated citations: %s", citation_res.unverified)
      for fake in citation_res.unverified:
        final_output = final_output.replace(fake, f"[REMOVED: INVALID CITATION {fake}]")

    # Filter out citations that are already in the state, to avoid duplicates on retry
    new_verified = [
      c for c in citation_res.verified if c not in state.get("citations", [])
    ]

    return {
      "literature_intel": final_output,
      "citations": new_verified,
    }

  except Exception as e:
    logger.exception("Error in medical_affairs_node")
    return {
      "literature_intel": (f"ERROR in Medical Affairs Agent: {e}"),
      "citations": state.get("citations", []),
    }
