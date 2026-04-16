"""Clinical Intel Agent Node.

Responsible for querying the SQLite AACT database using SQL tools
and synthesizing a report on clinical trial landscapes.
"""

from __future__ import annotations

import logging
from pprint import pformat

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from warroom.agents.prompts import CLINICAL_INTEL_PROMPT
from warroom.constants import LLM_MODEL_AGENT
from warroom.graph.state import WarRoomState
from warroom.tools.sql_tools import SQL_TOOLS

logger = logging.getLogger(__name__)


def clinical_intel_node(state: WarRoomState, config: RunnableConfig) -> dict:
  """Execute the Clinical Intel Agent.

  Args:
    state: Current graph state.
    config: Runnable configuration for tracing.

  Returns:
    State delta with updated `clinical_intel`.
  """
  logger.info("Executing Clinical Intel Agent...")

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

    agent = create_react_agent(llm, tools=SQL_TOOLS)

    messages = [
      SystemMessage(content=CLINICAL_INTEL_PROMPT),
      HumanMessage(content=f"Strategic Query: {state['query']}"),
    ]

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
      "Clinical Intel output:\n%s",
      pformat(final_output[:200]) + "...",
    )

    return {"clinical_intel": final_output}

  except Exception as e:
    logger.exception("Error in clinical_intel_node")
    return {"clinical_intel": (f"ERROR in Clinical Intel Agent: {e}")}
