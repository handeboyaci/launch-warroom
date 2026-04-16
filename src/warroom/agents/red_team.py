"""Red Team War-Gaming Agent Node.

Simulates the Commercial Strategy VP of a rival pharmaceutical
company. Analyzes the finalized launch brief to identify clinical,
commercial, and safety weaknesses, outputting an adversarial
counter-messaging plan.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI

from warroom.agents.prompts import TIME_CONSTRAINT_PREAMBLE
from warroom.constants import LLM_MODEL_JUDGE
from warroom.graph.state import WarRoomState

logger = logging.getLogger(__name__)

RED_TEAM_PROMPT = (
  TIME_CONSTRAINT_PREAMBLE
  + """
You are the VP of Commercial Strategy at a rival Top 10 Pharma company.
Your competitor is about to launch a new oncology drug and their strategy
brief has just leaked to your desk.

Your job is to act as the "Red Team". You must completely dismantle their
launch strategy by building a highly aggressive counter-messaging plan for
your own sales force to use in the field.

Instructions:
1. Analyze their Launch Strategy Brief relentlessly.
2. Identify their critical vulnerabilities:
   - Do they have severe toxicities or black box warnings you can exploit?
   - Are their efficacy numbers (ORR, PFS) weak compared to standard of care?
   - Is their target patient population too narrow?
3. Generate a "Counter-Attack Plan" containing exactly 3 strategic
   messages your sales reps will use to convince oncologists NOT to
   prescribe their drug.
4. Keep your tone highly strategic, adversarial, and mercenary. Do not
   be polite. This is a corporate war game.
"""
)


def red_team_node(state: WarRoomState, config: RunnableConfig) -> dict:
  """Execute the Red Team Agent.

  Args:
    state: Current graph state taking the final, compliant
      strategy brief.
    config: Runnable configuration for tracing.

  Returns:
    State delta with the `competitor_counter_plan`.
  """
  logger.info("Executing Red Team War-Gaming Agent...")

  try:
    # Use the JUDGE model (gemini-2.5-pro) for superior strategic
    # adversarial reasoning, and higher temperature for creativity.
    from langchain_google_genai import HarmBlockThreshold, HarmCategory

    llm = ChatGoogleGenerativeAI(
      model=LLM_MODEL_JUDGE,
      temperature=0.6,
      safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
      },
    )

    human_prompt = (
      f"--- LEAKED COMPETITOR LAUNCH STRATEGY ---\n"
      f"{state.get('strategy_brief', 'None')}"
    )

    messages = [
      SystemMessage(content=RED_TEAM_PROMPT),
      HumanMessage(content=human_prompt),
    ]

    result = llm.invoke(messages, config=config)
    final_output = result.content

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

    return {"competitor_counter_plan": final_output}

  except Exception as e:
    logger.exception("Error in red_team_node")
    return {"competitor_counter_plan": f"ERROR in Red Team Agent: {e}"}
