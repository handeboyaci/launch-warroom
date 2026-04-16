"""Promotional Review Committee (PRC) Agent Node.

Evaluates the Launch Strategist's brief for off-label promotion
or non-compliant claims. Acts as the Medical/Legal/Regulatory
(MLR) firewall.
"""

from __future__ import annotations

import logging
from pprint import pformat
from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from warroom.constants import LLM_MODEL_JUDGE
from warroom.graph.state import WarRoomState

logger = logging.getLogger(__name__)

PRC_PROMPT = """
You are a strict Pharma Regulatory Lawyer acting as the Promotional Review
Committee (PRC). Your job is to review a Launch Strategy Brief intended for
Commercial Sales and Marketing teams.

You will be provided with:
1. The approved Medical Affairs Literature (the facts).
2. The Launch Strategy Draft (the promotional brief).

YOUR CRITICAL INSTRUCTION:
Pharma Commercial (Sales/Marketing) teams are FORBIDDEN by the FDA to
promote a drug "off-label". They can ONLY promote data, indications,
and claims that are explicitly supported by the approved label/literature.

However, certain "SAFE HARBORS" are permitted for pre-approval activities:
1. Disease State Education (DSE): General education about a disease,
   a mutation (e.g., KRAS G12C), or an unmet need is PERMITTED, provided
   the company's unapproved product is NOT mentioned or promoted.
2. Internal Competitive Intelligence (CI): Comparing competitor clinical
   trial designs, enrollment, and timelines for INTERNAL strategic
   readiness is PERMITTED and COMPLIANT.
3. Scientific Exchange: Non-promotional, objective sharing of data
   (e.g., medical conference posters) is PERMITTED for Medical Affairs,
   but not for Sales Reps.

If the Launch Strategy Draft instructs Commercial teams to:
- Promote an indication not yet approved.
- Use experimental / Phase 1 / Phase 2 data in EXTERNAL marketing materials.
- Make comparative claims against competitors without head-to-head trial data.
- Downplay severe adverse events.

Then the brief is NON-COMPLIANT. If the brief focuses on DSE, CI, or
internal readiness, it is COMPLIANT.

Output a structured JSON evaluating compliance.
If non-compliant, provide a list of specific "violations" (e.g., quotes
from the text that break the rules and why).
"""


class PRCReviewResult(BaseModel):
  """Structured output for the PRC Agent."""

  compliant: bool = Field(
    description="True if the brief is free of off-label promotion."
  )
  violations: list[str] = Field(
    description="List of specific sentences or claims that violate compliance.",
    default_factory=list,
  )


def prc_agent_node(state: WarRoomState, config: RunnableConfig) -> dict:
  """Execute the Legal/Compliance Review (PRC) Agent.

  Args:
    state: Current graph state.
    config: Runnable configuration for tracing.

  Returns:
    State delta with verification results. Does not modify
    the brief directly; either passes it or populates compliance_warnings.
  """
  logger.info("Executing PRC (Legal/Compliance) Agent...")

  try:
    from langchain_google_genai import HarmBlockThreshold, HarmCategory

    # Use the JUDGE model (typically more capable of nuanced evaluation)
    llm = ChatGoogleGenerativeAI(
      model=LLM_MODEL_JUDGE,
      temperature=0.0,
      safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
      },
    )

    # Force structured output
    structured_llm = llm.with_structured_output(PRCReviewResult)

    human_prompt = (
      f"--- Approved Medical Literature (The Facts) ---\n"
      f"{state.get('literature_intel', 'None')}\n\n"
      f"--- Launch Strategy Draft (The Promotional Material) ---\n"
      f"{state.get('strategy_brief', 'None')}"
    )

    messages = [
      SystemMessage(content=PRC_PROMPT),
      HumanMessage(content=human_prompt),
    ]

    result = cast(PRCReviewResult, structured_llm.invoke(messages, config=config))

    if result.compliant:
      logger.info("PRC Review: COMPLIANT (0 violations).")
      return {"compliance_warnings": []}

    logger.warning("PRC Review: NON-COMPLIANT (%d violations).", len(result.violations))
    logger.debug("PRC Violations:\n%s", pformat(result.violations))

    return {"compliance_warnings": result.violations}

  except Exception as e:
    logger.exception("Error in prc_agent_node")
    return {"compliance_warnings": [f"ERROR in PRC Agent: {e}"]}
