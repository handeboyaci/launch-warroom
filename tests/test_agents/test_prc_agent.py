"""Unit tests for the Promotional Review Committee (PRC) agent."""

from warroom.agents.prc_agent import PRCReviewResult, prc_agent_node
from warroom.graph.state import WarRoomState


def test_prc_agent_compliant(mocker):
  """PRC agent should return compliant for fact-based summaries."""

  mock_result = PRCReviewResult(compliant=True, violations=[])

  mock_structured = mocker.MagicMock()
  mock_structured.invoke.return_value = mock_result

  mock_llm = mocker.MagicMock()
  mock_llm.with_structured_output.return_value = mock_structured

  mocker.patch(
    "warroom.agents.prc_agent.ChatGoogleGenerativeAI",
    return_value=mock_llm,
  )

  state = WarRoomState(
    query="Test",
    clinical_intel="",
    literature_intel=(
      "The approved label for sotorasib indicates it for KRAS G12C mutated NSCLC."
    ),
    strategy_brief=(
      "We will target patients with KRAS G12C mutated NSCLC as per the approved label."
    ),
    citations=[],
    warnings=[],
    iteration_count=0,
    compliance_warnings=[],
    prc_iteration_count=0,
  )

  result = prc_agent_node(state, config={})
  assert result["compliance_warnings"] == []


def test_prc_agent_noncompliant(mocker):
  """PRC agent should flag off-label promotional claims."""

  mock_result = PRCReviewResult(
    compliant=False,
    violations=["Promoting use in colorectal cancer which is not FDA approved."],
  )

  mock_structured = mocker.MagicMock()
  mock_structured.invoke.return_value = mock_result

  mock_llm = mocker.MagicMock()
  mock_llm.with_structured_output.return_value = mock_structured

  mocker.patch(
    "warroom.agents.prc_agent.ChatGoogleGenerativeAI",
    return_value=mock_llm,
  )

  state = WarRoomState(
    query="Test",
    clinical_intel="",
    literature_intel=(
      "The approved label for sotorasib indicates it for KRAS G12C mutated NSCLC."
    ),
    strategy_brief=(
      "Sales teams should aggressively promote sotorasib for colorectal cancer."
    ),
    citations=[],
    warnings=[],
    iteration_count=0,
    compliance_warnings=[],
    prc_iteration_count=0,
  )

  result = prc_agent_node(state, config={})
  assert len(result["compliance_warnings"]) == 1
  assert "colorectal" in result["compliance_warnings"][0]
