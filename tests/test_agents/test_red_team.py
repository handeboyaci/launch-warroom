"""Unit tests for the Red Team War-Gaming agent."""

from warroom.agents.red_team import red_team_node
from warroom.graph.state import WarRoomState


def test_red_team_agent(mocker):
  """Red Team agent should return an adversarial strategy."""

  mock_result = mocker.MagicMock()
  mock_result.content = (
    "COUNTER-ATTACK:\n1. Attack their safety profile.\n2. Highlight our superior ORR."
  )

  mock_llm = mocker.MagicMock()
  mock_llm.invoke.return_value = mock_result

  mocker.patch(
    "warroom.agents.red_team.ChatGoogleGenerativeAI",
    return_value=mock_llm,
  )

  state = WarRoomState(
    query="Test",
    clinical_intel="",
    literature_intel="",
    strategy_brief=(
      "We will launch with a focus on our unprecedented safety "
      "profile despite the modest efficacy."
    ),
    citations=[],
    warnings=[],
    iteration_count=0,
    compliance_warnings=[],
    prc_iteration_count=0,
    competitor_counter_plan="",
  )

  result = red_team_node(state, config={})
  assert "competitor_counter_plan" in result
  assert "Attack their safety profile" in result["competitor_counter_plan"]
