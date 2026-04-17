"""Unit tests for the Defense Strategist agent."""

from warroom.agents.defense_strategist import defense_strategist_node
from warroom.graph.state import WarRoomState


def test_defense_strategist_agent(mocker):
  """Defense Strategist agent should return a rebuttal."""

  mock_result = mocker.MagicMock()
  mock_result.content = (
    "REBUTTAL MEMO:\n"
    "1. The Red Team attacks our safety, but standard of care is worse.\n"
  )

  mock_llm = mocker.MagicMock()
  mock_llm.invoke.return_value = mock_result

  mocker.patch(
    "warroom.agents.defense_strategist.ChatGoogleGenerativeAI",
    return_value=mock_llm,
  )

  state = WarRoomState(
    query="Test",
    clinical_intel="We have toxicity but OS is higher.",
    literature_intel="",
    strategy_brief="We will launch.",
    citations=[],
    warnings=[],
    iteration_count=0,
    compliance_warnings=[],
    prc_iteration_count=0,
    competitor_counter_plan="They have toxicity, attack them there.",
    defense_rebuttal="",
  )

  result = defense_strategist_node(state, config={})
  assert "defense_rebuttal" in result
  assert "standard of care is worse" in result["defense_rebuttal"]
