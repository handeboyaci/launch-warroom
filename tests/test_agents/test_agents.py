"""Tests for the War-Room Agents.

Validates that the agents invoke tools correctly and produce
expected output shapes without needing live API calls.
"""

from langchain_core.messages import AIMessage

from warroom.agents.clinical_intel import clinical_intel_node
from warroom.agents.launch_strategist import launch_strategist_node
from warroom.agents.medical_affairs import medical_affairs_node
from warroom.graph.state import WarRoomState


def test_clinical_intel_node(mocker):
  # Mock create_react_agent to return our dummy text.
  mock_agent = mocker.MagicMock()
  mock_agent.invoke.return_value = {
    "messages": [AIMessage(content="Clinical Intel Report...")]
  }
  mocker.patch(
    "warroom.agents.clinical_intel.create_react_agent",
    return_value=mock_agent,
  )
  mocker.patch("warroom.agents.clinical_intel.ChatGoogleGenerativeAI")

  state: WarRoomState = {"query": "Test"}
  result = clinical_intel_node(state, config={})

  assert "clinical_intel" in result
  assert result["clinical_intel"] == "Clinical Intel Report..."
  mock_agent.invoke.assert_called_once()


def test_medical_affairs_node(mocker):
  mock_agent = mocker.MagicMock()
  mock_agent.invoke.return_value = {
    "messages": [AIMessage(content="Medical Affairs Report with PMID: 11111")]
  }
  mocker.patch(
    "warroom.agents.medical_affairs.create_react_agent",
    return_value=mock_agent,
  )
  mocker.patch("warroom.agents.medical_affairs.ChatGoogleGenerativeAI")

  # Mock citation validator so we don't hit the DB.
  mock_validate = mocker.patch(
    "warroom.agents.medical_affairs.validate_citations",
  )
  mock_res = mocker.MagicMock()
  mock_res.verified = ["PMID: 11111"]
  mock_validate.return_value = mock_res

  state: WarRoomState = {"query": "Test", "citations": ["existing_cit"]}
  result = medical_affairs_node(state, config={})

  assert "literature_intel" in result
  assert "citations" in result
  assert "PMID: 11111" in result["literature_intel"]
  assert "PMID: 11111" in result["citations"]
  assert "existing_cit" not in result["citations"]  # LangGraph reducer handles addition
  mock_agent.invoke.assert_called_once()


def test_launch_strategist_node(mocker):
  mock_llm = mocker.MagicMock()
  mock_llm.invoke.return_value = AIMessage(content="Final Strategy Brief...")
  mocker.patch(
    "warroom.agents.launch_strategist.ChatGoogleGenerativeAI",
    return_value=mock_llm,
  )

  state: WarRoomState = {
    "query": "Test",
    "clinical_intel": "CI summary",
    "literature_intel": "MA summary",
  }
  result = launch_strategist_node(state, config={})

  assert "strategy_brief" in result
  assert result["strategy_brief"] == "Final Strategy Brief..."
  mock_llm.invoke.assert_called_once()
