"""Tests for the SQL tools layer."""

import pytest

from warroom.tools.sql_tools import (
  query_competitor_landscape,
  query_trial_timeline,
  query_trials_by_intervention,
)


@pytest.fixture
def mock_db_connection(mocker):
  """Mock the database connection for testing SQL tools."""
  mock_conn = mocker.MagicMock()
  mock_cur = mocker.MagicMock()
  mock_conn.execute.return_value = mock_cur
  mocker.patch(
    "warroom.tools.query_builder.get_connection",
    return_value=mock_conn,
  )
  mocker.patch(
    "warroom.tools.sql_tools.get_connection",
    return_value=mock_conn,
  )
  return mock_conn, mock_cur


def test_query_trials_by_intervention_no_results(mocker):
  """Tool returns a fallback message when no results are found."""
  # Mock execute_safe_query directly to avoid DB connection details.
  mocker.patch(
    "warroom.tools.sql_tools.execute_safe_query",
    return_value=[],
  )

  result = query_trials_by_intervention.invoke({"drug_name": "sotorasib"})

  assert "No clinical trials found matching intervention 'sotorasib'" in result
  assert "(before January 15, 2021)" in result


def test_query_trials_by_intervention_with_results(mocker):
  """Tool formats multiple results properly."""
  mocker.patch(
    "warroom.tools.sql_tools.execute_safe_query",
    return_value=[
      {
        "nct_id": "NCT03600883",
        "brief_title": "CodeBreaK 100",
        "phase": "Phase 1/Phase 2",
        "overall_status": "Active, not recruiting",
        "enrollment": 713,
        "enrollment_type": "Actual",
        "start_date": "2018-07-17",
        "completion_date": "2024-01-15",
        "study_first_submitted_date": "2018-07-25",
        "source": "Amgen",
        "intervention_name": "AMG 510",
        "intervention_type": "Drug",
      }
    ],
  )

  result = query_trials_by_intervention.invoke({"drug_name": "amg 510"})

  assert "=== Clinical Trials for 'amg 510' ===" in result
  assert "Found 1 matching trial record(s):" in result
  assert "NCT03600883" in result
  assert "CodeBreaK 100" in result
  assert "Phase 1/Phase 2" in result
  assert "Amgen" in result


def test_query_trial_timeline_no_results(mocker):
  """Tool returns a fallback message when trial is not found."""
  mocker.patch(
    "warroom.tools.sql_tools.execute_safe_query",
    return_value=[],
  )

  result = query_trial_timeline.invoke({"nct_id": "NCT99999999"})

  assert "No trial found with ID 'NCT99999999'" in result


def test_query_trial_timeline_full_profile(mocker):
  """Tool returns a fully formatted trial profile."""
  mocker.patch(
    "warroom.tools.sql_tools.execute_safe_query",
    return_value=[
      {
        "nct_id": "NCT03600883",
        "brief_title": "CodeBreaK 100",
        "official_title": "A Phase 1/2 Study...",
        "phase": "Phase 1/Phase 2",
        "overall_status": "Active, not recruiting",
        "enrollment": 713,
        "enrollment_type": "Actual",
        "start_date": "2018-07-17",
        "completion_date": "2024-01-15",
        "study_first_submitted_date": "2018-07-25",
        "source": "Amgen",
        "brief_summary": "This is a study of AMG 510.",
      }
    ],
  )

  mocker.patch(
    "warroom.tools.sql_tools._query_child_table",
    side_effect=lambda sql, nct_id, conn: (
      [{"name": "AMG 510", "intervention_type": "Drug", "description": ""}]
      if "interventions" in sql
      else [
        {
          "criteria": "Inc: Adults with NSCLC",
          "gender": "All",
          "minimum_age": "18 Years",
          "maximum_age": "N/A",
        }
      ]
      if "eligibilities" in sql
      else [
        {
          "outcome_type": "Primary",
          "title": "Objective Response Rate",
          "time_frame": "Up to 3 years",
        }
      ]
      if "outcomes" in sql
      else [
        {"name": "Amgen", "agency_class": "Industry", "lead_or_collaborator": "lead"}
      ]
    ),
  )

  # Mock get_connection where it's actually imported from
  mocker.patch("warroom.db.schema.get_connection", return_value=mocker.MagicMock())

  result = query_trial_timeline.invoke({"nct_id": "NCT03600883"})

  assert "=== Trial: NCT03600883 ===" in result
  assert "CodeBreaK 100" in result
  assert "--- Interventions ---" in result
  assert "AMG 510 (Drug)" in result
  assert "--- Eligibility ---" in result
  assert "Inc: Adults with NSCLC" in result
  assert "--- Outcomes ---" in result
  assert "[Primary] Objective Response Rate" in result
  assert "--- Sponsors ---" in result
  assert "Amgen (Industry, lead)" in result


def test_query_competitor_landscape_found(mocker):
  """Tool formats a phase-grouped overview of competitors."""
  mocker.patch(
    "warroom.tools.sql_tools.execute_safe_query",
    return_value=[
      {
        "nct_id": "NCT03600883",
        "brief_title": "CodeBreaK 100",
        "phase": "Phase 1/Phase 2",
        "overall_status": "Active, not recruiting",
        "enrollment": 713,
        "source": "Amgen",
        "start_date": "2018-07-17",
        "intervention_name": "AMG 510",
      },
      {
        "nct_id": "NCT03785249",
        "brief_title": "KRYSTAL-1",
        "phase": "Phase 1/Phase 2",
        "overall_status": "Recruiting",
        "enrollment": 400,
        "source": "Mirati Therapeutics",
        "start_date": "2019-01-11",
        "intervention_name": "MRTX849",
      },
      {
        "nct_id": "NCT11111111",
        "brief_title": "Other Trial",
        "phase": "Phase 3",
        "overall_status": "Not yet recruiting",
        "enrollment": 500,
        "source": "OtherSponsor",
        "start_date": "2020-01-01",
        "intervention_name": "Drug X",
      },
    ],
  )

  result = query_competitor_landscape.invoke({"condition": "NSCLC"})

  assert "=== Competitive Landscape: 'NSCLC' ===" in result
  assert "--- Phase 1/Phase 2 (2 records) ---" in result
  assert "CodeBreaK 100" in result
  assert "Mirati Therapeutics" in result
  assert "--- Phase 3 (1 records) ---" in result
  assert "OtherSponsor" in result
