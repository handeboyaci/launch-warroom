"""Tests for the analytical SQL tool."""

from warroom.tools.sql_tools import execute_analytical_sql


def test_execute_analytical_sql_formatting(mocker):
  """Verify that results are formatted as a table/list."""
  mock_data = [{"phase": "Phase 1", "count": 10}, {"phase": "Phase 2", "count": 5}]
  mocker.patch("warroom.tools.sql_tools.execute_safe_query", return_value=mock_data)

  result = execute_analytical_sql.invoke(
    {"query": "SELECT phase, COUNT(*) FROM studies GROUP BY phase"}
  )

  assert "=== Custom Query Results ===" in result
  assert "Found 2 row(s):" in result
  assert "Phase 1" in result
  assert "count: 10" in result
  assert "Phase 2" in result


def test_execute_analytical_sql_empty(mocker):
  """Verify fallback message for empty results."""
  mocker.patch("warroom.tools.sql_tools.execute_safe_query", return_value=[])

  result = execute_analytical_sql.invoke(
    {"query": "SELECT * FROM studies WHERE nct_id='FAKE'"}
  )

  assert "No results found" in result
  assert "(before January 15, 2021)" in result


def test_execute_analytical_sql_row_capping(mocker):
  """Verify truncation at 20 rows."""
  # Create 25 rows
  mock_data = [{"id": i} for i in range(25)]
  mocker.patch("warroom.tools.sql_tools.execute_safe_query", return_value=mock_data)

  result = execute_analytical_sql.invoke({"query": "SELECT * FROM studies"})

  assert "Found 25 row(s):" in result
  # It should only show 20 rows.
  # Let's check for the presence of the 19th and absence of 21st (0-indexed)
  assert "id: 19" in result
  assert "id: 20" not in result
  assert "... and 5 more results" in result
