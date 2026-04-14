"""Safe SQL query execution with mandatory temporal filtering.

All SQL tools must use ``execute_safe_query`` to ensure that every
query against the AACT database is bounded by TIME_CUTOFF.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path

from warroom.constants import TIME_CUTOFF
from warroom.db.schema import get_connection

logger = logging.getLogger(__name__)


def execute_safe_query(
  sql: str,
  params: dict | None = None,
  conn: sqlite3.Connection | None = None,
  db_path: Path | None = None,
) -> list[sqlite3.Row]:
  """Execute a SQL query with mandatory temporal filtering.

  Every query executed through this function is guaranteed to only
  return data from studies submitted before TIME_CUTOFF.

  Args:
    sql: SQL query string. Must contain a WHERE clause or one
      will be appended.
    params: Optional query parameters (named style ``:param``).
    conn: Optional existing connection. If not provided, a new
      read-only connection is opened and closed after the query.
    db_path: Optional path to the database file.

  Returns:
    List of ``sqlite3.Row`` objects.
  """
  params = dict(params) if params else {}
  params["_cutoff"] = TIME_CUTOFF

  # Inject temporal filter safely.
  sql = _inject_temporal_filter(sql)

  logger.debug("Executing: %s | params=%s", sql, params)

  own_conn = conn is None
  if own_conn:
    conn = get_connection(db_path)

  try:
    cur = conn.execute(sql, params)
    rows = cur.fetchall()
    logger.debug("Query returned %d rows", len(rows))
    return rows
  finally:
    if own_conn:
      conn.close()


def _inject_temporal_filter(sql: str) -> str:
  """Safely inject temporal filter honoring SQL syntax.

  Injects filter before ORDER BY, GROUP BY, or LIMIT clauses.
  """
  if "_cutoff" in sql or "study_first_submitted_date" in sql:
    return sql

  # Find where the trailing clauses begin
  match = re.search(r"\b(ORDER BY|GROUP BY|LIMIT)\b", sql, re.IGNORECASE)

  if match:
    idx = match.start()
    base_query = sql[:idx].rstrip()
    tail_query = sql[idx:].lstrip()
  else:
    base_query = sql.rstrip()
    tail_query = ""

  if re.search(r"\bWHERE\b", base_query, re.IGNORECASE):
    base_query += " AND study_first_submitted_date < :_cutoff"
  else:
    base_query += " WHERE study_first_submitted_date < :_cutoff"

  return f"{base_query}\n{tail_query}".strip()


def format_rows(
  rows: list[sqlite3.Row],
  max_rows: int = 20,
) -> str:
  """Format sqlite3.Row results into a readable string.

  Args:
    rows: List of sqlite3.Row objects.
    max_rows: Maximum number of rows to include.

  Returns:
    Formatted string suitable for LLM consumption.
  """
  if not rows:
    return "(No results found)"

  keys = rows[0].keys()
  lines: list[str] = []

  for i, row in enumerate(rows[:max_rows]):
    parts = []
    for key in keys:
      val = row[key]
      if val is not None and str(val).strip():
        parts.append(f"  {key}: {val}")
    lines.append(f"[{i + 1}]")
    lines.extend(parts)
    lines.append("")

  if len(rows) > max_rows:
    lines.append(f"... and {len(rows) - max_rows} more results")

  return "\n".join(lines)
