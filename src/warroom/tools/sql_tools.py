"""SQL-based tools for the Clinical Intel Agent.

These tools query the AACT SQLite database to retrieve structured
clinical trial data. All queries are temporally filtered via
``query_builder.execute_safe_query``.
"""

from __future__ import annotations

import logging
import sqlite3

from langchain_core.tools import tool

from warroom.tools.query_builder import execute_safe_query, format_rows

logger = logging.getLogger(__name__)


@tool
def query_trials_by_intervention(
  drug_name: str,
) -> str:
  """Search clinical trials by drug/intervention name.

  Returns a summary of trials matching the drug name, including
  phase, status, enrollment, and sponsor.

  Args:
    drug_name: Drug name to search for (e.g., "sotorasib",
      "AMG 510", "MRTX849"). Case-insensitive partial match.
  """
  sql = """
    SELECT
      s.nct_id,
      s.brief_title,
      s.phase,
      s.overall_status,
      s.enrollment,
      s.enrollment_type,
      s.start_date,
      s.completion_date,
      s.study_first_submitted_date,
      s.source,
      i.name AS intervention_name,
      i.intervention_type
    FROM studies s
    JOIN interventions i ON s.nct_id = i.nct_id
    WHERE LOWER(i.name) LIKE :drug_pattern
  """
  rows = execute_safe_query(sql, {"drug_pattern": f"%{drug_name.lower()}%"})

  if not rows:
    return (
      f"No clinical trials found matching "
      f"intervention '{drug_name}' "
      f"(before January 15, 2021)."
    )

  header = (
    f"=== Clinical Trials for '{drug_name}' ===\n"
    f"Found {len(rows)} matching trial record(s):\n"
  )
  return header + format_rows(rows)


@tool
def query_trial_timeline(nct_id: str) -> str:
  """Get detailed timeline and design for a specific clinical trial.

  Returns comprehensive trial information including dates,
  interventions, eligibility criteria, and primary outcomes.

  Args:
    nct_id: ClinicalTrials.gov identifier (e.g., "NCT03600883").
  """
  # Normalize input.
  nct_id = nct_id.strip().upper()

  # Trial overview.
  study_sql = """
    SELECT * FROM studies
    WHERE nct_id = :nct_id
  """
  studies = execute_safe_query(study_sql, {"nct_id": nct_id})
  if not studies:
    return f"No trial found with ID '{nct_id}' (before January 15, 2021)."

  study = studies[0]
  lines = [
    f"=== Trial: {nct_id} ===",
    f"Title: {study['brief_title']}",
    f"Official Title: {study['official_title']}",
    f"Phase: {study['phase']}",
    f"Status: {study['overall_status']}",
    f"Enrollment: {study['enrollment']} ({study['enrollment_type']})",
    f"Start Date: {study['start_date']}",
    f"Completion Date: {study['completion_date']}",
    f"First Submitted: {study['study_first_submitted_date']}",
    f"Sponsor: {study['source']}",
    "",
    f"Summary: {study['brief_summary']}",
    "",
  ]

  # Open a single connection for all child table reads.
  from warroom.db.schema import get_connection

  conn = get_connection()
  try:
    # Interventions.
    interv_sql = """
      SELECT name, intervention_type, description
      FROM interventions
      WHERE nct_id = :nct_id
    """
    interventions = _query_child_table(interv_sql, nct_id, conn)
    if interventions:
      lines.append("--- Interventions ---")
      for row in interventions:
        lines.append(f"  • {row['name']} ({row['intervention_type']})")
        if row["description"]:
          desc = str(row["description"])[:200]
          lines.append(f"    {desc}")
      lines.append("")

    # Eligibility.
    elig_sql = """
      SELECT criteria, gender, minimum_age, maximum_age
      FROM eligibilities
      WHERE nct_id = :nct_id
    """
    eligibilities = _query_child_table(elig_sql, nct_id, conn)
    if eligibilities:
      elig = eligibilities[0]
      lines.append("--- Eligibility ---")
      lines.append(f"  Gender: {elig['gender']}")
      lines.append(f"  Age: {elig['minimum_age']} – {elig['maximum_age']}")
      if elig["criteria"]:
        criteria = str(elig["criteria"])[:500]
        lines.append(f"  Criteria: {criteria}")
      lines.append("")

    # Outcomes.
    out_sql = """
      SELECT outcome_type, title, time_frame
      FROM outcomes
      WHERE nct_id = :nct_id
    """
    outcomes = _query_child_table(out_sql, nct_id, conn)
    if outcomes:
      lines.append("--- Outcomes ---")
      for row in outcomes:
        lines.append(
          f"  • [{row['outcome_type']}] {row['title']} "
          f"(Time frame: {row['time_frame']})"
        )
      lines.append("")

    # Sponsors.
    spon_sql = """
      SELECT name, agency_class, lead_or_collaborator
      FROM sponsors
      WHERE nct_id = :nct_id
    """
    sponsors = _query_child_table(spon_sql, nct_id, conn)
    if sponsors:
      lines.append("--- Sponsors ---")
      for row in sponsors:
        lines.append(
          f"  • {row['name']} ({row['agency_class']}, {row['lead_or_collaborator']})"
        )
  finally:
    conn.close()

  return "\n".join(lines)


@tool
def query_competitor_landscape(condition: str) -> str:
  """Analyze the competitive landscape for a disease condition.

  Returns a phase-grouped overview of all trials targeting the
  condition, with enrollment numbers and sponsors.

  Args:
    condition: Disease condition or MeSH term to search for
      (e.g., "NSCLC", "lung cancer", "KRAS").
  """
  sql = """
    SELECT
      s.nct_id,
      s.brief_title,
      s.phase,
      s.overall_status,
      s.enrollment,
      s.source,
      s.start_date,
      i.name AS intervention_name
    FROM studies s
    JOIN browse_conditions bc ON s.nct_id = bc.nct_id
    LEFT JOIN interventions i ON s.nct_id = i.nct_id
    WHERE LOWER(bc.mesh_term) LIKE :condition_pattern
  """
  rows = execute_safe_query(sql, {"condition_pattern": f"%{condition.lower()}%"})

  if not rows:
    # Fallback: search brief_title and brief_summary.
    fallback_sql = """
      SELECT
        s.nct_id,
        s.brief_title,
        s.phase,
        s.overall_status,
        s.enrollment,
        s.source,
        s.start_date
      FROM studies s
      WHERE LOWER(s.brief_title) LIKE :cond
        OR LOWER(s.brief_summary) LIKE :cond
    """
    rows = execute_safe_query(
      fallback_sql,
      {"cond": f"%{condition.lower()}%"},
    )

  if not rows:
    return f"No trials found for condition '{condition}' (before January 15, 2021)."

  # Group by phase for analysis.
  by_phase: dict[str, list] = {}
  for row in rows:
    phase = row["phase"] or "Not specified"
    by_phase.setdefault(phase, []).append(row)

  lines = [
    f"=== Competitive Landscape: '{condition}' ===",
    f"Total matching records: {len(rows)}",
    "",
  ]

  for phase in sorted(by_phase.keys()):
    phase_rows = by_phase[phase]
    lines.append(f"--- {phase} ({len(phase_rows)} records) ---")
    # Deduplicate by nct_id for display.
    seen_ncts: set[str] = set()
    for row in phase_rows:
      nct = row["nct_id"]
      if nct in seen_ncts:
        continue
      seen_ncts.add(nct)
      intervention = (
        row["intervention_name"] if "intervention_name" in row.keys() else "N/A"
      )
      lines.append(f"  {nct} | {row['brief_title'][:60]}")
      lines.append(
        f"    Status: {row['overall_status']} | "
        f"Enrollment: {row['enrollment']} | "
        f"Sponsor: {row['source']}"
      )
      if intervention and intervention != "N/A":
        lines.append(f"    Intervention: {intervention}")
    lines.append("")

  return "\n".join(lines)


def _query_child_table(
  sql: str,
  nct_id: str,
  conn: sqlite3.Connection | None = None,
) -> list[sqlite3.Row]:
  """Query a child table by nct_id without temporal filter.

  Child tables don't have date columns — their temporal integrity
  is guaranteed by the studies table cascade during ingestion.

  Args:
    sql: SQL query with ``:nct_id`` parameter.
    nct_id: Trial NCT ID.
    conn: Optional existing connection to reuse.
  """
  own_conn = conn is None
  if own_conn:
    from warroom.db.schema import get_connection

    conn = get_connection()
  try:
    cur = conn.execute(sql, {"nct_id": nct_id})
    return cur.fetchall()
  finally:
    if own_conn:
      conn.close()


@tool
def execute_analytical_sql(query: str) -> str:
  """Execute a raw SQL query against the AACT clinical trial database.

  Use this for analytical operations like COUNT, GROUP BY, and complex
  JOINs across 'studies', 'interventions', 'browse_conditions', and 'sponsors' tables.
  The temporal cutoff filter is automatically applied.

  Args:
    query: Valid SQLite query
      (e.g., "SELECT phase, COUNT(*) FROM studies GROUP BY phase").
  """
  try:
    rows = execute_safe_query(query)
    if not rows:
      return "No results found for your query (before January 15, 2021)."
    header = f"=== Custom Query Results ===\nFound {len(rows)} row(s):\n"
    return header + format_rows(rows, max_rows=20)
  except Exception as e:
    return f"SQL Execution Error: {e}"


# Export all tools as a list for agent binding.
SQL_TOOLS = [
  query_trials_by_intervention,
  query_trial_timeline,
  query_competitor_landscape,
  execute_analytical_sql,
]
