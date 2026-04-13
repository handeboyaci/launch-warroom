"""SQLite schema definitions for the AACT clinical trials database.

Creates tables mirroring the AACT subset relevant to the KRAS G12C
inhibitor competitive landscape analysis. All date columns are TEXT
in ISO-8601 format for straightforward temporal filtering.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from warroom.constants import AACT_DB_PATH

# ── Schema DDL ───────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS studies (
  nct_id                      TEXT PRIMARY KEY,
  brief_title                 TEXT,
  official_title              TEXT,
  overall_status              TEXT,
  phase                       TEXT,
  enrollment                  INTEGER,
  enrollment_type             TEXT,
  start_date                  TEXT,
  completion_date             TEXT,
  study_first_submitted_date  TEXT,
  results_first_submitted_date TEXT,
  last_update_submitted_date  TEXT,
  source                      TEXT,
  brief_summary               TEXT
);

CREATE TABLE IF NOT EXISTS interventions (
  id                  INTEGER PRIMARY KEY AUTOINCREMENT,
  nct_id              TEXT NOT NULL,
  intervention_type   TEXT,
  name                TEXT,
  description         TEXT,
  FOREIGN KEY (nct_id) REFERENCES studies(nct_id)
);

CREATE TABLE IF NOT EXISTS eligibilities (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  nct_id        TEXT NOT NULL,
  criteria      TEXT,
  gender        TEXT,
  minimum_age   TEXT,
  maximum_age   TEXT,
  healthy_volunteers TEXT,
  FOREIGN KEY (nct_id) REFERENCES studies(nct_id)
);

CREATE TABLE IF NOT EXISTS outcomes (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  nct_id          TEXT NOT NULL,
  outcome_type    TEXT,
  title           TEXT,
  description     TEXT,
  time_frame      TEXT,
  population      TEXT,
  FOREIGN KEY (nct_id) REFERENCES studies(nct_id)
);

CREATE TABLE IF NOT EXISTS sponsors (
  id                    INTEGER PRIMARY KEY AUTOINCREMENT,
  nct_id                TEXT NOT NULL,
  agency_class          TEXT,
  lead_or_collaborator  TEXT,
  name                  TEXT,
  FOREIGN KEY (nct_id) REFERENCES studies(nct_id)
);

CREATE TABLE IF NOT EXISTS browse_conditions (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  nct_id    TEXT NOT NULL,
  mesh_term TEXT,
  FOREIGN KEY (nct_id) REFERENCES studies(nct_id)
);

CREATE TABLE IF NOT EXISTS browse_interventions (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  nct_id    TEXT NOT NULL,
  mesh_term TEXT,
  FOREIGN KEY (nct_id) REFERENCES studies(nct_id)
);

-- Indexes for temporal filtering and lookups
CREATE INDEX IF NOT EXISTS idx_studies_submitted
  ON studies(study_first_submitted_date);
CREATE INDEX IF NOT EXISTS idx_studies_status
  ON studies(overall_status);
CREATE INDEX IF NOT EXISTS idx_interventions_nct
  ON interventions(nct_id);
CREATE INDEX IF NOT EXISTS idx_interventions_name
  ON interventions(name);
CREATE INDEX IF NOT EXISTS idx_sponsors_nct
  ON sponsors(nct_id);
CREATE INDEX IF NOT EXISTS idx_browse_conditions_nct
  ON browse_conditions(nct_id);
CREATE INDEX IF NOT EXISTS idx_browse_conditions_mesh
  ON browse_conditions(mesh_term);
CREATE INDEX IF NOT EXISTS idx_browse_interventions_nct
  ON browse_interventions(nct_id);
"""


def init_db(db_path: Path | None = None) -> sqlite3.Connection:
  """Create the database and all tables. Returns the connection.

  Args:
    db_path: Path to the SQLite database file. Defaults to
      ``constants.AACT_DB_PATH``.

  Returns:
    An open ``sqlite3.Connection`` with WAL mode and foreign keys
    enabled.
  """
  db_path = db_path or AACT_DB_PATH
  db_path.parent.mkdir(parents=True, exist_ok=True)

  conn = sqlite3.connect(str(db_path))
  conn.execute("PRAGMA journal_mode=WAL")
  conn.execute("PRAGMA foreign_keys=ON")
  conn.executescript(SCHEMA_SQL)
  conn.commit()
  return conn


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
  """Open a connection to an existing database in Read-Only mode.

  Args:
    db_path: Path to the SQLite database file. Defaults to
      ``constants.AACT_DB_PATH``.

  Returns:
    An open, read-only ``sqlite3.Connection`` with row factory set to
    ``sqlite3.Row`` for dict-like access.
  """
  db_path = db_path or AACT_DB_PATH
  # Enforce strictly read-only mode to prevent LLM SQL Injection destructive operations
  conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
  conn.row_factory = sqlite3.Row
  conn.execute("PRAGMA foreign_keys=ON")
  return conn


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
  """Check whether a table exists in the database."""
  cur = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
    (table_name,),
  )
  return cur.fetchone() is not None


def row_count(conn: sqlite3.Connection, table_name: str) -> int:
  """Return the number of rows in a table.

  Raises:
    ValueError: If ``table_name`` is not in the AACT_TABLES
      allowlist.
  """
  from warroom.constants import AACT_TABLES

  if table_name not in AACT_TABLES:
    raise ValueError(
      f"Invalid table name: {table_name!r}. Must be one of {AACT_TABLES}"
    )
  cur = conn.execute(f"SELECT COUNT(*) FROM {table_name}")  # noqa: S608
  return cur.fetchone()[0]
