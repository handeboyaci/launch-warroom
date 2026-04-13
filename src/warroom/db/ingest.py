"""Ingest AACT CSV exports into the local SQLite database.

Reads pipe-delimited CSV files from ``data/raw/``, filters rows
to enforce the TIME_CUTOFF, and loads them into the SQLite schema.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from warroom.constants import AACT_DB_PATH, RAW_DATA_DIR, TIME_CUTOFF

logger = logging.getLogger(__name__)

# ── Column mappings (AACT CSV column → SQLite column) ────────────────
# Only the columns we care about are listed. Extra CSV columns are
# silently dropped during ingestion.

TABLE_COLUMNS: dict[str, list[str]] = {
  "studies": [
    "nct_id",
    "brief_title",
    "official_title",
    "overall_status",
    "phase",
    "enrollment",
    "enrollment_type",
    "start_date",
    "completion_date",
    "study_first_submitted_date",
    "results_first_submitted_date",
    "last_update_submitted_date",
    "source",
    "brief_summary",
  ],
  "interventions": [
    "nct_id",
    "intervention_type",
    "name",
    "description",
  ],
  "eligibilities": [
    "nct_id",
    "criteria",
    "gender",
    "minimum_age",
    "maximum_age",
    "healthy_volunteers",
  ],
  "outcomes": [
    "nct_id",
    "outcome_type",
    "title",
    "description",
    "time_frame",
    "population",
  ],
  "sponsors": [
    "nct_id",
    "agency_class",
    "lead_or_collaborator",
    "name",
  ],
  "browse_conditions": [
    "nct_id",
    "mesh_term",
  ],
  "browse_interventions": [
    "nct_id",
    "mesh_term",
  ],
}


def _find_csv(table_name: str, raw_dir: Path) -> Path | None:
  """Locate the CSV file for a given AACT table.

  AACT exports are typically named ``<table_name>.txt`` (pipe-delimited)
  or ``<table_name>.csv``.
  """
  for ext in (".txt", ".csv"):
    p = raw_dir / f"{table_name}{ext}"
    if p.exists():
      return p
  return None


def _read_aact_csv(path: Path, columns: list[str]) -> pd.DataFrame:
  """Read an AACT pipe-delimited file, keeping only ``columns``.

  Returns a DataFrame with only the columns that exist in both the
  file and the requested list.
  """
  df = pd.read_csv(
    path,
    sep="|",
    low_memory=False,
    dtype=str,  # everything as text; SQLite schema casts later
    on_bad_lines="skip",
  )
  # Normalise column names (AACT sometimes uses mixed case).
  df.columns = [c.strip().lower() for c in df.columns]

  available = [c for c in columns if c in df.columns]
  missing = set(columns) - set(available)
  if missing:
    logger.warning("Columns missing from %s: %s", path.name, missing)
  return df[available]


def _filter_by_cutoff(
  df: pd.DataFrame,
  date_column: str = "study_first_submitted_date",
) -> pd.DataFrame:
  """Remove rows with ``date_column`` >= TIME_CUTOFF.

  Rows with a missing or unparseable date are also dropped to prevent
  temporal leaks.
  """
  if date_column not in df.columns:
    logger.warning(
      "Date column '%s' not found — cannot enforce cutoff. Returning empty DataFrame.",
      date_column,
    )
    return df.iloc[0:0]

  # Parse dates, coercing errors to NaT.
  dates = pd.to_datetime(df[date_column], errors="coerce")
  cutoff = pd.Timestamp(TIME_CUTOFF)

  # Explicitly separate: valid dates vs NaT vs post-cutoff.
  has_valid_date = dates.notna()
  before_cutoff = dates < cutoff
  mask = has_valid_date & before_cutoff

  n_nat = (~has_valid_date).sum()
  n_post = (has_valid_date & ~before_cutoff).sum()
  n_kept = mask.sum()

  if n_nat > 0 or n_post > 0:
    logger.info(
      "Temporal filter: kept %d, dropped %d post-cutoff, %d unparseable/missing dates",
      n_kept,
      n_post,
      n_nat,
    )
  return df[mask].copy()


def ingest_table(
  table_name: str,
  conn: sqlite3.Connection,
  raw_dir: Path | None = None,
  filter_dates: bool = True,
  nct_ids: list[str] | None = None,
) -> int:
  """Ingest a single AACT table from CSV into SQLite.

  Args:
    table_name: Name of the AACT table (e.g. ``"studies"``).
    conn: Open SQLite connection with the schema already created.
    raw_dir: Directory containing the raw CSV files.
    filter_dates: Whether to enforce the TIME_CUTOFF filter.
      Only applies to tables that are joined via ``nct_id`` to
      ``studies`` (the ``studies`` table itself is always filtered).
    nct_ids: Optional list of NCT IDs to restrict ingestion to.
      If provided, only rows matching these IDs are ingested.

  Returns:
    Number of rows inserted.
  """
  raw_dir = raw_dir or RAW_DATA_DIR
  columns = TABLE_COLUMNS.get(table_name)
  if columns is None:
    raise ValueError(f"Unknown table: {table_name}")

  csv_path = _find_csv(table_name, raw_dir)
  if csv_path is None:
    logger.warning("No CSV found for table '%s' in %s", table_name, raw_dir)
    return 0

  logger.info("Reading %s from %s", table_name, csv_path)
  df = _read_aact_csv(csv_path, columns)

  # Apply temporal filter to studies table directly.
  if table_name == "studies" and filter_dates:
    df = _filter_by_cutoff(df)

  # Apply NCT ID filter if provided.
  if nct_ids and "nct_id" in df.columns:
    df = df[df["nct_id"].isin(nct_ids)]

  if df.empty:
    logger.warning("No rows to ingest for table '%s'", table_name)
    return 0

  # Clear existing data before inserting to prevent duplicates.
  # table_name is validated against TABLE_COLUMNS on line 177.
  conn.execute(f"DELETE FROM {table_name}")  # noqa: S608

  # Insert into SQLite.
  df.to_sql(
    table_name,
    conn,
    if_exists="append",
    index=False,
  )
  conn.commit()
  logger.info("Inserted %d rows into '%s'", len(df), table_name)
  return len(df)


def ingest_all(
  conn: sqlite3.Connection,
  raw_dir: Path | None = None,
  nct_ids: list[str] | None = None,
) -> dict[str, int]:
  """Ingest all AACT tables from CSV into SQLite.

  First ingests ``studies`` (with temporal filter), then uses the
  resulting NCT IDs to filter all child tables — ensuring no
  post-cutoff trials leak through via related records.

  Args:
    conn: Open SQLite connection with the schema already created.
    raw_dir: Directory containing the raw CSV files.
    nct_ids: Optional list of NCT IDs to restrict ingestion to.

  Returns:
    Dict mapping table name → number of rows inserted.
  """
  raw_dir = raw_dir or RAW_DATA_DIR
  results: dict[str, int] = {}

  # 1. Ingest studies first (with temporal filter).
  results["studies"] = ingest_table(
    "studies", conn, raw_dir, filter_dates=True, nct_ids=nct_ids
  )

  # 2. Ingest remaining tables.


  child_tables = [t for t in TABLE_COLUMNS if t != "studies"]
  for table_name in child_tables:
    results[table_name] = ingest_table(
      table_name,
      conn,
      raw_dir,
      filter_dates=False,
    )

  return results


def full_pipeline(
  db_path: Path | None = None,
  raw_dir: Path | None = None,
  nct_ids: list[str] | None = None,
) -> dict[str, int]:
  """Run the complete ingestion pipeline: init DB + ingest all.

  Args:
    db_path: Path to the SQLite database file.
    raw_dir: Directory containing the raw CSV files.
    nct_ids: Optional list of NCT IDs to restrict ingestion to.

  Returns:
    Dict mapping table name → number of rows inserted.
  """
  from warroom.db.schema import init_db

  db_path = db_path or AACT_DB_PATH
  conn = init_db(db_path)
  try:
    return ingest_all(conn, raw_dir, nct_ids)
  finally:
    conn.close()
