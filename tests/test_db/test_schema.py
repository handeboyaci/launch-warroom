"""Tests for the SQLite schema module."""

import sqlite3

import pytest

from warroom.db.schema import (
  get_connection,
  init_db,
  row_count,
  table_exists,
)


def test_init_db_creates_all_tables(tmp_db):
  """init_db should create all 7 AACT tables."""
  expected_tables = [
    "studies",
    "interventions",
    "eligibilities",
    "outcomes",
    "sponsors",
    "browse_conditions",
    "browse_interventions",
  ]
  for table in expected_tables:
    assert table_exists(tmp_db, table), f"Table '{table}' was not created"


def test_init_db_is_idempotent(tmp_path):
  """Calling init_db twice should not raise errors."""
  db_path = tmp_path / "test.db"
  conn1 = init_db(db_path)
  conn1.close()
  conn2 = init_db(db_path)
  assert table_exists(conn2, "studies")
  conn2.close()


def test_row_count_empty(tmp_db):
  """Empty tables should have zero rows."""
  assert row_count(tmp_db, "studies") == 0
  assert row_count(tmp_db, "interventions") == 0


def test_foreign_keys_enabled(tmp_db):
  """Foreign key enforcement should be enabled."""
  cur = tmp_db.execute("PRAGMA foreign_keys")
  assert cur.fetchone()[0] == 1


def test_schema_columns(tmp_db):
  """Verify studies table has the expected columns."""
  cur = tmp_db.execute("PRAGMA table_info(studies)")
  columns = {row[1] for row in cur.fetchall()}
  expected = {
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
  }
  assert expected.issubset(columns)


def test_get_connection_is_read_only(tmp_path):
  """Ensure get_connection prevents destructive SQL queries."""
  db_path = tmp_path / "test.db"

  # Init DB (RW mode)
  conn_rw = init_db(db_path)
  conn_rw.execute(
    "INSERT INTO studies (nct_id, brief_title) VALUES ('NCT00000001', 'Test')"
  )
  conn_rw.commit()
  conn_rw.close()

  # Get RO connection
  conn_ro = get_connection(db_path)

  # Reads should work
  cur = conn_ro.execute("SELECT * FROM studies")
  assert len(cur.fetchall()) == 1

  # Writes/Drops should fail with sqlite3.OperationalError: attempt to write
  # a readonly database
  with pytest.raises(sqlite3.OperationalError, match="readonly database"):
    conn_ro.execute("DELETE FROM studies")

  with pytest.raises(sqlite3.OperationalError, match="readonly database"):
    conn_ro.execute("DROP TABLE studies")

  conn_ro.close()
