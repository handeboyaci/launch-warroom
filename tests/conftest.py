"""Shared test fixtures for the Oncology War-Room test suite."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from warroom.db.schema import init_db


@pytest.fixture
def tmp_db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
  """Create a temporary SQLite database with the AACT schema.

  Yields an open connection. The database is automatically
  cleaned up after the test.
  """
  db_path = tmp_path / "test_aact.db"
  conn = init_db(db_path)
  yield conn
  conn.close()


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
  """Create a temporary data directory structure."""
  raw_dir = tmp_path / "raw"
  raw_dir.mkdir()
  processed_dir = tmp_path / "processed"
  processed_dir.mkdir()
  return tmp_path


@pytest.fixture
def sample_studies_csv(tmp_path: Path) -> Path:
  """Create a sample AACT studies CSV file for testing.

  Contains 3 studies: 2 before cutoff, 1 after cutoff.
  """
  raw_dir = tmp_path / "raw"
  raw_dir.mkdir(exist_ok=True)

  csv_content = (
    "nct_id|brief_title|official_title|overall_status|phase"
    "|enrollment|enrollment_type|start_date|completion_date"
    "|study_first_submitted_date|results_first_submitted_date"
    "|last_update_submitted_date|source|brief_summary\n"
    # Study 1: Before cutoff (2019).
    "NCT03600883|CodeBreaK 100|Phase I/II AMG 510|"
    "Active, not recruiting|Phase 1/Phase 2|"
    "713|Actual|2018-07-17|2024-01-15|"
    "2018-07-25|||Amgen|"
    "AMG 510 in KRAS G12C solid tumors\n"
    # Study 2: Before cutoff (2020).
    "NCT03785249|KRYSTAL-1|Phase I/II MRTX849|"
    "Recruiting|Phase 1/Phase 2|"
    "400|Anticipated|2019-01-11|2023-06-30|"
    "2018-12-26|||Mirati Therapeutics|"
    "MRTX849 in KRAS G12C solid tumors\n"
    # Study 3: AFTER cutoff (should be filtered out).
    "NCT99999999|PostCutoff Trial|Phase III Post|"
    "Not yet recruiting|Phase 3|"
    "500|Anticipated|2021-06-01|2025-12-31|"
    "2021-03-15|||FuturePharma|"
    "This trial is after the cutoff\n"
  )

  csv_path = raw_dir / "studies.txt"
  csv_path.write_text(csv_content)
  return raw_dir


@pytest.fixture
def sample_interventions_csv(sample_studies_csv: Path) -> Path:
  """Create a sample interventions CSV alongside the studies CSV."""
  csv_content = (
    "nct_id|intervention_type|name|description\n"
    "NCT03600883|Drug|AMG 510|KRAS G12C inhibitor\n"
    "NCT03785249|Drug|MRTX849|KRAS G12C inhibitor\n"
    "NCT99999999|Drug|FutureDrug|Post-cutoff drug\n"
  )
  csv_path = sample_studies_csv / "interventions.txt"
  csv_path.write_text(csv_content)
  return sample_studies_csv


@pytest.fixture
def sample_sponsors_csv(sample_studies_csv: Path) -> Path:
  """Create a sample sponsors CSV alongside the studies CSV."""
  csv_content = (
    "nct_id|agency_class|lead_or_collaborator|name\n"
    "NCT03600883|Industry|lead|Amgen\n"
    "NCT03785249|Industry|lead|Mirati Therapeutics\n"
    "NCT99999999|Industry|lead|FuturePharma\n"
  )
  csv_path = sample_studies_csv / "sponsors.txt"
  csv_path.write_text(csv_content)
  return sample_studies_csv
