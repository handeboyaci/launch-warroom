"""Tests for the AACT CSV ingestion pipeline."""

from warroom.constants import TIME_CUTOFF
from warroom.db.ingest import ingest_all, ingest_table
from warroom.db.schema import row_count


class TestIngestTable:
  """Tests for single-table ingestion."""

  def test_ingest_studies_filters_postcutoff(self, tmp_db, sample_studies_csv):
    """Post-cutoff studies must be filtered out."""
    n = ingest_table("studies", tmp_db, raw_dir=sample_studies_csv)
    # 2 of 3 studies are before cutoff.
    assert n == 2
    assert row_count(tmp_db, "studies") == 2

  def test_no_postcutoff_dates_in_db(self, tmp_db, sample_studies_csv):
    """Verify no study_first_submitted_date >= TIME_CUTOFF."""
    ingest_table("studies", tmp_db, raw_dir=sample_studies_csv)
    cur = tmp_db.execute(
      "SELECT nct_id, study_first_submitted_date "
      "FROM studies "
      "WHERE study_first_submitted_date >= ?",
      (TIME_CUTOFF,),
    )
    violations = cur.fetchall()
    assert violations == [], f"Post-cutoff studies found: {violations}"

  def test_postcutoff_nct_id_absent(self, tmp_db, sample_studies_csv):
    """The post-cutoff study NCT99999999 must not be present."""
    ingest_table("studies", tmp_db, raw_dir=sample_studies_csv)
    cur = tmp_db.execute(
      "SELECT nct_id FROM studies WHERE nct_id = ?",
      ("NCT99999999",),
    )
    assert cur.fetchone() is None

  def test_nct_id_filter(self, tmp_db, sample_studies_csv):
    """When nct_ids is specified, only those IDs are ingested."""
    n = ingest_table(
      "studies",
      tmp_db,
      raw_dir=sample_studies_csv,
      nct_ids=["NCT03600883"],
    )
    assert n == 1

  def test_missing_csv_returns_zero(self, tmp_db, tmp_path):
    """Missing CSV file should return 0, not raise."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    n = ingest_table("studies", tmp_db, raw_dir=empty_dir)
    assert n == 0


class TestIngestAll:
  """Tests for the full ingestion pipeline."""

  def test_ingest_all_cascades_nct_filter(self, tmp_db, sample_interventions_csv):
    """Child table rows for post-cutoff studies are excluded."""
    results = ingest_all(tmp_db, raw_dir=sample_interventions_csv)

    # Studies: 2 before cutoff.
    assert results["studies"] == 2

    # Interventions: only for the 2 valid studies.
    assert results["interventions"] == 2

    # Verify post-cutoff intervention is not present.
    cur = tmp_db.execute("SELECT name FROM interventions WHERE nct_id = 'NCT99999999'")
    assert cur.fetchone() is None

  def test_ingest_all_with_sponsors(self, tmp_db, sample_sponsors_csv):
    """Sponsors for post-cutoff studies are excluded."""
    results = ingest_all(tmp_db, raw_dir=sample_sponsors_csv)
    assert results["sponsors"] == 2

    # Verify only Amgen and Mirati are present.
    cur = tmp_db.execute("SELECT name FROM sponsors ORDER BY name")
    names = [row[0] for row in cur.fetchall()]
    assert "Amgen" in names
    assert "Mirati Therapeutics" in names
    assert "FuturePharma" not in names
