"""Tests for the citation validator."""

from warroom.validators.citation_validator import (
  extract_citations,
  validate_citations,
)


def test_extract_citations():
  text = """
    Trial NCT03600883 showed good results.
    According to PMID: 12345678, the toxicity was low.
    Also NCT11111111 and PMID99999999.
  """
  pmids, nct_ids = extract_citations(text)
  assert "12345678" in pmids
  assert "99999999" in pmids
  assert "NCT03600883" in nct_ids
  assert "NCT11111111" in nct_ids


def test_validate_citations_all_valid(mocker):
  # Mock the DB queries to return true (verified).
  mock_conn = mocker.MagicMock()
  mock_cur = mocker.MagicMock()
  mock_cur.fetchone.return_value = ("NCT03600883",)
  mock_conn.execute.return_value = mock_cur

  mock_col = mocker.MagicMock()
  mock_col.get.return_value = {"ids": [["doc_1"]]}

  text = "Trial NCT03600883 and PMID: 12345678."
  res = validate_citations(text, conn=mock_conn, collection=mock_col)

  assert res.valid is True
  assert len(res.unverified) == 0
  assert "NCT03600883" in res.verified
  assert "PMID:12345678" in res.verified


def test_validate_citations_unverified(mocker):
  # Mock the DB queries to return false (unverified).
  mock_conn = mocker.MagicMock()
  mock_cur = mocker.MagicMock()
  mock_cur.fetchone.return_value = None  # Not found
  mock_conn.execute.return_value = mock_cur

  mock_col = mocker.MagicMock()
  mock_col.get.return_value = {"ids": []}  # Not found

  text = "Trial NCT99999999 and PMID 99999999."
  res = validate_citations(text, conn=mock_conn, collection=mock_col)

  assert res.valid is False
  assert len(res.verified) == 0
  assert "NCT99999999" in res.unverified
  assert "PMID:99999999" in res.unverified
