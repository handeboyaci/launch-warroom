"""Tests for the RAG tools layer."""

from warroom.tools.rag_tools import (
  search_drug_labels,
  search_literature,
  search_safety_signals,
)


def _mock_query_result():
  return {
    "ids": [["pubmed_111", "pubmed_222"]],
    "documents": [["Abstract for 111", "Abstract for 222"]],
    "metadatas": [
      [
        {
          "source": "pubmed",
          "doc_type": "abstract",
          "pmid": "11111",
          "title": "Study 1",
          "journal": "NEJM",
          "published_date": "2020-05-01",
        },
        {
          "source": "pubmed",
          "doc_type": "abstract",
          "pmid": "22222",
          "title": "Study 2",
          "journal": "Lancet",
          "published_date": "2020-08-15",
        },
      ]
    ],
    "distances": [[0.1, 0.2]],
  }


def _mock_label_result(drug_name):
  return {
    "ids": [[f"fda_{drug_name}_1"]],
    "documents": [[f"Adverse reactions for {drug_name}..."]],
    "metadatas": [
      [
        {
          "source": "openfda",
          "doc_type": "label",
          "drug_name": drug_name,
          "section": "adverse_reactions",
          "set_id": "abc-123",
          "published_date": "2020-01-01",
        }
      ]
    ],
    "distances": [[0.15]],
  }


def test_search_literature(mocker):
  mocker.patch(
    "warroom.tools.rag_tools.query",
    return_value=_mock_query_result(),
  )

  result = search_literature.invoke({"query_text": "KRAS efficacy"})

  assert "=== Literature Search: 'KRAS efficacy' ===" in result
  assert "Found 2 relevant document(s):" in result
  assert "[1] PMID: 11111" in result
  assert "Title: Study 1" in result
  assert "[2] PMID: 22222" in result


def test_search_drug_labels(mocker):
  mocker.patch(
    "warroom.tools.rag_tools.query_by_drug",
    return_value=_mock_label_result("sotorasib"),
  )

  result = search_drug_labels.invoke({"drug_name": "sotorasib"})

  assert "=== FDA Labels: 'sotorasib' ===" in result
  assert "SOTORASIB" in result
  assert "Section: adverse_reactions" in result


def test_search_safety_signals(mocker):
  mocker.patch(
    "warroom.tools.rag_tools.query_by_drug",
    return_value=_mock_label_result("sotorasib"),
  )
  mocker.patch(
    "warroom.tools.rag_tools.query",
    return_value=_mock_query_result(),
  )

  result = search_safety_signals.invoke({"drug_name": "sotorasib"})

  assert "=== Safety Profile: 'sotorasib' ===" in result
  assert "--- FDA Label Safety Data ---" in result
  assert "SOTORASIB" in result
  assert "--- Published Safety Literature ---" in result
  assert "Study 1" in result
