"""Tests for the document chunking module."""

import json

import pytest

from warroom.rag.chunker import (
  Document,
  chunk_text,
  parse_openfda_labels,
  parse_pubmed_abstracts,
)


class TestChunkText:
  """Tests for the chunk_text function."""

  def test_short_text_single_chunk(self):
    """Text shorter than chunk_size returns a single chunk."""
    text = "This is a short text."
    chunks = chunk_text(text, chunk_size=512)
    assert len(chunks) == 1
    assert chunks[0] == text

  def test_empty_text(self):
    """Empty text returns an empty list."""
    assert chunk_text("") == []
    assert chunk_text("   ") == []

  def test_long_text_multiple_chunks(self):
    """Long text is split into multiple chunks."""
    text = "word " * 200  # ~1000 chars
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    # Each chunk should be ≤ chunk_size (roughly).
    for chunk in chunks:
      assert len(chunk) <= 120  # Allow some slack for boundary

  def test_overlap_creates_shared_content(self):
    """Overlapping chunks should share content at boundaries."""
    text = "A" * 200
    chunks = chunk_text(text, chunk_size=100, overlap=30)
    assert len(chunks) >= 2
    # The end of chunk[0] should overlap with start of chunk[1].

  def test_overlap_ge_chunk_size_raises(self):
    """Overlap >= chunk_size should raise ValueError."""
    with pytest.raises(ValueError, match="overlap"):
      chunk_text("A" * 200, chunk_size=100, overlap=100)

    with pytest.raises(ValueError, match="overlap"):
      chunk_text("A" * 200, chunk_size=100, overlap=200)

  def test_sentence_boundary_splitting(self):
    """Chunks should prefer to break at sentence boundaries."""
    text = (
      "First sentence here. "
      "Second sentence follows. "
      "Third sentence comes next. "
      "Fourth sentence at end."
    )
    chunks = chunk_text(text, chunk_size=60, overlap=10)
    # At least one chunk should end with a period.
    assert any(c.rstrip().endswith(".") for c in chunks)


class TestDocumentId:
  """Tests for Document.id uniqueness."""

  def test_id_with_pmid(self):
    """Documents with PMIDs have deterministic IDs."""
    doc = Document(
      text="test",
      metadata={"source": "pubmed", "pmid": "12345"},
    )
    assert doc.id == "pubmed_12345_0"

  def test_id_without_pmid_uses_hash(self):
    """Documents without PMIDs use content hash fallback."""
    doc = Document(
      text="some text",
      metadata={"source": "pubmed"},
    )
    assert doc.id.startswith("pubmed_")
    assert len(doc.id) > len("pubmed__0")  # Has a hash

  def test_different_text_different_hash_ids(self):
    """Two documents with different text should have different IDs."""
    doc1 = Document(text="text one", metadata={"source": "pubmed"})
    doc2 = Document(text="text two", metadata={"source": "pubmed"})
    assert doc1.id != doc2.id

  def test_same_text_same_hash_ids(self):
    """Two documents with identical text produce the same hash ID."""
    doc1 = Document(text="same text", metadata={"source": "pubmed"})
    doc2 = Document(text="same text", metadata={"source": "pubmed"})
    assert doc1.id == doc2.id


class TestParsePubmedAbstracts:
  """Tests for PubMed abstract parsing."""

  @pytest.fixture
  def pubmed_json(self, tmp_path):
    """Create a sample PubMed JSON file."""
    articles = [
      {
        "pmid": "11111",
        "title": "KRAS G12C Study",
        "abstract": "This is a study about KRAS G12C.",
        "authors": ["Smith J", "Doe A"],
        "journal": "Nature",
        "published_date": "2020-06-15",
      },
      {
        "pmid": "22222",
        "title": "Post-cutoff study",
        "abstract": "Should be filtered out.",
        "authors": ["Future R"],
        "journal": "Science",
        "published_date": "2021-05-01",
      },
      {
        "pmid": "33333",
        "title": "No date study",
        "abstract": "Missing publication date.",
        "authors": [],
        "journal": "BMJ",
        "published_date": "",
      },
    ]
    path = tmp_path / "pubmed_abstracts.json"
    with open(path, "w") as f:
      json.dump(articles, f)
    return path

  def test_filters_postcutoff(self, pubmed_json):
    """Post-cutoff articles should be excluded."""
    docs = parse_pubmed_abstracts(pubmed_json)
    pmids = {d.metadata["pmid"] for d in docs}
    assert "11111" in pmids
    assert "22222" not in pmids  # Post-cutoff

  def test_filters_empty_date(self, pubmed_json):
    """Articles with empty published_date should be excluded."""
    docs = parse_pubmed_abstracts(pubmed_json)
    pmids = {d.metadata["pmid"] for d in docs}
    assert "33333" not in pmids  # Empty date

  def test_metadata_populated(self, pubmed_json):
    """Document metadata should be properly populated."""
    docs = parse_pubmed_abstracts(pubmed_json)
    doc = docs[0]
    assert doc.metadata["source"] == "pubmed"
    assert doc.metadata["doc_type"] == "abstract"
    assert doc.metadata["pmid"] == "11111"
    assert doc.metadata["published_date"] == "2020-06-15"


class TestParseOpenfdaLabels:
  """Tests for OpenFDA label parsing."""

  @pytest.fixture
  def openfda_json(self, tmp_path):
    """Create a sample OpenFDA labels JSON file."""
    labels = [
      {
        "set_id": "abc-123",
        "effective_date": "20200101",
        "generic_name": "pembrolizumab",
        "brand_name": "Keytruda",
        "manufacturer": "Merck",
        "indications_and_usage": "For NSCLC treatment.",
        "adverse_reactions": "Immune-related adverse events.",
        "warnings_and_precautions": "",
        "clinical_pharmacology": "",
        "clinical_studies": "Phase III data.",
        "dosage_and_administration": "",
      },
      {
        "set_id": "def-456",
        "effective_date": "20210301",  # Post-cutoff
        "generic_name": "future_drug",
        "brand_name": "FutureBrand",
        "manufacturer": "FuturePharma",
        "indications_and_usage": "Should be filtered.",
        "adverse_reactions": "",
        "warnings_and_precautions": "",
        "clinical_pharmacology": "",
        "clinical_studies": "",
        "dosage_and_administration": "",
      },
    ]
    path = tmp_path / "openfda_labels.json"
    with open(path, "w") as f:
      json.dump(labels, f)
    return path

  def test_filters_postcutoff(self, openfda_json):
    """Post-cutoff labels should be excluded."""
    docs = parse_openfda_labels(openfda_json)
    set_ids = {d.metadata["set_id"] for d in docs}
    assert "abc-123" in set_ids
    assert "def-456" not in set_ids

  def test_sections_become_separate_docs(self, openfda_json):
    """Each non-empty label section becomes a separate doc."""
    docs = parse_openfda_labels(openfda_json)
    sections = {d.metadata["section"] for d in docs}
    # Only non-empty sections should appear.
    assert "indications" in sections
    assert "adverse_reactions" in sections
    assert "clinical_studies" in sections
    # Empty sections should not appear.
    assert "warnings" not in sections
    assert "dosage" not in sections

  def test_normalises_yyyymmdd_date(self, openfda_json):
    """YYYYMMDD dates should be normalised to YYYY-MM-DD."""
    docs = parse_openfda_labels(openfda_json)
    for doc in docs:
      pub = doc.metadata["published_date"]
      assert pub == "2020-01-01"
