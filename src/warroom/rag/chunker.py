"""Document chunking for PubMed abstracts and FDA labels.

Parses raw JSON exports into standardized ``Document`` objects with
metadata, ready for embedding and ingestion into ChromaDB.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from warroom.constants import date_to_int, is_before_cutoff

logger = logging.getLogger(__name__)


@dataclass
class Document:
  """A text chunk with metadata for vector store ingestion."""

  text: str
  metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

  @property
  def id(self) -> str:
    """Unique identifier for the document chunk."""
    source = self.metadata.get("source", "unknown")
    doc_id = self.metadata.get("pmid") or self.metadata.get("set_id", "")
    chunk_idx = self.metadata.get("chunk_index", "0")
    if not doc_id:
      # Fallback: hash the text content to avoid collisions
      # when multiple documents lack a PMID/set_id.
      doc_id = hashlib.md5(  # noqa: S324
        self.text.encode()
      ).hexdigest()[:12]
    return f"{source}_{doc_id}_{chunk_idx}"


def chunk_text(
  text: str,
  chunk_size: int = 512,
  overlap: int = 64,
) -> list[str]:
  """Split text into overlapping chunks by word boundaries.

  Args:
    text: Input text to split.
    chunk_size: Maximum number of characters per chunk.
    overlap: Number of overlapping characters between chunks.

  Returns:
    List of text chunks.
  """
  if not text or not text.strip():
    return []
  if len(text) <= chunk_size:
    return [text]

  if overlap >= chunk_size:
    raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

  chunks: list[str] = []
  start = 0
  while start < len(text):
    end = start + chunk_size

    # Try to break at a sentence boundary.
    if end < len(text):
      # Look for period, question mark, or newline near the end.
      for sep in (".\n", ". ", "? ", "! ", "\n"):
        idx = text.rfind(sep, start + chunk_size // 2, end)
        if idx != -1:
          end = idx + len(sep)
          break

    chunks.append(text[start:end].strip())
    # Ensure forward progress even if end was moved back significantly
    # by sentence boundary detection and overlap is large.
    next_start = end - overlap
    start = max(start + 1, next_start)

  return [c for c in chunks if c]


def parse_pubmed_abstracts(
  json_path: Path,
) -> list[Document]:
  """Parse PubMed abstracts JSON into Document objects.

  Args:
    json_path: Path to the pubmed_abstracts.json file.

  Returns:
    List of Document objects with metadata.
  """
  with open(json_path) as f:
    articles = json.load(f)

  documents: list[Document] = []
  for article in articles:
    pub_date = article.get("published_date", "")

    # Enforce time cutoff (reject empty, unparseable, or
    # post-cutoff dates).
    if not is_before_cutoff(pub_date):
      logger.warning(
        "Skipping PMID %s: published_date=%r (missing, unparseable, or after cutoff)",
        article.get("pmid"),
        pub_date,
      )
      continue

    # Combine title + abstract for richer context.
    title = article.get("title", "")
    abstract = article.get("abstract", "")
    if not abstract:
      continue

    full_text = f"{title}\n\n{abstract}"
    chunks = chunk_text(full_text)

    for i, chunk in enumerate(chunks):
      doc = Document(
        text=chunk,
        metadata={
          "source": "pubmed",
          "doc_type": "abstract",
          "pmid": str(article.get("pmid") or ""),
          "title": str(title),
          "journal": str(article.get("journal") or ""),
          "published_date": str(pub_date),
          "published_date_int": date_to_int(pub_date),
          "authors": ", ".join(article.get("authors") or [])[
            :200
          ],  # Keep it reasonable
          "chunk_index": str(i),
        },
      )
      documents.append(doc)

  logger.info(
    "Parsed %d chunks from %d PubMed articles",
    len(documents),
    len(articles),
  )
  return documents


def parse_openfda_labels(
  json_path: Path,
) -> list[Document]:
  """Parse OpenFDA labels JSON into Document objects.

  Each label section (indications, adverse reactions, etc.) becomes
  a separate document for more targeted retrieval.

  Args:
    json_path: Path to the openfda_labels.json file.

  Returns:
    List of Document objects with metadata.
  """
  with open(json_path) as f:
    labels = json.load(f)

  # Sections to extract as separate documents.
  sections = [
    ("indications_and_usage", "indications"),
    ("warnings_and_precautions", "warnings"),
    ("adverse_reactions", "adverse_reactions"),
    ("clinical_pharmacology", "pharmacology"),
    ("clinical_studies", "clinical_studies"),
    ("dosage_and_administration", "dosage"),
  ]

  documents: list[Document] = []
  for label in labels:
    effective_date = label.get("effective_date", "")
    # Normalise YYYYMMDD → YYYY-MM-DD.
    if len(effective_date) == 8:
      effective_date = (
        f"{effective_date[:4]}-{effective_date[4:6]}-{effective_date[6:]}"
      )

    # Enforce time cutoff.
    if not is_before_cutoff(effective_date):
      logger.warning(
        "Skipping label %s: effective_date=%r (missing, unparseable, or after cutoff)",
        label.get("set_id"),
        effective_date,
      )
      continue

    drug_name = label.get("generic_name") or label.get("brand_name") or "unknown"

    for field_key, section_name in sections:
      text = label.get(field_key, "")
      if not text:
        continue

      chunks = chunk_text(text)
      for i, chunk in enumerate(chunks):
        doc = Document(
          text=f"[{drug_name.upper()} — {section_name}]\n{chunk}",
          metadata={
            "source": "openfda",
            "doc_type": "label",
            "section": section_name,
            "set_id": label.get("set_id", ""),
            "drug_name": drug_name.lower(),
            "brand_name": label.get("brand_name", ""),
            "published_date": effective_date,
            "manufacturer": label.get("manufacturer", ""),
            "chunk_index": str(i),
          },
        )
        documents.append(doc)

  logger.info(
    "Parsed %d chunks from %d FDA labels",
    len(documents),
    len(labels),
  )
  return documents
