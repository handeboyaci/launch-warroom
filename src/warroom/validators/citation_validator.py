"""Citation validation for LLM outputs.

Extracts PMID and NCT ID references from agent-generated text and
cross-references them against the SQLite database and ChromaDB
metadata to verify they exist in our pre-cutoff corpus.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Regex Patterns ───────────────────────────────────────────────────

# Matches: PMID: 12345678, PMID12345678, PMID 12345678
PMID_PATTERN = re.compile(r"PMID[:\s]*(\d{6,10})", re.IGNORECASE)

# Matches: NCT12345678, NCT 12345678
NCT_PATTERN = re.compile(r"NCT[:\s]?(\d{8})", re.IGNORECASE)


@dataclass
class CitationResult:
  """Result of citation validation."""

  valid: bool
  citations_expected: bool = False
  verified: list[str] = field(default_factory=list)
  unverified: list[str] = field(default_factory=list)
  extracted_pmids: list[str] = field(default_factory=list)
  extracted_nct_ids: list[str] = field(default_factory=list)

  def __str__(self) -> str:
    parts = [
      f"Citations: {len(self.verified)} verified, {len(self.unverified)} unverified"
    ]
    if self.unverified:
      parts.append("Unverified: " + ", ".join(self.unverified))
    return " | ".join(parts)


def extract_citations(text: str) -> tuple[list[str], list[str]]:
  """Extract PMID and NCT ID references from text.

  Args:
    text: LLM-generated text to scan.

  Returns:
    Tuple of (pmids, nct_ids) — deduplicated lists.
  """
  if not isinstance(text, str):
    text = str(text)
  pmids = list(dict.fromkeys(PMID_PATTERN.findall(text)))
  raw_ncts = NCT_PATTERN.findall(text)
  nct_ids = list(dict.fromkeys(f"NCT{nct}" for nct in raw_ncts))
  return pmids, nct_ids


def validate_citations(
  text: str,
  conn=None,
  collection=None,
) -> CitationResult:
  """Validate all citations found in LLM output.

  Cross-references PMIDs against ChromaDB metadata and NCT IDs
  against the SQLite studies table.

  Args:
    text: LLM-generated text to validate.
    conn: Optional SQLite connection (for NCT ID lookups).
    collection: Optional ChromaDB collection (for PMID lookups).

  Returns:
    CitationResult with verification details.
  """
  pmids, nct_ids = extract_citations(text)

  if not pmids and not nct_ids:
    return CitationResult(valid=True, citations_expected=True)

  verified: list[str] = []
  unverified: list[str] = []

  # Verify NCT IDs against SQLite.
  if nct_ids:
    verified_ncts, unverified_ncts = _verify_nct_ids(nct_ids, conn)
    verified.extend(verified_ncts)
    unverified.extend(unverified_ncts)

  # Verify PMIDs against ChromaDB.
  if pmids:
    verified_pmids, unverified_pmids = _verify_pmids(pmids, collection)
    verified.extend(verified_pmids)
    unverified.extend(unverified_pmids)

  result = CitationResult(
    valid=len(unverified) == 0,
    verified=verified,
    unverified=unverified,
    extracted_pmids=pmids,
    extracted_nct_ids=nct_ids,
  )

  if not result.valid:
    logger.warning("Unverified citations: %s", unverified)

  return result


def _verify_nct_ids(
  nct_ids: list[str],
  conn=None,
) -> tuple[list[str], list[str]]:
  """Check if NCT IDs exist in the studies table."""
  verified: list[str] = []
  unverified: list[str] = []

  if conn is None:
    try:
      from warroom.db.schema import get_connection

      conn = get_connection()
      own_conn = True
    except Exception:
      logger.warning("Cannot verify NCT IDs: no DB connection")
      return [], nct_ids
  else:
    own_conn = False

  try:
    for nct_id in nct_ids:
      cur = conn.execute(
        "SELECT nct_id FROM studies WHERE nct_id = ?",
        (nct_id,),
      )
      if cur.fetchone():
        verified.append(nct_id)
      else:
        unverified.append(nct_id)
  finally:
    if own_conn:
      conn.close()

  return verified, unverified


def _verify_pmids(
  pmids: list[str],
  collection=None,
) -> tuple[list[str], list[str]]:
  """Check if PMIDs exist in the ChromaDB collection."""
  verified: list[str] = []
  unverified: list[str] = []

  if collection is None:
    try:
      from warroom.rag.vectorstore import get_collection

      collection = get_collection()
    except Exception:
      logger.warning("Cannot verify PMIDs: no ChromaDB collection")
      return [], pmids

  for pmid in pmids:
    try:
      # Search for documents with this PMID in metadata.
      results = collection.get(
        where={"pmid": pmid},
        limit=1,
      )
      if results and results.get("ids"):
        verified.append(f"PMID:{pmid}")
      else:
        unverified.append(f"PMID:{pmid}")
    except Exception:
      unverified.append(f"PMID:{pmid}")

  return verified, unverified
