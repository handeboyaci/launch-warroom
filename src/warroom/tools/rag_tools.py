"""RAG-based tools for the Medical Affairs Agent.

These tools query the ChromaDB vector store to retrieve published
literature and drug label information. All queries are temporally
filtered by the vectorstore layer.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from warroom.rag.vectorstore import query, query_by_drug

logger = logging.getLogger(__name__)


def _format_rag_results(results: dict, source_label: str) -> str:
  """Format ChromaDB query results into a readable string.

  Args:
    results: ChromaDB query results dict.
    source_label: Label for the search type (e.g., "Literature").

  Returns:
    Formatted string with citations for LLM consumption.
  """
  ids = results.get("ids", [[]])[0]
  docs = results.get("documents", [[]])[0]
  metas = results.get("metadatas", [[]])[0]
  distances = results.get("distances", [[]])[0]

  if not ids:
    return f"(No {source_label.lower()} results found)"

  lines: list[str] = []
  for i, (doc_id, doc, meta, dist) in enumerate(zip(ids, docs, metas, distances)):
    # Build citation header.
    source = meta.get("source", "unknown")
    pub_date = meta.get("published_date", "")
    relevance = f"{1 - dist:.2f}" if dist else "N/A"

    if source == "pubmed":
      pmid = meta.get("pmid", "")
      title = meta.get("title", "")
      journal = meta.get("journal", "")
      lines.append(f"[{i + 1}] PMID: {pmid}")
      lines.append(f"  Title: {title}")
      lines.append(f"  Journal: {journal} | Date: {pub_date}")
      lines.append(f"  Relevance: {relevance}")
    elif source == "openfda":
      drug = meta.get("drug_name", "")
      section = meta.get("section", "")
      lines.append(f"[{i + 1}] FDA Label: {drug.upper()}")
      lines.append(f"  Section: {section}")
      lines.append(f"  Effective: {pub_date}")
      lines.append(f"  Relevance: {relevance}")
    else:
      lines.append(f"[{i + 1}] {doc_id}")
      lines.append(f"  Date: {pub_date}")
      lines.append(f"  Relevance: {relevance}")

    # Truncate document text for readability.
    text = doc[:400] if doc else ""
    lines.append(f"  Content: {text}")
    lines.append("")

  return "\n".join(lines)


@tool
def search_literature(
  query_text: str,
  top_k: int = 5,
) -> str:
  """Search published biomedical literature by semantic query.

  Searches PubMed abstracts and FDA labels in the vector store.
  Returns relevant passages with citations (PMIDs, journal names,
  publication dates).

  Args:
    query_text: Natural language search query (e.g.,
      "KRAS G12C inhibitor efficacy in NSCLC",
      "sotorasib phase I clinical results").
    top_k: Number of results to return (default: 5).
  """
  results = query(
    query_text=query_text,
    top_k=top_k,
  )

  n_results = len(results.get("ids", [[]])[0])
  header = (
    f"=== Literature Search: '{query_text[:60]}' ===\n"
    f"Found {n_results} relevant document(s):\n\n"
  )
  return header + _format_rag_results(results, "Literature")


@tool
def search_drug_labels(drug_name: str) -> str:
  """Search FDA drug labels for a specific drug.

  Returns label sections including indications, dosage,
  warnings, adverse reactions, and clinical study results.

  Args:
    drug_name: Generic drug name (e.g., "pembrolizumab",
      "nivolumab", "osimertinib").
  """
  results = query_by_drug(
    drug_name=drug_name,
    query_text=(
      f"{drug_name} indications dosage warnings adverse reactions clinical studies"
    ),
    top_k=5,
  )

  n_results = len(results.get("ids", [[]])[0])
  header = f"=== FDA Labels: '{drug_name}' ===\nFound {n_results} label section(s):\n\n"
  return header + _format_rag_results(results, "Drug Label")


@tool
def search_safety_signals(drug_name: str) -> str:
  """Search for safety signals and adverse events for a drug.

  Combines FDA label adverse reactions with published safety
  literature for a comprehensive safety profile.

  Args:
    drug_name: Generic drug name (e.g., "sotorasib",
      "adagrasib", "docetaxel").
  """
  # Search for adverse reactions in labels.
  label_results = query_by_drug(
    drug_name=drug_name,
    query_text=(f"{drug_name} adverse reactions toxicity safety hepatotoxicity QTc"),
    top_k=3,
  )

  # Search broader literature for safety signals.
  lit_results = query(
    query_text=(
      f"{drug_name} safety adverse events toxicity dose-limiting side effects"
    ),
    top_k=3,
  )

  lines = [
    f"=== Safety Profile: '{drug_name}' ===",
    "",
    "--- FDA Label Safety Data ---",
    _format_rag_results(label_results, "Safety Label"),
    "--- Published Safety Literature ---",
    _format_rag_results(lit_results, "Safety Literature"),
  ]
  return "\n".join(lines)


# Export all tools as a list for agent binding.
RAG_TOOLS = [
  search_literature,
  search_drug_labels,
  search_safety_signals,
]
