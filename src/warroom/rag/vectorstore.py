"""ChromaDB vector store with mandatory temporal filtering.

Every query automatically enforces ``published_date < TIME_CUTOFF``
to prevent temporal leaks in the RAG pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb

from warroom.constants import CHROMA_DIR, TIME_CUTOFF_INT, is_before_cutoff
from warroom.rag.chunker import Document
from warroom.rag.embeddings import get_embedding_function

logger = logging.getLogger(__name__)

COLLECTION_NAME = "warroom_docs"


# Initialize a single global client to avoid Rust binding conflicts
# in concurrent graph execution.
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
_CLIENT_INSTANCE = chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_client(
  persist_dir: Path | None = None,
) -> chromadb.ClientAPI:
  """Get the global persistent ChromaDB client."""
  return _CLIENT_INSTANCE


def get_collection(
  client: chromadb.ClientAPI | None = None,
  persist_dir: Path | None = None,
) -> chromadb.Collection:
  """Get or create the vector store collection.

  Args:
    client: Optional existing ChromaDB client.
    persist_dir: Directory for ChromaDB persistence.

  Returns:
    The ChromaDB collection configured with the PubMedBERT
    embedding function.
  """
  if client is None:
    client = get_client(persist_dir)

  embedding_fn = get_embedding_function()
  return client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},
  )


def add_documents(
  documents: list[Document],
  collection: chromadb.Collection | None = None,
  batch_size: int = 100,
) -> int:
  """Add documents to the vector store.

  Validates that all documents have a ``published_date`` before
  the TIME_CUTOFF. Documents missing a date or with a post-cutoff
  date are rejected.

  Args:
    documents: List of Document objects to add.
    collection: Optional existing collection.
    batch_size: Number of documents to add per batch.

  Returns:
    Number of documents successfully added.
  """
  if collection is None:
    collection = get_collection()

  # Validate temporal constraint.
  valid_docs: list[Document] = []
  for doc in documents:
    pub_date = str(doc.metadata.get("published_date", ""))
    if not is_before_cutoff(pub_date):
      logger.warning(
        "Rejecting doc %s: published_date=%r (missing, unparseable, or after cutoff)",
        doc.id,
        pub_date,
      )
      continue
    valid_docs.append(doc)

  if not valid_docs:
    logger.warning("No valid documents to add.")
    return 0

  # Add in batches.
  added = 0
  for i in range(0, len(valid_docs), batch_size):
    batch = valid_docs[i : i + batch_size]
    collection.add(
      ids=[d.id for d in batch],
      documents=[d.text for d in batch],
      metadatas=[d.metadata for d in batch],
    )
    added += len(batch)

  logger.info(
    "Added %d documents to collection '%s' (%d rejected)",
    added,
    COLLECTION_NAME,
    len(documents) - added,
  )
  return added


def query(
  query_text: str,
  top_k: int = 5,
  collection: chromadb.Collection | None = None,
  extra_where: dict | None = None,
) -> Any:
  """Query the vector store with mandatory temporal filtering.

  Every query automatically includes:
    ``where={"published_date": {"$lt": TIME_CUTOFF}}``

  Args:
    query_text: The search query.
    top_k: Number of results to return.
    collection: Optional existing collection.
    extra_where: Additional metadata filters to AND with the
      temporal filter.

  Returns:
    ChromaDB query results dict with keys: ``ids``,
    ``documents``, ``metadatas``, ``distances``.
  """
  if collection is None:
    collection = get_collection()

  # Build the where clause with mandatory temporal filter.
  temporal_filter = {"published_date": {"$lt": TIME_CUTOFF}}

  if extra_where:
    where_clause = {"$and": [temporal_filter, extra_where]}
  else:
    where_clause = temporal_filter

  results = collection.query(
    query_texts=[query_text],
    n_results=top_k,
    where=where_clause,
    include=["documents", "metadatas", "distances"],
  )

  logger.debug(
    "Query returned %d results for: %s",
    len(results.get("ids", [[]])[0]),
    query_text[:80],
  )
  return results


def query_by_drug(
  drug_name: str,
  query_text: str | None = None,
  top_k: int = 5,
  collection: chromadb.Collection | None = None,
) -> dict:
  """Query for documents related to a specific drug.

  Combines semantic search with a drug name metadata filter.

  Args:
    drug_name: Drug name to filter by.
    query_text: Optional semantic query. If not provided, uses
      the drug name as the query.
    top_k: Number of results to return.
    collection: Optional existing collection.

  Returns:
    ChromaDB query results.
  """
  search_text = query_text or f"{drug_name} clinical trial safety efficacy"
  return query(
    query_text=search_text,
    top_k=top_k,
    collection=collection,
    extra_where={"drug_name": drug_name.lower()},
  )
