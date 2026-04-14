"""Embedding model configuration for the RAG pipeline.

Uses ``pritamdeka/S-PubMedBert-MS-MARCO`` via sentence-transformers
for biomedical-specialized embeddings.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from warroom.constants import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_function():
  """Return a ChromaDB-compatible embedding function.

  Uses sentence-transformers with the PubMedBERT model. The model
  is downloaded on first use (~400 MB) and cached locally.

  Returns:
    A ChromaDB ``EmbeddingFunction`` instance.
  """
  try:
    from chromadb.utils.embedding_functions import (
      SentenceTransformerEmbeddingFunction,
    )
  except ImportError as e:
    raise ImportError(
      "chromadb is required. Install with: pip install chromadb sentence-transformers"
    ) from e

  logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
  return SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL,
  )
