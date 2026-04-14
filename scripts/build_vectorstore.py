"""Build the complete RAG vector store from downloaded data.

Parses PubMed abstracts and OpenFDA labels, chunks them, and
ingests into ChromaDB.

Usage:
  python scripts/build_vectorstore.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from warroom.constants import CHROMA_DIR  # noqa: E402
from warroom.rag.chunker import (  # noqa: E402
  parse_openfda_labels,
  parse_pubmed_abstracts,
)
from warroom.rag.vectorstore import (  # noqa: E402
  add_documents,
  get_collection,
)

logger = logging.getLogger(__name__)
console = Console()

PUBMED_JSON = Path("data/raw/pubmed/pubmed_abstracts.json")
OPENFDA_JSON = Path("data/raw/openfda/openfda_labels.json")


def build_vectorstore(
  pubmed_path: Path = PUBMED_JSON,
  openfda_path: Path = OPENFDA_JSON,
  chroma_dir: Path | None = None,
) -> dict[str, int]:
  """Build the complete vector store.

  Args:
    pubmed_path: Path to PubMed abstracts JSON.
    openfda_path: Path to OpenFDA labels JSON.
    chroma_dir: ChromaDB persistence directory.

  Returns:
    Dict with counts: pubmed_docs, openfda_docs, total.
  """
  chroma_dir = chroma_dir or CHROMA_DIR
  collection = get_collection(persist_dir=chroma_dir)
  counts: dict[str, int] = {}

  # 1. Parse and ingest PubMed abstracts.
  if pubmed_path.exists():
    console.print("[bold blue]Parsing PubMed abstracts...[/]")
    pubmed_docs = parse_pubmed_abstracts(pubmed_path)
    counts["pubmed_docs"] = add_documents(pubmed_docs, collection)
    console.print(f"  [green]✓[/] {counts['pubmed_docs']} PubMed chunks")
  else:
    console.print(f"  [yellow]⚠[/] {pubmed_path} not found — skipping")
    counts["pubmed_docs"] = 0

  # 2. Parse and ingest OpenFDA labels.
  if openfda_path.exists():
    console.print("[bold blue]Parsing OpenFDA labels...[/]")
    openfda_docs = parse_openfda_labels(openfda_path)
    counts["openfda_docs"] = add_documents(openfda_docs, collection)
    console.print(f"  [green]✓[/] {counts['openfda_docs']} FDA label chunks")
  else:
    console.print(f"  [yellow]⚠[/] {openfda_path} not found — skipping")
    counts["openfda_docs"] = 0

  counts["total"] = counts["pubmed_docs"] + counts["openfda_docs"]
  console.print(
    f"\n[bold green]Vector store built: "
    f"{counts['total']} total chunks in {chroma_dir}[/]"
  )
  return counts


def main() -> None:  # pragma: no cover
  """CLI entry point."""
  logging.basicConfig(level=logging.INFO)
  build_vectorstore()


if __name__ == "__main__":
  main()
