"""Download PubMed abstracts for KRAS G12C inhibitor literature.

Uses NCBI E-utilities (esearch + efetch) to retrieve abstracts
published before the TIME_CUTOFF.

Usage:
  python scripts/download_pubmed.py [--output-dir data/raw/pubmed]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from warroom.constants import (  # noqa: E402
  TARGET_SEARCH_TERMS,
  TIME_CUTOFF,
  is_before_cutoff,
)

logger = logging.getLogger(__name__)
console = Console()

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_OUTPUT_DIR = Path("data/raw/pubmed")

# Max date for PubMed search (YYYY/MM/DD format).
MAX_DATE = TIME_CUTOFF.replace("-", "/")


def esearch(
  query: str,
  max_results: int = 200,
  api_key: str | None = None,
) -> list[str]:
  """Search PubMed and return a list of PMIDs.

  Args:
    query: PubMed search query string.
    max_results: Maximum number of results to return.
    api_key: Optional NCBI API key for higher rate limits.

  Returns:
    List of PMID strings.
  """
  params: dict[str, str | int] = {
    "db": "pubmed",
    "term": query,
    "retmax": max_results,
    "retmode": "json",
    "datetype": "pdat",
    "maxdate": MAX_DATE,
    "sort": "relevance",
  }
  if api_key:
    params["api_key"] = api_key

  resp = requests.get(
    f"{EUTILS_BASE}/esearch.fcgi",
    params=params,
    timeout=30,
  )
  resp.raise_for_status()
  data = resp.json()

  pmids = data.get("esearchresult", {}).get("idlist", [])
  logger.info("esearch returned %d PMIDs for query: %s", len(pmids), query)
  return pmids


def efetch_abstracts(
  pmids: list[str],
  api_key: str | None = None,
) -> list[dict]:
  """Fetch article details from PubMed for given PMIDs.

  Args:
    pmids: List of PubMed IDs.
    api_key: Optional NCBI API key.

  Returns:
    List of dicts with keys: pmid, title, abstract, authors,
    journal, published_date.
  """
  if not pmids:
    return []

  articles: list[dict] = []

  # Fetch in batches of 50.
  batch_size = 50
  for i in range(0, len(pmids), batch_size):
    batch = pmids[i : i + batch_size]
    params: dict[str, str] = {
      "db": "pubmed",
      "id": ",".join(batch),
      "retmode": "xml",
      "rettype": "abstract",
    }
    if api_key:
      params["api_key"] = api_key

    resp = requests.get(
      f"{EUTILS_BASE}/efetch.fcgi",
      params=params,
      timeout=60,
    )
    resp.raise_for_status()

    articles.extend(_parse_pubmed_xml(resp.text))

    # Respect NCBI rate limits (3 requests/sec without key).
    time.sleep(0.4)

  return articles


def _parse_pubmed_xml(xml_text: str) -> list[dict]:
  """Parse PubMed XML response into structured records."""
  records: list[dict] = []
  root = ET.fromstring(xml_text)  # noqa: S314

  for article in root.findall(".//PubmedArticle"):
    pmid_el = article.find(".//PMID")
    title_el = article.find(".//ArticleTitle")
    journal_el = article.find(".//Journal/Title")

    # Handle structured abstracts (multiple AbstractText elements
    # with Label attributes like BACKGROUND, METHODS, RESULTS).
    abstract_parts = article.findall(".//Abstract/AbstractText")
    if abstract_parts:
      sections = []
      for part in abstract_parts:
        label = part.get("Label", "")
        text = part.text or ""
        if label and text:
          sections.append(f"{label}: {text}")
        elif text:
          sections.append(text)
      abstract = " ".join(sections)
    else:
      abstract = ""

    # Extract publication date.
    pub_date_el = article.find(".//PubDate")
    pub_date = _extract_pub_date(pub_date_el)

    # Extract authors.
    authors = []
    for author in article.findall(".//Author"):
      last = author.findtext("LastName", "")
      first = author.findtext("ForeName", "")
      if last:
        authors.append(f"{last} {first}".strip())

    record = {
      "pmid": pmid_el.text if pmid_el is not None else "",
      "title": title_el.text if title_el is not None else "",
      "abstract": abstract,
      "authors": authors,
      "journal": journal_el.text if journal_el is not None else "",
      "published_date": pub_date,
    }
    records.append(record)

  return records


def _extract_pub_date(pub_date_el: ET.Element | None) -> str:
  """Extract publication date as YYYY-MM-DD string."""
  if pub_date_el is None:
    return ""
  year = pub_date_el.findtext("Year", "")
  month = pub_date_el.findtext("Month", "01")
  day = pub_date_el.findtext("Day", "01")

  # Month might be abbreviated (e.g., "Jan").
  month_map = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
  }
  if month.lower() in month_map:
    month = month_map[month.lower()]
  month = month.zfill(2)
  day = day.zfill(2)

  return f"{year}-{month}-{day}" if year else ""


def download_pubmed_corpus(
  output_dir: Path | None = None,
  api_key: str | None = None,
) -> list[dict]:
  """Download PubMed abstracts for all KRAS search terms.

  Args:
    output_dir: Directory to save the output JSON.
    api_key: Optional NCBI API key.

  Returns:
    Deduplicated list of article records.
  """
  output_dir = output_dir or PUBMED_OUTPUT_DIR
  output_dir.mkdir(parents=True, exist_ok=True)

  all_pmids: set[str] = set()
  for target, terms in TARGET_SEARCH_TERMS.items():
    console.print(f"[bold magenta]Searching for target: {target}[/]")
    for term in terms:
      query = f"({term}) AND (cancer OR NSCLC OR lung)"
      pmids = esearch(
        query, max_results=20, api_key=api_key
      )  # Limit to 20 per term to keep it fast
      all_pmids.update(pmids)
      time.sleep(0.4)

  console.print(
    f"[bold blue]Found {len(all_pmids)} unique PMIDs "
    f"across {len(TARGET_SEARCH_TERMS)} targets[/]"
  )

  articles = efetch_abstracts(list(all_pmids), api_key=api_key)

  # Filter out articles without abstracts.
  articles = [a for a in articles if a.get("abstract")]

  # Double-check temporal filter using canonical date parser.
  filtered = []
  for a in articles:
    pub = a.get("published_date", "")
    if is_before_cutoff(pub):
      filtered.append(a)
    else:
      logger.warning(
        "Dropping PMID %s: published_date=%r (missing, unparseable, or after cutoff)",
        a.get("pmid"),
        pub,
      )

  console.print(
    f"[green]Retained {len(filtered)} articles (with abstract, before {TIME_CUTOFF})[/]"
  )

  # Save to JSON.
  output_path = output_dir / "pubmed_abstracts.json"
  with open(output_path, "w") as f:
    json.dump(filtered, f, indent=2)
  console.print(f"[green]Saved to {output_path}[/]")

  return filtered


def main() -> None:  # pragma: no cover
  """CLI entry point."""
  parser = argparse.ArgumentParser(
    description="Download PubMed abstracts for KRAS G12C research."
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=PUBMED_OUTPUT_DIR,
  )
  parser.add_argument("--api-key", type=str, default=None)
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO)
  download_pubmed_corpus(args.output_dir, args.api_key)


if __name__ == "__main__":
  main()
