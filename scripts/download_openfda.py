"""Download OpenFDA drug label data for relevant oncology drugs.

Queries the OpenFDA Drug Label API for labels effective before the
TIME_CUTOFF.

Usage:
  python scripts/download_openfda.py [--output-dir data/raw/openfda]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import requests
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from warroom.constants import (  # noqa: E402
  TIME_CUTOFF_YYYYMMDD,
)

logger = logging.getLogger(__name__)
console = Console()

OPENFDA_BASE = "https://api.fda.gov/drug/label.json"
OPENFDA_OUTPUT_DIR = Path("data/raw/openfda")

# Drugs relevant to the KRAS G12C competitive landscape and
# broader NSCLC treatment context (as of Jan 2021).
TARGET_DRUGS = [
  # Direct KRAS targets (may not have labels by Jan 2021).
  "sotorasib",
  "adagrasib",
  # NSCLC standard of care / comparators (for context).
  "docetaxel",
  "pembrolizumab",
  "nivolumab",
  "atezolizumab",
  "carboplatin",
  "pemetrexed",
  "erlotinib",
  "osimertinib",
]


def search_openfda_labels(
  drug_name: str,
  limit: int = 10,
) -> list[dict]:
  """Search OpenFDA for drug labels matching a drug name.

  Args:
    drug_name: Generic drug name to search for.
    limit: Maximum number of results.

  Returns:
    List of label records (raw OpenFDA JSON results).
  """
  # Enforce temporal cutoff in OpenFDA search syntax.
  # Use normal spaces — requests.get(params=...) handles
  # URL encoding automatically.
  search = (
    f'openfda.generic_name:"{drug_name}"'
    f" AND effective_time:"
    f"[19000101 TO {TIME_CUTOFF_YYYYMMDD}]"
  )
  params = {
    "search": search,
    "limit": limit,
  }

  resp = requests.get(OPENFDA_BASE, params=params, timeout=30)

  if resp.status_code == 404:
    # No results found — not an error for drugs without labels.
    logger.info("No OpenFDA labels found for '%s'", drug_name)
    return []

  resp.raise_for_status()
  data = resp.json()
  results = data.get("results", [])
  logger.info("Found %d labels for '%s'", len(results), drug_name)
  return results


def _extract_label_sections(raw_label: dict) -> dict:
  """Extract key sections from an OpenFDA label record.

  Returns:
    Dict with standardized keys for the label sections we need.
  """
  openfda = raw_label.get("openfda", {})
  return {
    "set_id": raw_label.get("set_id", ""),
    "effective_date": raw_label.get("effective_time", ""),
    "brand_name": (openfda.get("brand_name") or [""])[0],
    "generic_name": (openfda.get("generic_name") or [""])[0],
    "manufacturer": (openfda.get("manufacturer_name") or [""])[0],
    "indications_and_usage": (raw_label.get("indications_and_usage") or [""])[0],
    "dosage_and_administration": (raw_label.get("dosage_and_administration") or [""])[
      0
    ],
    "warnings_and_precautions": (raw_label.get("warnings_and_precautions") or [""])[0],
    "adverse_reactions": (raw_label.get("adverse_reactions") or [""])[0],
    "clinical_pharmacology": (raw_label.get("clinical_pharmacology") or [""])[0],
    "clinical_studies": (raw_label.get("clinical_studies") or [""])[0],
  }


def download_openfda_labels(
  output_dir: Path | None = None,
) -> list[dict]:
  """Download OpenFDA labels for all target drugs.

  Args:
    output_dir: Directory to save the output JSON.

  Returns:
    List of extracted label records.
  """
  output_dir = output_dir or OPENFDA_OUTPUT_DIR
  output_dir.mkdir(parents=True, exist_ok=True)

  all_labels: list[dict] = []
  for drug in TARGET_DRUGS:
    console.print(f"  [cyan]Searching:[/] {drug}")
    raw_labels = search_openfda_labels(drug)
    for raw in raw_labels:
      extracted = _extract_label_sections(raw)
      extracted["source_drug_query"] = drug
      all_labels.append(extracted)
    # Respect rate limits (40 req/min without key).
    time.sleep(1.5)

  # Deduplicate by set_id (keep latest effective date).
  seen: dict[str, dict] = {}
  for label in all_labels:
    sid = label["set_id"]
    if sid not in seen or (label["effective_date"] > seen[sid]["effective_date"]):
      seen[sid] = label
  deduped = list(seen.values())

  console.print(
    f"\n[bold green]Downloaded {len(deduped)} unique labels "
    f"for {len(TARGET_DRUGS)} drugs[/]"
  )

  output_path = output_dir / "openfda_labels.json"
  with open(output_path, "w") as f:
    json.dump(deduped, f, indent=2)
  console.print(f"[green]Saved to {output_path}[/]")

  return deduped


def main() -> None:  # pragma: no cover
  """CLI entry point."""
  parser = argparse.ArgumentParser(description="Download OpenFDA drug labels.")
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=OPENFDA_OUTPUT_DIR,
  )
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO)
  download_openfda_labels(args.output_dir)


if __name__ == "__main__":
  main()
