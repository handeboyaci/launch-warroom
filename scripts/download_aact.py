"""Download AACT Clinical Trials CSV exports.

Downloads pipe-delimited flat-file exports for the tables defined in
``constants.AACT_TABLES`` from the AACT static file server.

Usage:
  python scripts/download_aact.py [--output-dir data/raw]
"""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path

import requests
from rich.console import Console
from rich.progress import Progress

# Add src to path so we can import warroom.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from warroom.constants import AACT_TABLES, RAW_DATA_DIR  # noqa: E402

logger = logging.getLogger(__name__)
console = Console()

# AACT provides a static download of pipe-delimited files as a zip.
# This URL points to the latest monthly dump.
AACT_DOWNLOAD_URL = (
  "https://aact.ctti-clinicaltrials.org/static/exported_aact_files.zip"
)


def download_aact_zip(
  output_dir: Path,
  tables: list[str] | None = None,
  url: str = AACT_DOWNLOAD_URL,
) -> list[Path]:
  """Download and extract relevant AACT table files.

  Args:
    output_dir: Directory to save extracted CSV/TXT files.
    tables: List of table names to extract. Defaults to
      ``constants.AACT_TABLES``.
    url: URL to the AACT zip file.

  Returns:
    List of paths to the extracted files.
  """
  tables = tables or AACT_TABLES
  output_dir.mkdir(parents=True, exist_ok=True)

  console.print(f"[bold blue]Downloading AACT data from:[/] {url}")
  console.print("[dim]This may take a few minutes (file is ~2-3 GB)...[/]")

  # Stream to a temp file to avoid buffering the entire zip in memory.
  import tempfile

  tmp_zip = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
  try:
    with Progress() as progress:
      task = progress.add_task("[cyan]Downloading...", total=None)
      response = requests.get(url, stream=True, timeout=600)
      response.raise_for_status()

      for chunk in response.iter_content(chunk_size=8192):
        tmp_zip.write(chunk)
        progress.advance(task, len(chunk))

    tmp_zip.close()
    console.print("[green]Download complete. Extracting...[/]")

    extracted: list[Path] = []
    with zipfile.ZipFile(tmp_zip.name) as zf:
      for name in zf.namelist():
        # AACT files are named like "studies.txt",
        # "interventions.txt", etc.
        stem = Path(name).stem.lower()
        if stem in tables:
          target = output_dir / Path(name).name
          with (
            zf.open(name) as src,
            open(target, "wb") as dst,
          ):
            dst.write(src.read())
          extracted.append(target)
          console.print(f"  [green]✓[/] {target.name}")
        else:
          logger.debug("Skipping %s (not in target tables)", name)
  finally:
    Path(tmp_zip.name).unlink(missing_ok=True)

  console.print(f"\n[bold green]Extracted {len(extracted)} tables to {output_dir}[/]")
  return extracted


def main() -> None:  # pragma: no cover
  """CLI entry point."""
  parser = argparse.ArgumentParser(description="Download AACT clinical trials data.")
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=RAW_DATA_DIR,
    help="Directory to save downloaded files.",
  )
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO)
  download_aact_zip(args.output_dir)


if __name__ == "__main__":
  main()
