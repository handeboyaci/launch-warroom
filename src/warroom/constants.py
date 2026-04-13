"""Global constants for the Oncology War-Room system.

All tools, agents, and pipelines must import TIME_CUTOFF from here
and enforce it in every query, API call, and vector search.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

# ── Time-Travel Sandbox ─────────────────────────────────────────────
# The entire system operates as if the current date is this value.
# No data after this date may be accessed or generated.
TIME_CUTOFF = "2021-01-15"
TIME_CUTOFF_YYYYMMDD = "20210115"
TIME_CUTOFF_INT = 20210115
TIME_CUTOFF_DATE = date(2021, 1, 15)


def date_to_int(date_str: str) -> int:
  """Convert YYYY-MM-DD or partial date to YYYYMMDD integer for ChromaDB."""
  parsed = parse_date(date_str)
  if parsed:
    return int(parsed.strftime("%Y%m%d"))
  return 0


def parse_date(date_str: str) -> date | None:
  """Parse a date string in ISO-8601 format (YYYY-MM-DD).

  Returns None if the string is empty or unparseable.
  """
  if not date_str or not date_str.strip():
    return None
  try:
    return datetime.strptime(date_str.strip()[:10], "%Y-%m-%d").date()
  except (ValueError, TypeError):
    return None


def is_before_cutoff(date_str: str) -> bool:
  """Check if a date string is strictly before TIME_CUTOFF.

  Returns False for empty, unparseable, or post-cutoff dates.
  This is the canonical date check — use it everywhere instead
  of raw string comparison.
  """
  parsed = parse_date(date_str)
  if parsed is None:
    return False
  return parsed < TIME_CUTOFF_DATE


# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
AACT_DB_PATH = DATA_DIR / "aact.db"
CHROMA_DIR = PROCESSED_DATA_DIR / "chroma"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.json"
EVAL_DIR = DATA_DIR / "eval"

# ── Models ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
LLM_MODEL_AGENT = "gemini-flash-lite-latest"
LLM_MODEL_JUDGE = "gemini-flash-lite-latest"

# ── AACT Tables ──────────────────────────────────────────────────────
# Only these tables are downloaded and ingested.
AACT_TABLES = [
  "studies",
  "interventions",
  "eligibilities",
  "outcomes",
  "sponsors",
  "browse_conditions",
  "browse_interventions",
]

# ── Search Terms ─────────────────────────────────────────────────────
# Used by download scripts to filter AACT data.
KRAS_SEARCH_TERMS = [
  "KRAS",
  "KRAS G12C",
  "sotorasib",
  "adagrasib",
  "AMG 510",
  "MRTX849",
  "lumakras",
  "krazati",
]

TARGET_SEARCH_TERMS = {
  "KRAS": KRAS_SEARCH_TERMS,
  "EGFR": [
    "EGFR",
    "osimertinib",
    "tagrisso",
    "amivantamab",
    "rybrevant",
    "FLAURA",
    "CHRYSALIS",
  ],
  "ALK": [
    "ALK",
    "alectinib",
    "alecensa",
    "brigatinib",
    "alunbrig",
    "lorlatinib",
    "lorviqua",
    "ALEX",
    "ALTA-1L",
    "CROWN",
  ],
  "HER2": [
    "HER2",
    "Trastuzumab deruxtecan",
    "Enhertu",
    "DS-8201",
    "T-DXd",
    "DESTINY-Breast",
  ],
  "BRAF": ["BRAF", "encorafenib", "braftovi", "binimetinib", "mektovi", "COLUMBUS"],
  "PD-L1": [
    "PD-L1",
    "pembrolizumab",
    "keytruda",
    "nivolumab",
    "opdivo",
    "KEYNOTE-189",
    "KEYNOTE-091",
    "CheckMate 227",
  ],
  "ROS1": ["ROS1", "repotrectinib", "augtyro", "TRIDENT-1"],
}

# Known trial NCT IDs for the KRAS G12C race.
# All verified: study_first_submitted_date < 2021-01-15.
KRAS_NCT_IDS = [
  "NCT03600883",  # CodeBreaK 100 (Amgen) — submitted 2018-07-25
  "NCT04303780",  # CodeBreaK 200 (Amgen) — submitted 2020-03-25
  "NCT03785249",  # KRYSTAL-1 (Mirati)   — submitted 2018-12-26
  "NCT04330664",  # KRYSTAL-7 (Mirati)   — submitted 2020-04-14
  "NCT04613596",  # KRYSTAL-12 (Mirati)  — submitted 2020-11-04
]
