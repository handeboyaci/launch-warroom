"""Temporal leak detection for LLM outputs.

Scans agent-generated text for any references to events, data, or
knowledge that postdate the TIME_CUTOFF (January 15, 2021). This is
the primary runtime guardrail for temporal integrity.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from warroom.constants import TIME_CUTOFF, parse_date

logger = logging.getLogger(__name__)

# ── Known Post-Cutoff Events ────────────────────────────────────────
# Each entry: (regex pattern, description of the leak).
# These are events that definitively occurred after January 15, 2021.
POST_CUTOFF_EVENTS: list[tuple[str, str]] = [
  # Sotorasib / Lumakras.
  (
    r"sotorasib\s+(?:was\s+)?(?:approved|granted|received)",
    "Sotorasib FDA approval (May 2021)",
  ),
  (
    r"lumakras\s+(?:was\s+)?(?:approved|launched|granted)",
    "Lumakras FDA approval (May 2021)",
  ),
  (
    r"lumakras",
    "Lumakras brand name (not public before May 2021)",
  ),
  (
    r"CodeBreaK\s*200\s+(?:results?|data|showed|demonstrated)",
    "CodeBreaK 200 results (reported 2022)",
  ),
  (
    r"sotorasib.*phase\s*(?:III|3)\s+(?:results?|data)",
    "Sotorasib Phase III data (2022)",
  ),
  # Adagrasib / Krazati.
  (
    r"adagrasib\s+(?:was\s+)?(?:approved|granted|received)",
    "Adagrasib FDA approval (December 2022)",
  ),
  (
    r"krazati",
    "Krazati brand name (December 2022)",
  ),
  (
    r"KRYSTAL[- ]*1\s+(?:results?|data|showed|demonstrated)",
    "KRYSTAL-1 results (reported June 2022)",
  ),
  # General post-cutoff knowledge — scoped to KRAS drugs
  # to avoid false positives on pre-cutoff approvals (e.g.,
  # "pembrolizumab received approval in 2014").
  (
    r"(?:sotorasib|adagrasib|AMG\s*510|MRTX\s*849)"
    r".*(?:was|were|has been|had been)\s+approved",
    "Post-cutoff KRAS drug approval language",
  ),
  (
    r"(?:sotorasib|adagrasib|AMG\s*510|MRTX\s*849)"
    r".*(?:received|granted)\s+(?:accelerated\s+)?approval",
    "Post-cutoff KRAS drug approval language",
  ),
  (
    r"COVID[\s-]*19\s+(?:vaccine|pandemic\s+impact|variant)",
    "COVID-19 context (major events post Jan 2021)",
  ),
  # Literature Leaks.
  (
    r"\bPMID\s*[:\s]*(\d{8,})\b",
    "Potential post-cutoff publication (High PMID)",
  ),
]

# ── Future-Tense Knowledge Patterns ─────────────────────────────────
# Phrases that indicate the LLM is "remembering" future events.
FUTURE_KNOWLEDGE_PHRASES: list[tuple[str, str]] = [
  (
    r"(?:went on to|would later|subsequently|eventually)\s+"
    r"(?:show|demonstrate|receive|gain|achieve)",
    "Future knowledge phrasing",
  ),
  (
    r"(?:ultimately|in the end|as we now know)",
    "Retrospective knowledge phrasing",
  ),
  (
    r"(?:has since|have since)\s+(?:been|shown|demonstrated)",
    "Post-cutoff retrospective phrasing",
  ),
]


@dataclass
class TemporalLeakResult:
  """Result of a temporal leak scan."""

  clean: bool
  leaks: list[str] = field(default_factory=list)

  def __str__(self) -> str:
    if self.clean:
      return "CLEAN: No temporal leaks detected."
    return f"LEAK DETECTED ({len(self.leaks)} violation(s)):\n" + "\n".join(
      f"  • {leak}" for leak in self.leaks
    )


def scan_for_temporal_leaks(text: Any) -> TemporalLeakResult:
  """Scan text for post-cutoff temporal leaks.

  Handles both string and list (content blocks) inputs.
  """
  if not text:
    return TemporalLeakResult(clean=True)

  # Flatten list of content blocks if necessary
  if isinstance(text, list):
    parts = []
    for item in text:
      if isinstance(item, dict) and "text" in item:
        parts.append(str(item["text"]))
      else:
        parts.append(str(item))
    text = "\n".join(parts)

  if not isinstance(text, str):
    text = str(text)

  leaks: list[str] = []

  # 1. Check known post-cutoff events.
  for pattern, description in POST_CUTOFF_EVENTS:
    if re.search(pattern, text, re.IGNORECASE):
      leaks.append(f"[KNOWN EVENT] {description}")

  # 2. Check for explicit post-cutoff dates.
  date_leaks = _scan_dates(text)
  leaks.extend(date_leaks)

  # 3. Check future knowledge phrasing.
  for pattern, description in FUTURE_KNOWLEDGE_PHRASES:
    if re.search(pattern, text, re.IGNORECASE):
      leaks.append(f"[FUTURE PHRASING] {description}")

  if leaks:
    logger.warning(
      "Temporal leak detected (%d violations): %s",
      len(leaks),
      leaks,
    )

  return TemporalLeakResult(
    clean=len(leaks) == 0,
    leaks=leaks,
  )


def _scan_dates(text: str) -> list[str]:
  """Extract dates from text and flag any >= TIME_CUTOFF.

  Handles formats:
  - YYYY-MM-DD
  - Month YYYY (e.g., "May 2021")
  - MM/YYYY
  """
  leaks: list[str] = []

  # ISO dates: YYYY-MM-DD.
  for match in re.finditer(r"\b(\d{4}-\d{2}-\d{2})\b", text):
    date_str = match.group(1)
    parsed = parse_date(date_str)
    cutoff = parse_date(TIME_CUTOFF)
    if parsed and cutoff and parsed > cutoff:
      leaks.append(f"[DATE] Post-cutoff date found: {date_str}")

  # Month YYYY: "May 2021", "December 2022".
  month_map = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
  }
  pattern = r"\b(" + "|".join(month_map.keys()) + r")\s+(\d{4})\b"
  for match in re.finditer(pattern, text, re.IGNORECASE):
    month = month_map[match.group(1).lower()]
    year = match.group(2)
    # Defaulting to 01 of the month allows referring to the current month (Jan 2021)
    # while still flagging any future months (Feb 2021+).
    date_str = f"{year}-{month}-01"
    parsed = parse_date(date_str)
    cutoff = parse_date(TIME_CUTOFF)
    if parsed and cutoff and parsed > cutoff:
      leaks.append(f"[DATE] Post-cutoff month/year found: {match.group(1)} {year}")

  # Standalone Years: Catch any mention of 2021 or later.
  # Negative lookarounds prevent triggering on valid ISO dates like 2021-01-14.
  for match in re.finditer(r"(?<![-/])\b(202[1-9]|20[3-9]\d)\b(?![-/])", text):
    year = match.group(1)
    if int(year) > int(TIME_CUTOFF[:4]):
      leaks.append(f"[DATE] Post-cutoff year found: {year}")

  # PMID Check: Flag any PMID > 33450000 (roughly Jan 15, 2021).
  for match in re.finditer(r"\bPMID\s*[:\s]*(\d{8,})\b", text, re.IGNORECASE):
    pmid_val = int(match.group(1))
    if pmid_val > 33450000:
      leaks.append(f"[LITERATURE] Post-cutoff PMID detected: {pmid_val}")

  return leaks
