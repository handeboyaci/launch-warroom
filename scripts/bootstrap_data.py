"""Bootstrap the Oncology War-Room database with target trial data."""

import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from warroom.constants import AACT_DB_PATH
from warroom.db.schema import init_db

TRIALS = [
  # --- KRAS G12C ---
  {
    "nct_id": "NCT03600883",
    "brief_title": "CodeBreaK 100: AMG 510 in KRAS G12C Advanced Solid Tumors",
    "overall_status": "Active, not recruiting",
    "phase": "Phase 1/Phase 2",
    "study_first_submitted_date": "2018-07-25",
    "intervention": "Sotorasib (AMG 510)",
    "sponsor": "Amgen",
  },
  {
    "nct_id": "NCT04303780",
    "brief_title": "CodeBreaK 200: Sotorasib vs Docetaxel in Pretreated NSCLC",
    "overall_status": "Active, not recruiting",
    "phase": "Phase 3",
    "study_first_submitted_date": "2020-03-25",
    "intervention": "Sotorasib",
    "sponsor": "Amgen",
  },
  {
    "nct_id": "NCT03785249",
    "brief_title": (
      "KRYSTAL-1: MRTX849 in Patients With Cancer Having a KRAS G12C Mutation"
    ),
    "overall_status": "Recruiting",
    "phase": "Phase 1/Phase 2",
    "study_first_submitted_date": "2018-12-26",
    "intervention": "Adagrasib (MRTX849)",
    "sponsor": "Mirati Therapeutics",
  },
  # --- EGFR ---
  {
    "nct_id": "NCT02296125",
    "brief_title": "FLAURA: Osimertinib vs EGFR-TKI in 1L EGFRm NSCLC",
    "overall_status": "Completed",
    "phase": "Phase 3",
    "study_first_submitted_date": "2014-11-17",
    "intervention": "Osimertinib",
    "sponsor": "AstraZeneca",
  },
  {
    "nct_id": "NCT02609776",
    "brief_title": "CHRYSALIS: JNJ-61186372 in Patients With Advanced NSCLC",
    "overall_status": "Recruiting",
    "phase": "Phase 1",
    "study_first_submitted_date": "2015-11-17",
    "intervention": "Amivantamab",
    "sponsor": "Janssen Research & Development, LLC",
  },
  # --- ALK ---
  {
    "nct_id": "NCT02075840",
    "brief_title": "ALEX: Alectinib vs Crizotinib in 1L ALK+ NSCLC",
    "overall_status": "Completed",
    "phase": "Phase 3",
    "study_first_submitted_date": "2014-02-27",
    "intervention": "Alectinib",
    "sponsor": "Hoffmann-La Roche",
  },
  {
    "nct_id": "NCT03052608",
    "brief_title": "CROWN: Lorlatinib vs Crizotinib in 1L ALK+ NSCLC",
    "overall_status": "Active, not recruiting",
    "phase": "Phase 3",
    "study_first_submitted_date": "2017-02-09",
    "intervention": "Lorlatinib",
    "sponsor": "Pfizer",
  },
  # --- HER2 ---
  {
    "nct_id": "NCT03248492",
    "brief_title": "DESTINY-Breast01: Trastuzumab Deruxtecan in HER2+ Breast Cancer",
    "overall_status": "Completed",
    "phase": "Phase 2",
    "study_first_submitted_date": "2017-08-10",
    "intervention": "Trastuzumab deruxtecan",
    "sponsor": "Daiichi Sankyo, Inc.",
  },
  {
    "nct_id": "NCT03529110",
    "brief_title": "DESTINY-Breast03: T-DXd vs T-DM1 in HER2+ Breast Cancer",
    "overall_status": "Active, not recruiting",
    "phase": "Phase 3",
    "study_first_submitted_date": "2018-05-15",
    "intervention": "Trastuzumab deruxtecan",
    "sponsor": "Daiichi Sankyo, Inc.",
  },
  # --- PD-L1 ---
  {
    "nct_id": "NCT02578680",
    "brief_title": "KEYNOTE-189: Pembro + Chemo vs Chemo in 1L NSCLC",
    "overall_status": "Completed",
    "phase": "Phase 3",
    "study_first_submitted_date": "2015-10-14",
    "intervention": "Pembrolizumab",
    "sponsor": "Merck Sharp & Dohme LLC",
  },
  {
    "nct_id": "NCT02504372",
    "brief_title": "KEYNOTE-091: Adjuvant Pembro vs Placebo in Resected NSCLC",
    "overall_status": "Active, not recruiting",
    "phase": "Phase 3",
    "study_first_submitted_date": "2015-07-16",
    "intervention": "Pembrolizumab",
    "sponsor": "Merck Sharp & Dohme LLC",
  },
]


def seed_db():
  print(f"Initializing database at {AACT_DB_PATH}...")
  conn = init_db(AACT_DB_PATH)

  print("Seeding trials...")
  for trial in TRIALS:
    # Insert into studies
    conn.execute(
      """
            INSERT INTO studies (
                nct_id, brief_title, overall_status, phase,
                study_first_submitted_date, source
            ) VALUES (?, ?, ?, ?, ?, ?)
        """,
      (
        trial["nct_id"],
        trial["brief_title"],
        trial["overall_status"],
        trial["phase"],
        trial["study_first_submitted_date"],
        trial["sponsor"],
      ),
    )

    # Insert into interventions
    conn.execute(
      """
            INSERT INTO interventions (nct_id, intervention_type, name)
            VALUES (?, ?, ?)
        """,
      (trial["nct_id"], "Drug", trial["intervention"]),
    )

    # Insert into sponsors
    conn.execute(
      """
            INSERT INTO sponsors (nct_id, agency_class, lead_or_collaborator, name)
            VALUES (?, ?, ?, ?)
        """,
      (trial["nct_id"], "Industry", "lead", trial["sponsor"]),
    )

  conn.commit()
  conn.close()
  print("Database seeded successfully!")


if __name__ == "__main__":
  seed_db()
