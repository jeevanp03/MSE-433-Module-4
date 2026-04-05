"""
Prep Phase Data Generation
===========================
Generates AI-simulated prep-tracker data matching the format collected
by the EP Voice Tracker tool (same columns as AFib_Prep_Phase_Qualitative_Notes.xlsx).

The data simulates what the tool would capture across ~50 cases,
with realistic sub-phase timing distributions calibrated to the original
PT PREP/INTUBATION data (mean ~19 min, range 10-48 min).

Planted patterns (recoverable by analysis):
  - Dr. B has longer Draping times on average (+3 min)
  - Nurse Gomez is slower at Intubation (+2 min)
  - Equipment delays add 5-10 min and correlate with longer total prep
  - Cable delays cluster with Dr. C cases
  - Morning cases (first of day) have slightly longer Patient Entry -> Anesthesia
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from src.config import RANDOM_STATE, OUTPUT_DIR

PREP_DIR = OUTPUT_DIR / "prep_quant_analysis"

# Staff matching the teammate's existing 10 cases
DOCTORS = ["Dr A", "Dr B", "Dr C"]
NURSES = ["Nurse Smith", "Nurse Lee", "Nurse Patel", "Nurse Gomez"]

DELAY_TYPES = ["Equipment", "Cable", "Positioning", "Staff Wait", None]

# Sub-phase baseline durations (minutes): (mean, std)
# Calibrated so sequential sum ≈ real PT PREP/INTUBATION (mean ~19 min, range 10-48)
# Patient Entry -> Anesthesia (initial setup, monitoring)
ENTRY_TO_ANES = (5.0, 1.5)
# Anesthesia -> Intubation (induction + airway)
ANES_TO_INTUB = (4.5, 1.5)
# Intubation -> Draping (sterile prep + draping)
INTUB_TO_DRAPE = (5.5, 2.0)
# Draping -> Access (final checks + vascular access)
DRAPE_TO_ACCESS = (4.0, 1.5)

# Qualitative note templates by delay type
NOTE_TEMPLATES: Dict[str, List[str]] = {
    "Equipment": [
        "Team paused briefly to clarify equipment layout; communication remained calm and collaborative.",
        "Delay emerged due to missing cable; team discussed workaround while maintaining patient readiness.",
        "Equipment readiness required confirmation; team used time to review next procedural steps.",
        "Mapping system not powered on at start; brief delay while tech resolved.",
        "Backup catheter tray missing from room; retrieved from central supply.",
    ],
    "Cable": [
        "Smooth preparation overall, with minor verbal coordination to align anesthesia and nursing tasks.",
        "Mapping cable tangled with IV lines during positioning; untangled without incident.",
        "Cable routing required adjustment after initial draping; minor delay.",
        "Brief confusion about patient positioning led to additional discussion before proceeding.",
    ],
    "Positioning": [
        "Prep flow improved after early clarification of responsibilities among team members.",
        "Patient needed extra padding due to body habitus; additional positioning time.",
        "Repositioning required after initial placement was suboptimal for access.",
        "Staff double-checked access setup, prioritizing safety over speed during prep phase.",
    ],
    "Staff Wait": [
        "Circulating nurse pulled to assist adjacent room briefly; minor pause.",
        "Waiting for second nurse to arrive for patient transfer.",
        "Anaesthesia team delayed finishing intubation in adjacent room.",
        "Handoff delay: outgoing nurse had not completed sign-out.",
    ],
    None: [
        "Slight hesitation during bedside setup as roles were verbally reassigned in real time.",
        "Prep felt unstructured at first; nursing staff adapted workflow after initial uncertainty.",
        "Procedure setup highlighted reliance on verbal cues rather than standardized prep sequence.",
        "Smooth case with no notable delays; team worked efficiently through all prep steps.",
        "Standard prep workflow; minor verbal coordination between nursing and anesthesia.",
        "Prep proceeded without interruption; team familiar with procedure setup.",
    ],
}


def generate_prep_data(n_cases: int = 50, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate synthetic prep-tracker cases.

    Returns DataFrame with columns matching AFib_Prep_Phase_Qualitative_Notes.xlsx:
    Case ID, Doctor, Nurse, Patient Entry, Anesthesia, Intubation, Draping,
    Access, Delay Type, Delay Duration (min), Qualitative Note
    """
    rng = np.random.default_rng(seed)
    rows = []

    # Start case IDs from 201 (continuing from existing 10 cases)
    start_id = 201
    base_date = pd.Timestamp("2026-03-21")

    for i in range(n_cases):
        case_id = start_id + i
        doctor = rng.choice(DOCTORS, p=[0.45, 0.30, 0.25])  # match original distribution
        nurse = rng.choice(NURSES)
        case_date = base_date + pd.Timedelta(days=i % 30)

        # Base time: 08:00 for first case, stagger for subsequent cases same day
        hour = 8 + (i % 4) * 2  # 08, 10, 12, 14
        entry_time = case_date.replace(hour=hour, minute=0, second=0)

        # Generate sub-phase durations
        d_entry_anes = max(2.0, rng.normal(*ENTRY_TO_ANES))
        d_anes_intub = max(2.0, rng.normal(*ANES_TO_INTUB))
        d_intub_drape = max(2.0, rng.normal(*INTUB_TO_DRAPE))
        d_drape_access = max(2.0, rng.normal(*DRAPE_TO_ACCESS))

        # Planted patterns
        delay_type = None
        delay_dur = 0

        # Pattern 1: Dr B has longer draping times
        if doctor == "Dr B":
            d_intub_drape += rng.uniform(1.0, 3.0)

        # Pattern 2: Nurse Gomez slower at intubation
        if nurse == "Nurse Gomez":
            d_anes_intub += rng.uniform(1.0, 2.5)

        # Pattern 3: Equipment delays (~15% of cases)
        if rng.random() < 0.15:
            delay_type = "Equipment"
            delay_dur = rng.uniform(3.0, 7.0)
            # Equipment delays mostly hit draping or access phases
            if rng.random() < 0.6:
                d_intub_drape += delay_dur * 0.6
                d_drape_access += delay_dur * 0.4
            else:
                d_drape_access += delay_dur

        # Pattern 4: Cable delays cluster with Dr C (~20% of Dr C cases)
        elif doctor == "Dr C" and rng.random() < 0.20:
            delay_type = "Cable"
            delay_dur = rng.uniform(1.0, 4.0)
            d_intub_drape += delay_dur

        # Pattern 5: Positioning delays (~8% of cases)
        elif rng.random() < 0.08:
            delay_type = "Positioning"
            delay_dur = rng.uniform(2.0, 5.0)
            d_entry_anes += delay_dur * 0.5
            d_anes_intub += delay_dur * 0.5

        # Pattern 6: Staff wait delays (~6% of cases)
        elif rng.random() < 0.06:
            delay_type = "Staff Wait"
            delay_dur = rng.uniform(2.0, 5.0)
            d_entry_anes += delay_dur

        # Pattern 7: First case of day slightly longer entry
        if hour == 8:
            d_entry_anes += rng.uniform(0.5, 2.0)

        # Compute timestamps
        anes_time = entry_time + pd.Timedelta(minutes=round(d_entry_anes))
        intub_time = anes_time + pd.Timedelta(minutes=round(d_anes_intub))
        drape_time = intub_time + pd.Timedelta(minutes=round(d_intub_drape))
        access_time = drape_time + pd.Timedelta(minutes=round(d_drape_access))

        # Qualitative note
        note = rng.choice(NOTE_TEMPLATES[delay_type])

        rows.append({
            "Case ID": case_id,
            "Doctor": doctor,
            "Nurse": nurse,
            "Patient Entry": entry_time,
            "Anesthesia": anes_time,
            "Intubation": intub_time,
            "Draping": drape_time,
            "Access": access_time,
            "Delay Type": delay_type if delay_type else np.nan,
            "Delay Duration (min)": round(delay_dur),
            "Qualitative Note": note,
        })

    return pd.DataFrame(rows)


def run_prep_data_gen(n_cases: int = 50) -> pd.DataFrame:
    """Generate and save prep tracker data."""
    PREP_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("PREP PHASE: Generating Synthetic Tracker Data")
    print("=" * 60)

    df = generate_prep_data(n_cases=n_cases)

    # Save to CSV and Excel
    df.to_csv(PREP_DIR / "prep_tracker_data.csv", index=False)
    df.to_excel(PREP_DIR / "prep_tracker_data.xlsx", index=False)

    total_prep = (df["Access"] - df["Patient Entry"]).dt.total_seconds() / 60
    print(f"  Generated {n_cases} cases")
    print(f"  Total prep time: mean={total_prep.mean():.1f} min, "
          f"std={total_prep.std():.1f} min, range=[{total_prep.min():.0f}, {total_prep.max():.0f}]")
    print(f"  Delay cases: {df['Delay Type'].notna().sum()} / {n_cases}")
    print(f"  Saved: prep_tracker_data.csv, prep_tracker_data.xlsx")

    return df
