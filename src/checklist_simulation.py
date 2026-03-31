"""
Checklist Validation — Phase 1: Synthetic Data Generation
==========================================================
Generates synthetic prep-tracker sessions with **known ground-truth patterns**
so the downstream analysis can be validated against an answer key.

Ground-truth patterns planted:
  • Nurse RN-02 is consistently slow at 'airway' (+3-5 min)
  • Nurse RN-04 is consistently slow at 'sterile_draping' (+2-4 min)
  • 'equipment_not_ready' adds 4-8 min to 'equipment_readiness'
  • 'difficult_airway' adds 5-9 min to 'airway'
  • 'waiting_staff' adds 3-6 min to 'monitoring_setup'
  • Monday cases have 15% equipment delay rate (vs 5% other days)
  • Dr. B cases have slightly longer baseline prep (~2 min extra)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR, RANDOM_STATE

SIM_DIR = OUTPUT_DIR / "checklist_validation"

# ── Sub-phase catalogue (matches frontend PrepTracker) ────────────────────────
PHASE_KEYS = [
    "patient_transfer",
    "monitoring_setup",
    "anesthesia_induction",
    "airway",
    "sterile_prep",
    "sterile_draping",
    "equipment_readiness",
    "final_ready",
]

PHASE_LABELS = {
    "patient_transfer": "Patient transfer & positioning",
    "monitoring_setup": "Monitoring hookup / baseline setup",
    "anesthesia_induction": "Anaesthesia induction",
    "airway": "Airway / intubation",
    "sterile_prep": "Sterile prep",
    "sterile_draping": "Sterile draping",
    "equipment_readiness": "Equipment / room readiness",
    "final_ready": "Final ready-for-access check",
}

# ── Staff roster ──────────────────────────────────────────────────────────────
NURSES = ["RN-01", "RN-02", "RN-03", "RN-04"]
ANAESTHETISTS = ["AN-11", "AN-12"]
PHYSICIANS = ["Dr. A", "Dr. B", "Dr. C"]

# ── Reason codes (matches frontend) ──────────────────────────────────────────
REASON_CODES = [
    "difficult_airway", "patient_reposition", "monitoring_setup",
    "equipment_not_ready", "supply_retrieval", "cable_interference",
    "waiting_staff", "communication_delay", "room_setup_issue",
    "sterility_issue", "patient_complexity", "safety_pause", "other",
]

# ── Baseline durations per phase (minutes): mean, std ─────────────────────────
# Calibrated so individual phase durations are realistic; many phases run in
# parallel, and total wall-clock prep time matches real data (~19 min mean).
BASELINE_DURATIONS: Dict[str, Tuple[float, float]] = {
    "patient_transfer":      (3.5, 1.2),
    "monitoring_setup":      (6.5, 2.0),
    "anesthesia_induction":  (7.0, 2.5),
    "airway":                (6.0, 2.5),
    "sterile_prep":          (5.0, 1.5),
    "sterile_draping":       (5.5, 1.8),
    "equipment_readiness":   (8.0, 3.0),
    "final_ready":           (3.0, 1.0),
}

# ── Phase start offsets (minutes from session start): mean, std ───────────────
# Models parallel execution — phases overlap as they do in real clinical flow.
# Derived from the seed data in prepTrackerSeed.ts and real PT PREP/INTUBATION
# distribution (mean ~19 min, range 10-48 min).
PHASE_START_OFFSETS: Dict[str, Tuple[float, float]] = {
    "patient_transfer":      (0.0, 0.0),   # always first
    "monitoring_setup":      (1.0, 0.8),    # starts during transfer
    "anesthesia_induction":  (4.5, 1.5),    # after monitoring baseline
    "airway":                (6.5, 2.0),    # overlaps with late induction
    "sterile_prep":          (5.0, 1.5),    # parallel with anaesthesia
    "sterile_draping":       (7.5, 1.5),    # after sterile prep starts
    "equipment_readiness":   (1.5, 1.0),    # background task, starts early
    "final_ready":           (0.0, 0.0),    # computed dynamically: starts after all others end
}

# ═══════════════════════════════════════════════════════════════════════════════
# Ground-truth pattern definitions  (the "answer key")
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlantedPattern:
    """One testable pattern embedded in the synthetic data."""
    name: str
    description: str
    category: str          # 'nurse_effect', 'delay_cause', 'contextual'
    phase: Optional[str]   # which sub-phase is affected (None = global)
    expected_extra_min: Tuple[float, float]  # (low, high) of extra minutes

PLANTED_PATTERNS: List[PlantedPattern] = [
    PlantedPattern(
        name="RN-02 slow at airway",
        description="Nurse RN-02 adds 3-5 min to airway phase when assigned",
        category="nurse_effect",
        phase="airway",
        expected_extra_min=(3.0, 5.0),
    ),
    PlantedPattern(
        name="RN-04 slow at sterile draping",
        description="Nurse RN-04 adds 2-4 min to sterile draping when assigned",
        category="nurse_effect",
        phase="sterile_draping",
        expected_extra_min=(2.0, 4.0),
    ),
    PlantedPattern(
        name="Equipment not ready delay",
        description="'equipment_not_ready' reason code adds 4-8 min to equipment readiness",
        category="delay_cause",
        phase="equipment_readiness",
        expected_extra_min=(4.0, 8.0),
    ),
    PlantedPattern(
        name="Difficult airway delay",
        description="'difficult_airway' reason code adds 5-9 min to airway phase",
        category="delay_cause",
        phase="airway",
        expected_extra_min=(5.0, 9.0),
    ),
    PlantedPattern(
        name="Waiting on staff delay",
        description="'waiting_staff' reason code adds 3-6 min to monitoring setup",
        category="delay_cause",
        phase="monitoring_setup",
        expected_extra_min=(3.0, 6.0),
    ),
    PlantedPattern(
        name="Monday equipment delay",
        description="Monday cases have 15% equipment delay rate vs 5% on other days",
        category="contextual",
        phase="equipment_readiness",
        expected_extra_min=(4.0, 8.0),
    ),
    PlantedPattern(
        name="Dr. B baseline overhead",
        description="Dr. B cases have ~2 min extra across all phases",
        category="contextual",
        phase=None,
        expected_extra_min=(1.5, 2.5),
    ),
]

# ── Free-text note templates keyed by reason code ─────────────────────────────
NOTE_TEMPLATES: Dict[str, List[str]] = {
    "difficult_airway": [
        "Difficult intubation — required video laryngoscope on second attempt.",
        "Unexpected Cormack-Lehane grade 3; backup airway plan activated.",
        "Patient anatomy made intubation challenging; took two attempts.",
        "Airway management complicated by limited neck extension.",
    ],
    "equipment_not_ready": [
        "Mapping system not powered on; delayed start by several minutes.",
        "Backup catheter tray missing from room; had to retrieve from storage.",
        "Ablation generator failed self-test; swapped to backup unit.",
        "Sterile supplies not restocked from prior case.",
    ],
    "waiting_staff": [
        "Circulating nurse pulled to assist emergent case next door.",
        "Anaesthesia team delayed — finishing intubation in adjacent room.",
        "Handoff delay: outgoing nurse had not completed sign-out.",
        "Waiting for second nurse to arrive for patient transfer.",
    ],
    "patient_reposition": [
        "Patient needed extra padding due to body habitus.",
        "Repositioning required after initial placement was suboptimal.",
        "Patient mobility limited; required slide board and extra staff.",
    ],
    "monitoring_setup": [
        "ECG leads had poor contact; replaced pads and repositioned.",
        "Pulse oximeter reading unreliable; switched to forehead sensor.",
    ],
    "supply_retrieval": [
        "Extra drape kit retrieved from central supply.",
        "Needed additional contrast; runner dispatched to pharmacy.",
    ],
    "sterility_issue": [
        "Sterile field compromised — re-draped and re-prepped.",
        "Outer drape packaging was torn; replaced with new set.",
    ],
    "patient_complexity": [
        "Patient on anticoagulation; extra time for line placement planning.",
        "Morbid obesity required wider positioning equipment.",
    ],
    "communication_delay": [
        "Physician wanted to review imaging before proceeding; brief hold.",
        "Timeout repeated due to unclear laterality confirmation.",
    ],
    "cable_interference": [
        "Mapping cable tangled with IV lines during positioning.",
    ],
    "room_setup_issue": [
        "C-arm repositioned to accommodate left-side access.",
    ],
    "safety_pause": [
        "Extended surgical time-out for complex case review.",
    ],
    "other": [
        "Minor delay — no specific cause identified.",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Core generation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhaseRecord:
    """Ground-truth record for one phase within a session."""
    phase_key: str
    true_duration_min: float
    start_offset_min: float  # when this phase starts (minutes from session start)
    nurse_id: str
    anaesthetist_id: str
    physician_id: str
    is_extraordinary: bool
    reason_codes: List[str]
    true_note: str
    delay_min: float  # how much extra time was added by patterns


@dataclass
class SessionRecord:
    """Ground-truth record for one complete prep session."""
    session_id: int
    date: str
    day_of_week: str
    physician_id: str
    total_prep_min: float
    is_outlier: bool
    phases: List[PhaseRecord]
    active_patterns: List[str]  # names of patterns that fired


@dataclass
class TrackerObservation:
    """What the tracker would actually capture (with noise / missingness)."""
    session_id: int
    phase_key: str
    recorded_duration_min: Optional[float]  # None if missed
    nurse_id: str
    is_extraordinary: bool
    reason_codes: List[str]
    free_text_note: str
    is_missing: bool


def generate_ground_truth(
    n_sessions: int = 150,
    seed: int = RANDOM_STATE,
) -> Tuple[List[SessionRecord], Dict]:
    """Generate synthetic sessions with planted patterns.

    Returns:
        sessions: list of SessionRecord with full ground truth
        metadata: dict with pattern definitions and generation params
    """
    rng = np.random.default_rng(seed)

    # Date range: simulate ~6 weeks of data, 3-5 cases/day on weekdays
    dates = pd.bdate_range("2025-09-01", periods=40, freq="B")  # 40 business days
    # Assign sessions to dates
    session_dates = []
    for d in dates:
        n_cases = rng.integers(3, 6)  # 3-5 cases per day
        session_dates.extend([d] * n_cases)
    session_dates = session_dates[:n_sessions]
    # If we need more, cycle
    while len(session_dates) < n_sessions:
        session_dates.append(rng.choice(dates))

    sessions: List[SessionRecord] = []

    for i in range(n_sessions):
        date = session_dates[i]
        day_name = date.day_name()
        physician = rng.choice(PHYSICIANS)
        nurse = rng.choice(NURSES)
        anaesthetist = rng.choice(ANAESTHETISTS)

        active_patterns: List[str] = []
        phases: List[PhaseRecord] = []

        for phase_key in PHASE_KEYS:
            base_mean, base_std = BASELINE_DURATIONS[phase_key]
            duration = max(1.0, rng.normal(base_mean, base_std))
            delay = 0.0
            is_extraordinary = False
            reason_codes: List[str] = []
            note = ""

            # Phase start offset (parallel execution model)
            offset_mean, offset_std = PHASE_START_OFFSETS[phase_key]
            if phase_key == "final_ready":
                start_offset = 0.0  # placeholder — computed after all phases
            else:
                start_offset = max(0.0, rng.normal(offset_mean, offset_std))

            # Pattern: Dr. B baseline overhead
            if physician == "Dr. B":
                overhead = rng.uniform(1.5, 2.5) / len(PHASE_KEYS)  # spread across phases
                duration += overhead
                if phase_key == PHASE_KEYS[0]:  # count once
                    active_patterns.append("Dr. B baseline overhead")

            # Pattern: RN-02 slow at airway
            if nurse == "RN-02" and phase_key == "airway":
                extra = rng.uniform(3.0, 5.0)
                duration += extra
                delay += extra
                active_patterns.append("RN-02 slow at airway")

            # Pattern: RN-04 slow at sterile draping
            if nurse == "RN-04" and phase_key == "sterile_draping":
                extra = rng.uniform(2.0, 4.0)
                duration += extra
                delay += extra
                active_patterns.append("RN-04 slow at sterile draping")

            # Pattern: Monday equipment delays (15% vs 5%)
            equip_delay_prob = 0.15 if day_name == "Monday" else 0.05
            if phase_key == "equipment_readiness" and rng.random() < equip_delay_prob:
                extra = rng.uniform(4.0, 8.0)
                duration += extra
                delay += extra
                is_extraordinary = True
                reason_codes.append("equipment_not_ready")
                note = rng.choice(NOTE_TEMPLATES["equipment_not_ready"])
                active_patterns.append("Monday equipment delay" if day_name == "Monday"
                                       else "Equipment not ready delay")

            # Pattern: difficult airway (~8% of cases)
            if phase_key == "airway" and rng.random() < 0.08:
                extra = rng.uniform(5.0, 9.0)
                duration += extra
                delay += extra
                is_extraordinary = True
                reason_codes.append("difficult_airway")
                note = rng.choice(NOTE_TEMPLATES["difficult_airway"])
                active_patterns.append("Difficult airway delay")

            # Pattern: waiting on staff (~7% of monitoring_setup)
            if phase_key == "monitoring_setup" and rng.random() < 0.07:
                extra = rng.uniform(3.0, 6.0)
                duration += extra
                delay += extra
                is_extraordinary = True
                reason_codes.append("waiting_staff")
                note = rng.choice(NOTE_TEMPLATES["waiting_staff"])
                active_patterns.append("Waiting on staff delay")

            # Random minor issues (~5% of any phase)
            if not is_extraordinary and rng.random() < 0.05:
                minor_code = rng.choice([
                    "patient_reposition", "communication_delay",
                    "cable_interference", "safety_pause", "other",
                ])
                extra = rng.uniform(1.0, 3.0)
                duration += extra
                delay += extra
                is_extraordinary = True
                reason_codes.append(minor_code)
                if minor_code in NOTE_TEMPLATES:
                    note = rng.choice(NOTE_TEMPLATES[minor_code])
                else:
                    note = "Minor delay noted."

            duration = round(duration, 1)
            delay = round(delay, 1)

            phases.append(PhaseRecord(
                phase_key=phase_key,
                true_duration_min=duration,
                start_offset_min=round(start_offset, 1),
                nurse_id=nurse,
                anaesthetist_id=anaesthetist,
                physician_id=physician,
                is_extraordinary=is_extraordinary,
                reason_codes=reason_codes,
                true_note=note,
                delay_min=delay,
            ))

        # Compute final_ready start: after all other phases end
        max_end = max(
            p.start_offset_min + p.true_duration_min
            for p in phases if p.phase_key != "final_ready"
        )
        for p in phases:
            if p.phase_key == "final_ready":
                p.start_offset_min = round(max_end, 1)

        # Total prep = wall-clock span (max end time across all phases)
        total_prep = round(max(
            p.start_offset_min + p.true_duration_min for p in phases
        ), 1)
        # Mark outlier at 90th percentile (will recalculate after generation)
        sessions.append(SessionRecord(
            session_id=i + 1,
            date=date.strftime("%Y-%m-%d"),
            day_of_week=day_name,
            physician_id=physician,
            total_prep_min=total_prep,
            is_outlier=False,  # set below
            phases=phases,
            active_patterns=list(set(active_patterns)),
        ))

    # Set outlier labels at 90th percentile
    totals = [s.total_prep_min for s in sessions]
    p90 = float(np.percentile(totals, 90))
    for s in sessions:
        s.is_outlier = s.total_prep_min >= p90

    metadata = {
        "n_sessions": n_sessions,
        "seed": seed,
        "p90_threshold_min": round(p90, 1),
        "n_outliers": sum(1 for s in sessions if s.is_outlier),
        "planted_patterns": [asdict(p) for p in PLANTED_PATTERNS],
        "phase_keys": PHASE_KEYS,
        "nurses": NURSES,
        "physicians": PHYSICIANS,
    }

    return sessions, metadata


def simulate_tracker_data(
    sessions: List[SessionRecord],
    missing_rate: float = 0.08,
    timing_noise_std: float = 0.5,
    note_drop_rate: float = 0.10,
    seed: int = RANDOM_STATE,
) -> List[TrackerObservation]:
    """Simulate what the tracker would actually capture from ground truth.

    Args:
        sessions: ground-truth sessions
        missing_rate: probability a phase record is completely missing
        timing_noise_std: std dev (minutes) of timing measurement noise
        note_drop_rate: probability that a free-text note is left blank
        seed: random seed
    """
    rng = np.random.default_rng(seed + 1)  # different seed from generation
    observations: List[TrackerObservation] = []

    for session in sessions:
        for phase in session.phases:
            is_missing = rng.random() < missing_rate

            if is_missing:
                observations.append(TrackerObservation(
                    session_id=session.session_id,
                    phase_key=phase.phase_key,
                    recorded_duration_min=None,
                    nurse_id=phase.nurse_id,
                    is_extraordinary=False,
                    reason_codes=[],
                    free_text_note="",
                    is_missing=True,
                ))
                continue

            # Add timing noise
            noise = rng.normal(0, timing_noise_std)
            recorded = max(0.5, round(phase.true_duration_min + noise, 1))

            # Occasionally drop the note even if extraordinary
            note = phase.true_note
            if note and rng.random() < note_drop_rate:
                note = ""

            observations.append(TrackerObservation(
                session_id=session.session_id,
                phase_key=phase.phase_key,
                recorded_duration_min=recorded,
                nurse_id=phase.nurse_id,
                is_extraordinary=phase.is_extraordinary,
                reason_codes=phase.reason_codes,
                free_text_note=note,
                is_missing=False,
            ))

    return observations


def sessions_to_dataframe(sessions: List[SessionRecord]) -> pd.DataFrame:
    """Flatten ground-truth sessions into a tidy DataFrame."""
    rows = []
    for s in sessions:
        for p in s.phases:
            rows.append({
                "session_id": s.session_id,
                "date": s.date,
                "day_of_week": s.day_of_week,
                "physician": s.physician_id,
                "total_prep_min": s.total_prep_min,
                "is_outlier": s.is_outlier,
                "phase": p.phase_key,
                "phase_label": PHASE_LABELS[p.phase_key],
                "true_duration_min": p.true_duration_min,
                "start_offset_min": p.start_offset_min,
                "nurse": p.nurse_id,
                "anaesthetist": p.anaesthetist_id,
                "is_extraordinary": p.is_extraordinary,
                "reason_codes": "|".join(p.reason_codes) if p.reason_codes else "",
                "note": p.true_note,
                "delay_min": p.delay_min,
                "active_patterns": "|".join(s.active_patterns),
            })
    return pd.DataFrame(rows)


def observations_to_dataframe(obs: List[TrackerObservation]) -> pd.DataFrame:
    """Convert tracker observations to a tidy DataFrame."""
    rows = []
    for o in obs:
        rows.append({
            "session_id": o.session_id,
            "phase": o.phase_key,
            "recorded_duration_min": o.recorded_duration_min,
            "nurse": o.nurse_id,
            "is_extraordinary": o.is_extraordinary,
            "reason_codes": "|".join(o.reason_codes) if o.reason_codes else "",
            "free_text_note": o.free_text_note,
            "is_missing": o.is_missing,
        })
    return pd.DataFrame(rows)


def run_simulation(n_sessions: int = 150) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Run the full simulation pipeline: generate ground truth + tracker data.

    Returns:
        ground_truth_df: DataFrame with true values
        tracker_df: DataFrame with simulated observations
        metadata: generation metadata including planted patterns
    """
    SIM_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("CHECKLIST VALIDATION: Generating Synthetic Data")
    print("=" * 60)

    # Generate ground truth
    sessions, metadata = generate_ground_truth(n_sessions=n_sessions)
    gt_df = sessions_to_dataframe(sessions)

    print(f"  Generated {n_sessions} sessions across {gt_df['date'].nunique()} days")
    print(f"  90th percentile threshold: {metadata['p90_threshold_min']} min")
    print(f"  Outliers: {metadata['n_outliers']} / {n_sessions}")
    print(f"  Planted patterns: {len(PLANTED_PATTERNS)}")

    # Simulate tracker data at multiple noise levels for robustness testing
    tracker_obs = simulate_tracker_data(sessions)
    tracker_df = observations_to_dataframe(tracker_obs)

    missing_pct = tracker_df["is_missing"].mean() * 100
    print(f"  Tracker observations: {len(tracker_df)} ({missing_pct:.1f}% missing)")

    # Save artifacts
    gt_df.to_csv(SIM_DIR / "ground_truth.csv", index=False)
    tracker_df.to_csv(SIM_DIR / "tracker_observations.csv", index=False)
    with open(SIM_DIR / "simulation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved: ground_truth.csv, tracker_observations.csv, simulation_metadata.json")

    return gt_df, tracker_df, metadata
