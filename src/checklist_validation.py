"""
Checklist Validation — Phase 2: Analysis & Ground-Truth Comparison
===================================================================
Runs the same analyses a real team would run on tracker data, then
compares the recovered insights to the planted ground-truth patterns
to produce concrete validation metrics.

Validation dimensions:
  1. Phase variability detection  — can we identify the most variable phases?
  2. Nurse-phase effect detection — can we flag the planted nurse weaknesses?
  3. Delay-cause attribution      — do reason codes predict the right extra time?
  4. Qualitative coding accuracy  — do themes from notes match actual causes?
  5. Real vs simulated comparison  — are synthetic distributions statistically similar to actual data?
  6. Robustness / sensitivity     — how do results degrade with noise & missingness?
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.config import OUTPUT_DIR, RANDOM_STATE
from src.checklist_simulation import (
    BASELINE_DURATIONS,
    NURSES,
    PHASE_KEYS,
    PHASE_LABELS,
    PLANTED_PATTERNS,
    generate_ground_truth,
    simulate_tracker_data,
    sessions_to_dataframe,
    observations_to_dataframe,
)

SIM_DIR = OUTPUT_DIR / "checklist_validation"
SAVE_DPI = 150

# ── Qualitative keyword mapping for thematic coding ──────────────────────────
# Maps themes to regex patterns that would appear in free-text notes.
THEME_KEYWORDS: Dict[str, List[str]] = {
    "difficult_airway": [
        r"difficult\s+intub", r"video\s+laryngoscope", r"cormack",
        r"airway\s+manage", r"two\s+attempt", r"second\s+attempt",
        r"backup\s+airway", r"neck\s+extension",
    ],
    "equipment_issue": [
        r"not\s+powered", r"catheter\s+tray\s+missing", r"self-test",
        r"not\s+restocked", r"backup\s+unit", r"mapping\s+system",
        r"generator\s+fail",
    ],
    "staff_delay": [
        r"pulled\s+to\s+assist", r"delayed.*finishing", r"handoff\s+delay",
        r"waiting.*nurse", r"sign-out", r"emergent\s+case",
    ],
    "patient_factor": [
        r"body\s+habitus", r"mobility\s+limited", r"slide\s+board",
        r"repositioning", r"anticoagul", r"morbid\s+obesity",
    ],
    "sterility_breach": [
        r"sterile\s+field\s+compromised", r"re-draped", r"packaging.*torn",
    ],
    "communication": [
        r"review\s+imaging", r"timeout\s+repeated", r"laterality",
        r"clarification",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Quantitative analysis on tracker data
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_phase_variability(tracker_df: pd.DataFrame) -> pd.DataFrame:
    """Identify the most variable sub-phases from tracker data."""
    valid = tracker_df[~tracker_df["is_missing"]].copy()
    phase_stats = valid.groupby("phase")["recorded_duration_min"].agg(
        ["mean", "std", "median", "min", "max", "count"]
    )
    phase_stats["cv_pct"] = (phase_stats["std"] / phase_stats["mean"] * 100).round(1)
    phase_stats = phase_stats.sort_values("cv_pct", ascending=False)
    return phase_stats


def analyze_nurse_phase_performance(tracker_df: pd.DataFrame) -> pd.DataFrame:
    """Detect nurse-phase combinations with significantly higher durations."""
    valid = tracker_df[~tracker_df["is_missing"]].copy()
    results = []

    for phase in PHASE_KEYS:
        phase_data = valid[valid["phase"] == phase]
        global_mean = phase_data["recorded_duration_min"].mean()
        global_std = phase_data["recorded_duration_min"].std()

        for nurse in NURSES:
            nurse_data = phase_data[phase_data["nurse"] == nurse]["recorded_duration_min"]
            other_data = phase_data[phase_data["nurse"] != nurse]["recorded_duration_min"]

            if len(nurse_data) < 3 or len(other_data) < 3:
                continue

            # Welch's t-test: is this nurse slower than others at this phase?
            t_stat, p_val = stats.ttest_ind(nurse_data, other_data, equal_var=False)
            effect_size = (nurse_data.mean() - other_data.mean())

            results.append({
                "phase": phase,
                "nurse": nurse,
                "nurse_mean": round(nurse_data.mean(), 1),
                "other_mean": round(other_data.mean(), 1),
                "effect_min": round(effect_size, 1),
                "n_nurse": len(nurse_data),
                "t_stat": round(t_stat, 2),
                "p_value": round(p_val, 4),
                "significant": p_val < 0.05 and effect_size > 0,
            })

    return pd.DataFrame(results)


def analyze_delay_causes(tracker_df: pd.DataFrame) -> pd.DataFrame:
    """Estimate extra time associated with each reason code."""
    valid = tracker_df[~tracker_df["is_missing"]].copy()
    results = []

    for phase in PHASE_KEYS:
        phase_data = valid[valid["phase"] == phase]
        baseline_mean = phase_data[
            phase_data["reason_codes"] == ""
        ]["recorded_duration_min"].mean()

        # Get unique reason codes in this phase
        all_codes = phase_data[phase_data["reason_codes"] != ""]["reason_codes"]
        code_counts = Counter()
        for codes_str in all_codes:
            for code in codes_str.split("|"):
                code_counts[code] += 1

        for code, count in code_counts.items():
            if count < 2:
                continue
            flagged = phase_data[
                phase_data["reason_codes"].str.contains(code, na=False)
            ]["recorded_duration_min"]
            unflagged = phase_data[
                ~phase_data["reason_codes"].str.contains(code, na=False)
            ]["recorded_duration_min"]

            extra = flagged.mean() - unflagged.mean()
            results.append({
                "phase": phase,
                "reason_code": code,
                "n_flagged": count,
                "flagged_mean": round(flagged.mean(), 1),
                "unflagged_mean": round(unflagged.mean(), 1),
                "estimated_extra_min": round(extra, 1),
            })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Qualitative thematic coding
# ═══════════════════════════════════════════════════════════════════════════════

def code_themes(tracker_df: pd.DataFrame) -> pd.DataFrame:
    """Apply thematic coding to free-text notes using keyword matching."""
    valid = tracker_df[
        (~tracker_df["is_missing"]) & (tracker_df["free_text_note"] != "")
    ].copy()

    coded_rows = []
    for _, row in valid.iterrows():
        note = row["free_text_note"].lower()
        matched_themes = []
        for theme, patterns in THEME_KEYWORDS.items():
            if any(re.search(p, note) for p in patterns):
                matched_themes.append(theme)

        coded_rows.append({
            "session_id": row["session_id"],
            "phase": row["phase"],
            "note": row["free_text_note"],
            "coded_themes": "|".join(matched_themes) if matched_themes else "unclassified",
            "n_themes": len(matched_themes),
        })

    return pd.DataFrame(coded_rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Validation: compare analysis output to ground truth
# ═══════════════════════════════════════════════════════════════════════════════

def validate_phase_variability(
    phase_stats: pd.DataFrame,
    gt_df: pd.DataFrame,
) -> Dict:
    """Check if the analysis correctly identifies the most variable phases."""
    # Ground truth: compute true CV per phase
    true_stats = gt_df.groupby("phase")["true_duration_min"].agg(["mean", "std"])
    true_stats["cv_pct"] = (true_stats["std"] / true_stats["mean"] * 100).round(1)
    true_ranking = true_stats.sort_values("cv_pct", ascending=False).index.tolist()

    # Analysis ranking
    analysis_ranking = phase_stats.index.tolist()

    # Compare top-3 overlap
    true_top3 = set(true_ranking[:3])
    analysis_top3 = set(analysis_ranking[:3])
    top3_overlap = len(true_top3 & analysis_top3)

    # Rank correlation
    common = set(true_ranking) & set(analysis_ranking)
    if len(common) >= 3:
        true_ranks = [true_ranking.index(p) for p in common]
        analysis_ranks = [analysis_ranking.index(p) for p in common]
        rho, p_rho = stats.spearmanr(true_ranks, analysis_ranks)
    else:
        rho, p_rho = 0.0, 1.0

    return {
        "true_top3": list(true_top3),
        "analysis_top3": list(analysis_top3),
        "top3_overlap": top3_overlap,
        "top3_overlap_pct": round(top3_overlap / 3 * 100, 0),
        "rank_correlation_rho": round(float(rho), 3),
        "rank_correlation_p": round(float(p_rho), 4),
    }


def validate_nurse_effects(
    nurse_results: pd.DataFrame,
    gt_df: pd.DataFrame,
) -> Dict:
    """Check if planted nurse-phase weaknesses were detected."""
    planted_effects = [
        ("RN-02", "airway", "RN-02 slow at airway"),
        ("RN-04", "sterile_draping", "RN-04 slow at sterile draping"),
    ]

    detections = []
    for nurse, phase, pattern_name in planted_effects:
        match = nurse_results[
            (nurse_results["nurse"] == nurse) & (nurse_results["phase"] == phase)
        ]
        if len(match) > 0:
            row = match.iloc[0]
            detections.append({
                "pattern": pattern_name,
                "detected": bool(row["significant"]),
                "estimated_effect_min": row["effect_min"],
                "p_value": row["p_value"],
            })
        else:
            detections.append({
                "pattern": pattern_name,
                "detected": False,
                "estimated_effect_min": None,
                "p_value": None,
            })

    # Check false positives: significant results for non-planted combinations
    true_positives = {(n, p) for n, p, _ in planted_effects}
    false_positives = nurse_results[
        nurse_results["significant"] &
        ~nurse_results.apply(lambda r: (r["nurse"], r["phase"]) in true_positives, axis=1)
    ]

    n_detected = sum(1 for d in detections if d["detected"])
    return {
        "planted": len(planted_effects),
        "detected": n_detected,
        "sensitivity": round(n_detected / len(planted_effects) * 100, 0),
        "false_positives": len(false_positives),
        "false_positive_details": false_positives[["nurse", "phase", "effect_min", "p_value"]].to_dict("records")
            if len(false_positives) > 0 else [],
        "detections": detections,
    }


def validate_delay_causes(
    cause_results: pd.DataFrame,
    metadata: Dict,
) -> Dict:
    """Check if estimated delay magnitudes match planted ground truth."""
    planted_delays = {
        ("equipment_readiness", "equipment_not_ready"): (4.0, 8.0),
        ("airway", "difficult_airway"): (5.0, 9.0),
        ("monitoring_setup", "waiting_staff"): (3.0, 6.0),
    }

    validations = []
    for (phase, code), (low, high) in planted_delays.items():
        match = cause_results[
            (cause_results["phase"] == phase) & (cause_results["reason_code"] == code)
        ]
        if len(match) > 0:
            est = match.iloc[0]["estimated_extra_min"]
            in_range = low <= est <= high
            # Allow 1 min tolerance for noise
            near_range = (low - 1.0) <= est <= (high + 1.0)
            validations.append({
                "phase": phase,
                "reason_code": code,
                "planted_range": f"{low}-{high} min",
                "estimated_min": est,
                "exact_match": in_range,
                "within_tolerance": near_range,
            })
        else:
            validations.append({
                "phase": phase,
                "reason_code": code,
                "planted_range": f"{low}-{high} min",
                "estimated_min": None,
                "exact_match": False,
                "within_tolerance": False,
            })

    n_exact = sum(1 for v in validations if v["exact_match"])
    n_tolerant = sum(1 for v in validations if v["within_tolerance"])
    return {
        "planted": len(planted_delays),
        "exact_matches": n_exact,
        "within_tolerance": n_tolerant,
        "accuracy_exact_pct": round(n_exact / len(planted_delays) * 100, 0),
        "accuracy_tolerant_pct": round(n_tolerant / len(planted_delays) * 100, 0),
        "details": validations,
    }


def validate_qualitative_coding(
    coded_df: pd.DataFrame,
    gt_df: pd.DataFrame,
) -> Dict:
    """Check if thematic coding recovers the correct delay reasons."""
    # Map reason codes to expected themes
    code_to_theme = {
        "difficult_airway": "difficult_airway",
        "equipment_not_ready": "equipment_issue",
        "waiting_staff": "staff_delay",
        "patient_reposition": "patient_factor",
        "patient_complexity": "patient_factor",
        "sterility_issue": "sterility_breach",
        "communication_delay": "communication",
    }

    # For each coded note, check if the theme matches the ground-truth reason code
    gt_notes = gt_df[gt_df["note"] != ""][["session_id", "phase", "reason_codes", "note"]].copy()
    gt_notes = gt_notes.rename(columns={"reason_codes": "true_codes"})

    if coded_df.empty:
        return {"n_notes": 0, "accuracy_pct": 0, "details": []}

    merged = coded_df.merge(gt_notes, on=["session_id", "phase"], how="inner", suffixes=("", "_gt"))

    correct = 0
    total = 0
    for _, row in merged.iterrows():
        true_codes = row["true_codes"].split("|") if row["true_codes"] else []
        coded_themes = row["coded_themes"].split("|") if row["coded_themes"] else []

        expected_themes = set()
        for tc in true_codes:
            if tc in code_to_theme:
                expected_themes.add(code_to_theme[tc])

        if expected_themes:
            total += 1
            if expected_themes & set(coded_themes):
                correct += 1

    return {
        "n_notes_coded": len(coded_df),
        "n_matched_to_ground_truth": total,
        "n_correctly_themed": correct,
        "accuracy_pct": round(correct / total * 100, 0) if total > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Real vs Simulated comparison + Outlier classification + Confusion matrix
# ═══════════════════════════════════════════════════════════════════════════════

def validate_real_vs_simulated(
    real_df: pd.DataFrame,
    gt_df: pd.DataFrame,
) -> Dict:
    """Statistical comparison of real PT PREP/INTUBATION vs simulated total prep times.

    The simulated total_prep_min is the wall-clock span (accounting for parallel
    phases), which should approximate the real PT PREP/INTUBATION column.

    Tests:
      - Kolmogorov-Smirnov: are the distributions the same shape?
      - Mann-Whitney U: are the central tendencies comparable?
      - Levene's test: are the variances comparable?
      - Summary stat comparison: mean, std, skewness, kurtosis, percentiles
    """
    real_prep = real_df["PT PREP/INTUBATION"].dropna().values
    # Use wall-clock total (already computed with parallel overlap), not sum of durations
    sim_prep = gt_df.groupby("session_id")["total_prep_min"].first().values

    # ── Distribution tests ──
    ks_stat, ks_p = stats.ks_2samp(real_prep, sim_prep)
    mw_stat, mw_p = stats.mannwhitneyu(real_prep, sim_prep, alternative="two-sided")
    lev_stat, lev_p = stats.levene(real_prep, sim_prep)

    # ── Summary stats ──
    def _summary(arr):
        return {
            "n": len(arr),
            "mean": round(float(np.mean(arr)), 1),
            "median": round(float(np.median(arr)), 1),
            "std": round(float(np.std(arr, ddof=1)), 1),
            "min": round(float(np.min(arr)), 1),
            "max": round(float(np.max(arr)), 1),
            "skewness": round(float(stats.skew(arr)), 2),
            "kurtosis": round(float(stats.kurtosis(arr)), 2),
            "p10": round(float(np.percentile(arr, 10)), 1),
            "p25": round(float(np.percentile(arr, 25)), 1),
            "p75": round(float(np.percentile(arr, 75)), 1),
            "p90": round(float(np.percentile(arr, 90)), 1),
        }

    real_summary = _summary(real_prep)
    sim_summary = _summary(sim_prep)

    # ── Effect size (Cohen's d) ──
    pooled_std = np.sqrt(
        ((len(real_prep) - 1) * np.std(real_prep, ddof=1)**2 +
         (len(sim_prep) - 1) * np.std(sim_prep, ddof=1)**2) /
        (len(real_prep) + len(sim_prep) - 2)
    )
    cohens_d = abs(np.mean(real_prep) - np.mean(sim_prep)) / pooled_std if pooled_std > 0 else 0

    # ── Per-physician comparison (real physician distribution vs sim) ──
    phys_comparisons = {}
    for phys in sorted(real_df["PHYSICIAN"].dropna().unique()):
        phys_real = real_df[real_df["PHYSICIAN"] == phys]["PT PREP/INTUBATION"].dropna().values
        phys_sim_sessions = gt_df[gt_df["physician"] == phys]
        if len(phys_sim_sessions) == 0:
            continue
        phys_sim = phys_sim_sessions.groupby("session_id")["total_prep_min"].first().values
        if len(phys_real) < 3 or len(phys_sim) < 3:
            continue
        ks_s, ks_pp = stats.ks_2samp(phys_real, phys_sim)
        phys_comparisons[phys] = {
            "real_mean": round(float(np.mean(phys_real)), 1),
            "sim_mean": round(float(np.mean(phys_sim)), 1),
            "real_std": round(float(np.std(phys_real, ddof=1)), 1),
            "sim_std": round(float(np.std(phys_sim, ddof=1)), 1),
            "ks_stat": round(float(ks_s), 3),
            "ks_p": round(float(ks_pp), 4),
            "distributions_similar": ks_pp > 0.05,
        }

    return {
        "tests": {
            "ks_test": {
                "statistic": round(float(ks_stat), 3),
                "p_value": round(float(ks_p), 4),
                "interpretation": "Distributions similar (fail to reject H0)" if ks_p > 0.05
                    else "Distributions differ significantly",
            },
            "mann_whitney_u": {
                "statistic": round(float(mw_stat), 1),
                "p_value": round(float(mw_p), 4),
                "interpretation": "Central tendencies similar" if mw_p > 0.05
                    else "Central tendencies differ significantly",
            },
            "levene_variance": {
                "statistic": round(float(lev_stat), 3),
                "p_value": round(float(lev_p), 4),
                "interpretation": "Variances similar" if lev_p > 0.05
                    else "Variances differ significantly",
            },
            "cohens_d": round(float(cohens_d), 3),
            "effect_size_label": (
                "negligible" if cohens_d < 0.2 else
                "small" if cohens_d < 0.5 else
                "medium" if cohens_d < 0.8 else "large"
            ),
        },
        "real_summary": real_summary,
        "sim_summary": sim_summary,
        "per_physician": phys_comparisons,
        "verdict": "PASS" if (ks_p > 0.05 and cohens_d < 0.5) else
                   "MARGINAL" if (ks_p > 0.01 or cohens_d < 0.8) else "FAIL",
    }


def validate_outlier_classification(
    real_df: pd.DataFrame,
    gt_df: pd.DataFrame,
) -> Dict:
    """Compare outlier classification between real and simulated data.

    Real outliers are based on PT IN-OUT (the target variable), while simulated
    outliers are based on total wall-clock prep time. The confusion matrix tests
    whether the sim threshold, applied to sim data, agrees with the real threshold
    applied to the same sim data — i.e., do the two thresholds produce similar
    classifications?
    """
    # Real: outliers based on PT IN-OUT (the actual target variable)
    real_target = real_df["PT IN-OUT"].dropna()
    # Also get real prep times for profile comparison
    real_prep = real_df["PT PREP/INTUBATION"].dropna()
    # Simulated: wall-clock prep times
    sim_totals = gt_df.groupby("session_id")["total_prep_min"].first()

    # Thresholds (both at 90th percentile of their respective distributions)
    real_target_p90 = float(np.percentile(real_target, 90))
    real_prep_p90 = float(np.percentile(real_prep, 90))
    sim_p90 = float(np.percentile(sim_totals, 90))

    real_prep_outlier_vals = real_prep[real_prep >= real_prep_p90]
    real_prep_normal_vals = real_prep[real_prep < real_prep_p90]
    sim_outlier_vals = sim_totals[sim_totals >= sim_p90]
    sim_normal_vals = sim_totals[sim_totals < sim_p90]

    # ── Profile comparison: do outliers/normals look the same? ──
    def _profile(vals):
        return {
            "n": len(vals),
            "mean": round(float(vals.mean()), 1),
            "std": round(float(vals.std()), 1),
            "min": round(float(vals.min()), 1),
            "max": round(float(vals.max()), 1),
        }

    # ── Cross-classification: apply real prep P90 threshold to sim data ──
    # This tests: if we used the real-world threshold on our simulated data,
    # would we get the same outlier labels as the sim's own threshold?
    sim_classified_by_real_threshold = sim_totals >= real_prep_p90
    sim_classified_by_sim_threshold = sim_totals >= sim_p90

    # Confusion matrix: sim threshold (predicted) vs real threshold (reference)
    tp = int(( sim_classified_by_sim_threshold &  sim_classified_by_real_threshold).sum())
    fp = int(( sim_classified_by_sim_threshold & ~sim_classified_by_real_threshold).sum())
    fn = int((~sim_classified_by_sim_threshold &  sim_classified_by_real_threshold).sum())
    tn = int((~sim_classified_by_sim_threshold & ~sim_classified_by_real_threshold).sum())

    total = tp + fp + fn + tn
    accuracy = round((tp + tn) / total * 100, 1) if total > 0 else 0
    precision = round(tp / (tp + fp) * 100, 1) if (tp + fp) > 0 else 0
    recall = round(tp / (tp + fn) * 100, 1) if (tp + fn) > 0 else 0
    f1 = round(2 * precision * recall / (precision + recall), 1) if (precision + recall) > 0 else 0
    kappa_po = (tp + tn) / total if total > 0 else 0
    kappa_pe = (
        ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total ** 2)
    ) if total > 0 else 0
    kappa = round((kappa_po - kappa_pe) / (1 - kappa_pe), 3) if kappa_pe < 1 else 0

    # ── Statistical test: are outlier rates the same? ──
    real_rate = len(real_prep_outlier_vals) / len(real_prep)
    sim_rate = len(sim_outlier_vals) / len(sim_totals)
    # Two-proportion z-test
    pooled_rate = (len(real_prep_outlier_vals) + len(sim_outlier_vals)) / (len(real_prep) + len(sim_totals))
    se = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/len(real_prep) + 1/len(sim_totals)))
    z_stat = (real_rate - sim_rate) / se if se > 0 else 0
    z_p = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return {
        "thresholds": {
            "real_pt_inout_p90": round(real_target_p90, 1),
            "real_prep_p90": round(real_prep_p90, 1),
            "sim_prep_p90": round(sim_p90, 1),
            "difference_min": round(abs(real_prep_p90 - sim_p90), 1),
        },
        "outlier_rates": {
            "real_pct": round(real_rate * 100, 1),
            "sim_pct": round(sim_rate * 100, 1),
            "z_stat": round(float(z_stat), 3),
            "z_p_value": round(float(z_p), 4),
            "rates_similar": z_p > 0.05,
        },
        "profiles": {
            "real_outlier": _profile(real_prep_outlier_vals),
            "real_normal": _profile(real_prep_normal_vals),
            "sim_outlier": _profile(sim_outlier_vals),
            "sim_normal": _profile(sim_normal_vals),
        },
        "confusion_matrix": {
            "description": "Sim's own threshold vs Real threshold applied to sim data",
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy_pct": accuracy,
            "precision_pct": precision,
            "recall_pct": recall,
            "f1_pct": f1,
            "cohens_kappa": kappa,
        },
        "verdict": "PASS" if (accuracy >= 80 and z_p > 0.05) else
                   "MARGINAL" if accuracy >= 60 else "FAIL",
    }


def compare_baseline_vs_tracker(
    gt_df: pd.DataFrame,
    tracker_df: pd.DataFrame,
) -> Dict:
    """Compare outlier prediction: single aggregate feature vs granular tracker data.

    Outlier labels are defined by total wall-clock prep time (90th percentile).
    Neither model uses total_prep_min directly — that would be circular.

    Baseline model: logistic regression using the MEAN of individual phase durations
                    as a crude proxy for overall prep length (simulates having only
                    one aggregate number, like PT PREP/INTUBATION).
    Tracker model:  logistic regression using 8 sub-phase durations + nurse dummies +
                    reason-code flags + extraordinary count (what the tracker captures).

    The tracker model has access to WHERE time was spent, WHO was involved, and
    WHETHER extraordinary events occurred — the baseline has none of this.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score

    # Build session-level feature matrices from tracker data
    valid = tracker_df[~tracker_df["is_missing"]].copy()

    # Pivot: one row per session, columns = phase durations
    phase_pivot = valid.pivot_table(
        index="session_id", columns="phase",
        values="recorded_duration_min", aggfunc="first",
    ).reset_index()

    # Nurse dummy encoding per session
    nurse_per_session = valid.groupby("session_id")["nurse"].first().reset_index()
    nurse_dummies = pd.get_dummies(nurse_per_session["nurse"], prefix="nurse")
    nurse_dummies["session_id"] = nurse_per_session["session_id"]

    # Reason code flags per session
    reason_flags = valid.groupby("session_id").agg(
        has_extraordinary=("is_extraordinary", "any"),
        n_reason_codes=("reason_codes", lambda x: sum(1 for v in x if v != "")),
    ).reset_index()
    reason_flags["has_extraordinary"] = reason_flags["has_extraordinary"].astype(int)

    # Ground-truth labels
    session_labels = gt_df.groupby("session_id").agg(
        is_outlier=("is_outlier", "first"),
    ).reset_index()

    # Merge everything
    merged = (
        session_labels
        .merge(phase_pivot, on="session_id", how="inner")
        .merge(nurse_dummies, on="session_id", how="left")
        .merge(reason_flags, on="session_id", how="left")
    )
    merged = merged.dropna()

    y = merged["is_outlier"].astype(int).values
    if y.sum() < 2 or (1 - y).sum() < 2:
        return {"error": "Not enough outliers/normals for modeling"}

    # ── Phase columns available ──
    from src.checklist_simulation import PHASE_KEYS
    phase_cols = [c for c in PHASE_KEYS if c in merged.columns]

    # ── Baseline features: mean of individual phase durations ──
    # This is a crude proxy for "one aggregate number" — correlated with
    # wall-clock total but NOT identical (phases overlap in parallel).
    merged["mean_phase_duration"] = merged[phase_cols].mean(axis=1)
    X_baseline = merged[["mean_phase_duration"]].values

    # ── Tracker features: sub-phase durations + nurse dummies + reason flags ──
    nurse_cols = [c for c in merged.columns if c.startswith("nurse_")]
    tracker_feature_cols = phase_cols + nurse_cols + ["has_extraordinary", "n_reason_codes"]
    X_tracker = merged[tracker_feature_cols].values

    # ── Cross-validated predictions (5-fold stratified) ──
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def _run_model(X, y_true, cv_obj, label=""):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(
            class_weight="balanced", random_state=42, max_iter=1000,
        )
        y_pred = cross_val_predict(model, X_scaled, y_true, cv=cv_obj, method="predict")
        y_prob = cross_val_predict(model, X_scaled, y_true, cv=cv_obj, method="predict_proba")[:, 1]

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        total = tp + fp + fn + tn
        acc = round((tp + tn) / total * 100, 1) if total > 0 else 0
        prec = round(tp / (tp + fp) * 100, 1) if (tp + fp) > 0 else 0
        rec = round(tp / (tp + fn) * 100, 1) if (tp + fn) > 0 else 0
        f1 = round(2 * prec * rec / (prec + rec), 1) if (prec + rec) > 0 else 0

        try:
            auc = round(float(roc_auc_score(y_true, y_prob)), 3)
        except ValueError:
            auc = None

        return {
            "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "accuracy_pct": acc,
            "precision_pct": prec,
            "recall_pct": rec,
            "f1_pct": f1,
            "auc": auc,
        }

    baseline_results = _run_model(X_baseline, y, cv, "baseline")
    tracker_results = _run_model(X_tracker, y, cv, "tracker")

    # ── Determine winner ──
    baseline_f1 = baseline_results["f1_pct"]
    tracker_f1 = tracker_results["f1_pct"]
    improvement = round(tracker_f1 - baseline_f1, 1)

    auc_baseline = baseline_results["auc"] or 0
    auc_tracker = tracker_results["auc"] or 0
    auc_improvement = round(auc_tracker - auc_baseline, 3)

    return {
        "baseline": {
            "description": "Mean of phase durations (single aggregate — simulates PT PREP/INTUBATION)",
            "n_features": 1,
            **baseline_results,
        },
        "tracker": {
            "description": "8 sub-phase durations + nurse ID + reason codes + extraordinary flag",
            "n_features": len(tracker_feature_cols),
            "feature_names": tracker_feature_cols,
            **tracker_results,
        },
        "comparison": {
            "f1_improvement_pct": improvement,
            "auc_improvement": auc_improvement,
            "tracker_is_better": tracker_f1 > baseline_f1,
            "verdict": (
                "PASS" if improvement > 0 else
                "MARGINAL" if improvement == 0 else
                "FAIL"
            ),
        },
        "n_sessions": len(merged),
        "n_outliers": int(y.sum()),
    }


def plot_model_comparison(comparison: Dict) -> None:
    """Side-by-side confusion matrices and metric comparison for baseline vs tracker."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Baseline (Aggregate Prep Time) vs Tracker (Sub-Phase Data): Outlier Prediction",
                 fontsize=13, fontweight="bold")

    for idx, (label, key, color) in enumerate([
        ("Baseline: 1 Feature", "baseline", "Oranges"),
        ("Tracker: Sub-Phase Data", "tracker", "Greens"),
    ]):
        ax = axes[idx]
        cm = comparison[key]["confusion_matrix"]
        cm_arr = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        im = ax.imshow(cm_arr, cmap=color, aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Normal", "Pred Outlier"])
        ax.set_yticklabels(["True Normal", "True Outlier"])
        f1 = comparison[key]["f1_pct"]
        auc = comparison[key].get("auc", "N/A")
        ax.set_title(f"{label}\nF1={f1}%  AUC={auc}", fontsize=11)
        for i in range(2):
            for j in range(2):
                val = cm_arr[i, j]
                txt_color = "white" if val > cm_arr.max() / 2 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=18, fontweight="bold", color=txt_color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (2) — Metric comparison bar chart
    ax = axes[2]
    metrics = ["accuracy_pct", "precision_pct", "recall_pct", "f1_pct"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(metrics))
    width = 0.35
    baseline_vals = [comparison["baseline"][m] for m in metrics]
    tracker_vals = [comparison["tracker"][m] for m in metrics]

    bars1 = ax.bar(x - width/2, baseline_vals, width, label="Baseline",
                   color="#e67e22", edgecolor="black", alpha=0.8)
    bars2 = ax.bar(x + width/2, tracker_vals, width, label="Tracker",
                   color="#27ae60", edgecolor="black", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Metric Comparison")
    ax.legend(fontsize=9)

    # Add AUC annotation
    auc_b = comparison["baseline"].get("auc", 0)
    auc_t = comparison["tracker"].get("auc", 0)
    imp = comparison["comparison"]["f1_improvement_pct"]
    verdict = comparison["comparison"]["verdict"]
    ax.text(0.5, -0.18,
            f"AUC: Baseline={auc_b}  Tracker={auc_t}  |  "
            f"F1 improvement: {imp:+.1f}%  |  Verdict: {verdict}",
            transform=ax.transAxes, ha="center", fontsize=10,
            style="italic", color="#2c3e50",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1", alpha=0.8))

    plt.tight_layout()
    plt.savefig(SIM_DIR / "baseline_vs_tracker.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print("  Saved: baseline_vs_tracker.png")


def plot_real_vs_simulated(
    real_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    comparison: Dict,
    outlier_cls: Dict,
) -> None:
    """Visualization comparing real vs simulated distributions with significance tests."""
    real_prep = real_df["PT PREP/INTUBATION"].dropna().values
    sim_prep = gt_df.groupby("session_id")["total_prep_min"].first().values

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Real vs Simulated Data: Statistical Comparison",
                 fontsize=15, fontweight="bold")

    # (0,0) — Overlaid histograms
    ax = axes[0, 0]
    bins = np.linspace(
        min(real_prep.min(), sim_prep.min()) - 2,
        max(real_prep.max(), sim_prep.max()) + 2,
        30,
    )
    ax.hist(real_prep, bins=bins, alpha=0.5, color="#3498db", edgecolor="black",
            label=f"Real (n={len(real_prep)}, μ={np.mean(real_prep):.1f})", density=True)
    ax.hist(sim_prep, bins=bins, alpha=0.5, color="#e74c3c", edgecolor="black",
            label=f"Simulated (n={len(sim_prep)}, μ={np.mean(sim_prep):.1f})", density=True)
    ax.set_xlabel("Total Prep Time (min)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution Comparison")
    ax.legend(fontsize=8)

    # (0,1) — QQ plot
    ax = axes[0, 1]
    real_sorted = np.sort(real_prep)
    sim_sorted = np.sort(sim_prep)
    # Interpolate to same length for QQ
    n_points = min(len(real_sorted), len(sim_sorted))
    real_quantiles = np.percentile(real_prep, np.linspace(0, 100, n_points))
    sim_quantiles = np.percentile(sim_prep, np.linspace(0, 100, n_points))
    ax.scatter(real_quantiles, sim_quantiles, alpha=0.6, s=20, color="#9b59b6", edgecolor="black")
    qq_min = min(real_quantiles.min(), sim_quantiles.min())
    qq_max = max(real_quantiles.max(), sim_quantiles.max())
    ax.plot([qq_min, qq_max], [qq_min, qq_max], "r--", linewidth=2, label="Perfect match")
    ax.set_xlabel("Real Quantiles (min)")
    ax.set_ylabel("Simulated Quantiles (min)")
    ax.set_title("Q-Q Plot")
    ax.legend(fontsize=8)

    # (0,2) — Test results summary table
    ax = axes[0, 2]
    ax.axis("off")
    tests = comparison["tests"]
    table_data = [
        ["Test", "Statistic", "p-value", "Result"],
        ["KS Test", f"{tests['ks_test']['statistic']:.3f}",
         f"{tests['ks_test']['p_value']:.4f}",
         "Similar" if tests["ks_test"]["p_value"] > 0.05 else "Different"],
        ["Mann-Whitney U", f"{tests['mann_whitney_u']['statistic']:.0f}",
         f"{tests['mann_whitney_u']['p_value']:.4f}",
         "Similar" if tests["mann_whitney_u"]["p_value"] > 0.05 else "Different"],
        ["Levene's", f"{tests['levene_variance']['statistic']:.3f}",
         f"{tests['levene_variance']['p_value']:.4f}",
         "Similar" if tests["levene_variance"]["p_value"] > 0.05 else "Different"],
        ["Cohen's d", f"{tests['cohens_d']:.3f}", "—", tests["effect_size_label"]],
    ]
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    # Header row styling
    for j in range(4):
        table[0, j].set_facecolor("#34495e")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Color result cells
    for i in range(1, 5):
        cell = table[i, 3]
        text = cell.get_text().get_text()
        if text == "Similar" or text == "negligible" or text == "small":
            cell.set_facecolor("#d5f5e3")
        elif text == "Different" or text == "large":
            cell.set_facecolor("#fadbd8")
        else:
            cell.set_facecolor("#fdebd0")
    ax.set_title("Statistical Test Results", fontweight="bold", pad=20)

    # (1,0) — Box plots side by side
    ax = axes[1, 0]
    bp = ax.boxplot(
        [real_prep, sim_prep],
        labels=["Real", "Simulated"],
        patch_artist=True,
        widths=0.5,
    )
    bp["boxes"][0].set_facecolor("#3498db")
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("#e74c3c")
    bp["boxes"][1].set_alpha(0.6)
    ax.set_ylabel("Total Prep Time (min)")
    ax.set_title("Distribution Spread")

    # (1,1) — Confusion matrix heatmap
    ax = axes[1, 1]
    cm = outlier_cls["confusion_matrix"]
    cm_array = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    im = ax.imshow(cm_array, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal\n(Real threshold)", "Outlier\n(Real threshold)"])
    ax.set_yticklabels(["Normal\n(Sim threshold)", "Outlier\n(Sim threshold)"])
    ax.set_title(f"Confusion Matrix (Accuracy: {cm['accuracy_pct']}%)")
    for i in range(2):
        for j in range(2):
            val = cm_array[i, j]
            color = "white" if val > cm_array.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=18, fontweight="bold", color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (1,2) — Outlier profile comparison
    ax = axes[1, 2]
    profiles = outlier_cls["profiles"]
    x = np.arange(2)
    width = 0.35
    real_means = [profiles["real_normal"]["mean"], profiles["real_outlier"]["mean"]]
    sim_means = [profiles["sim_normal"]["mean"], profiles["sim_outlier"]["mean"]]
    real_stds = [profiles["real_normal"]["std"], profiles["real_outlier"]["std"]]
    sim_stds = [profiles["sim_normal"]["std"], profiles["sim_outlier"]["std"]]

    bars1 = ax.bar(x - width/2, real_means, width, yerr=real_stds, capsize=5,
                   label="Real", color="#3498db", edgecolor="black", alpha=0.7)
    bars2 = ax.bar(x + width/2, sim_means, width, yerr=sim_stds, capsize=5,
                   label="Simulated", color="#e74c3c", edgecolor="black", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["Normal Cases", "Outlier Cases"])
    ax.set_ylabel("Mean Prep Time (min)")
    ax.set_title("Outlier vs Normal Profiles")
    ax.legend(fontsize=8)

    # Add metric annotations
    metrics_text = (f"Precision: {cm['precision_pct']}%  |  "
                    f"Recall: {cm['recall_pct']}%  |  "
                    f"F1: {cm['f1_pct']}%  |  "
                    f"Cohen's κ: {cm['cohens_kappa']}")
    fig.text(0.5, 0.01, metrics_text, ha="center", fontsize=11,
             style="italic", color="#2c3e50",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1", alpha=0.8))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(SIM_DIR / "real_vs_simulated.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print("  Saved: real_vs_simulated.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Robustness / sensitivity analysis
# ═══════════════════════════════════════════════════════════════════════════════

def run_robustness_analysis(
    sessions,
    gt_df: pd.DataFrame,
) -> pd.DataFrame:
    """Test how validation metrics degrade with increasing noise/missingness."""
    configs = [
        {"label": "Ideal (0% missing, 0 noise)",   "missing_rate": 0.00, "noise_std": 0.0},
        {"label": "Low (5% missing, 0.3 noise)",    "missing_rate": 0.05, "noise_std": 0.3},
        {"label": "Baseline (8% missing, 0.5 noise)", "missing_rate": 0.08, "noise_std": 0.5},
        {"label": "High (15% missing, 1.0 noise)",  "missing_rate": 0.15, "noise_std": 1.0},
        {"label": "Severe (25% missing, 2.0 noise)", "missing_rate": 0.25, "noise_std": 2.0},
    ]

    results = []
    for cfg in configs:
        obs = simulate_tracker_data(
            sessions,
            missing_rate=cfg["missing_rate"],
            timing_noise_std=cfg["noise_std"],
            seed=RANDOM_STATE,
        )
        tdf = observations_to_dataframe(obs)

        # Run analyses
        ps = analyze_phase_variability(tdf)
        nr = analyze_nurse_phase_performance(tdf)

        # Validate
        pv = validate_phase_variability(ps, gt_df)
        nv = validate_nurse_effects(nr, gt_df)

        results.append({
            "config": cfg["label"],
            "missing_rate": cfg["missing_rate"],
            "noise_std": cfg["noise_std"],
            "phase_rank_rho": pv["rank_correlation_rho"],
            "phase_top3_overlap": pv["top3_overlap"],
            "nurse_sensitivity_pct": nv["sensitivity"],
            "nurse_false_positives": nv["false_positives"],
        })

    return pd.DataFrame(results)


def run_sample_size_analysis(
    gt_df: pd.DataFrame,
    sessions,
) -> pd.DataFrame:
    """Test minimum sample size needed for reliable detection."""
    sample_sizes = [30, 50, 75, 100, 125, 150]
    results = []

    for n in sample_sizes:
        sub_sessions = sessions[:n]
        sub_gt = gt_df[gt_df["session_id"] <= n]

        obs = simulate_tracker_data(sub_sessions, seed=RANDOM_STATE)
        tdf = observations_to_dataframe(obs)

        nr = analyze_nurse_phase_performance(tdf)
        nv = validate_nurse_effects(nr, sub_gt)

        results.append({
            "n_sessions": n,
            "nurse_sensitivity_pct": nv["sensitivity"],
            "nurse_false_positives": nv["false_positives"],
            "n_significant_findings": int(nr["significant"].sum()) if len(nr) > 0 else 0,
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_validation_summary(
    phase_validation: Dict,
    nurse_validation: Dict,
    cause_validation: Dict,
    qual_validation: Dict,
    robustness_df: pd.DataFrame,
    sample_size_df: pd.DataFrame,
) -> None:
    """Create a summary visualization of all validation results."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Checklist Tracker Validation: Monte Carlo Simulation Results",
                 fontsize=15, fontweight="bold")

    # (0,0) — Phase variability ranking accuracy
    ax = axes[0, 0]
    categories = ["Top-3 Overlap", "Rank Correlation"]
    values = [
        phase_validation["top3_overlap_pct"],
        phase_validation["rank_correlation_rho"] * 100,
    ]
    colors = ["#2ecc71" if v >= 66 else "#f39c12" if v >= 33 else "#e74c3c" for v in values]
    bars = ax.bar(categories, values, color=colors, edgecolor="black", alpha=0.8)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)")
    ax.set_title("Phase Variability Detection")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{v:.0f}%", ha="center", va="bottom", fontweight="bold")
    ax.axhline(y=66, color="gray", linestyle="--", alpha=0.5, label="Good threshold")
    ax.legend(fontsize=7)

    # (0,1) — Nurse effect detection
    ax = axes[0, 1]
    detections = nurse_validation["detections"]
    names = [d["pattern"].replace(" slow at ", "\n") for d in detections]
    detected = [1 if d["detected"] else 0 for d in detections]
    bar_colors = ["#2ecc71" if d else "#e74c3c" for d in detected]
    bars = ax.bar(names, [d.get("estimated_effect_min", 0) or 0 for d in detections],
                  color=bar_colors, edgecolor="black", alpha=0.8)
    ax.set_ylabel("Estimated Effect (min)")
    ax.set_title(f"Nurse Effect Detection ({nurse_validation['sensitivity']:.0f}% sensitivity)")

    # Add planted range as error bars
    planted_ranges = [(3.0, 5.0), (2.0, 4.0)]
    for i, (bar, (lo, hi)) in enumerate(zip(bars, planted_ranges)):
        mid = (lo + hi) / 2
        ax.plot([bar.get_x(), bar.get_x() + bar.get_width()], [mid, mid],
                color="blue", linewidth=2, linestyle="--")
        ax.fill_between(
            [bar.get_x(), bar.get_x() + bar.get_width()], lo, hi,
            alpha=0.15, color="blue",
        )
    ax.legend(["Planted range"], fontsize=7)

    # (0,2) — Delay cause estimation accuracy
    ax = axes[0, 2]
    cause_details = cause_validation["details"]
    cause_names = [f"{d['reason_code']}\n({d['phase'][:12]})" for d in cause_details]
    estimated = [d["estimated_min"] or 0 for d in cause_details]
    planted_mids = []
    planted_errs = []
    for d in cause_details:
        lo, hi = [float(x) for x in d["planted_range"].replace(" min", "").split("-")]
        planted_mids.append((lo + hi) / 2)
        planted_errs.append((hi - lo) / 2)

    x_pos = np.arange(len(cause_names))
    ax.bar(x_pos - 0.15, estimated, 0.3, label="Estimated", color="#3498db",
           edgecolor="black", alpha=0.8)
    ax.errorbar(x_pos + 0.15, planted_mids, yerr=planted_errs, fmt="s",
                color="#e74c3c", markersize=8, capsize=5, label="Planted range")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cause_names, fontsize=7)
    ax.set_ylabel("Extra Time (min)")
    ax.set_title(f"Delay Cause Attribution ({cause_validation['accuracy_tolerant_pct']:.0f}% within tolerance)")
    ax.legend(fontsize=7)

    # (1,0) — Qualitative coding accuracy
    ax = axes[1, 0]
    qual_labels = ["Notes Coded", "Matched to GT", "Correctly Themed"]
    qual_values = [
        qual_validation["n_notes_coded"],
        qual_validation["n_matched_to_ground_truth"],
        qual_validation["n_correctly_themed"],
    ]
    ax.bar(qual_labels, qual_values, color=["#3498db", "#f39c12", "#2ecc71"],
           edgecolor="black", alpha=0.8)
    ax.set_ylabel("Count")
    ax.set_title(f"Qualitative Coding ({qual_validation['accuracy_pct']:.0f}% accuracy)")
    for i, v in enumerate(qual_values):
        ax.text(i, v + 0.5, str(v), ha="center", va="bottom", fontweight="bold")

    # (1,1) — Robustness: metrics vs noise level
    ax = axes[1, 1]
    ax.plot(robustness_df["missing_rate"] * 100, robustness_df["nurse_sensitivity_pct"],
            "o-", color="#2ecc71", linewidth=2, markersize=8, label="Nurse sensitivity")
    ax.plot(robustness_df["missing_rate"] * 100, robustness_df["phase_rank_rho"] * 100,
            "s-", color="#3498db", linewidth=2, markersize=8, label="Phase rank ρ × 100")
    ax.set_xlabel("Missing Data Rate (%)")
    ax.set_ylabel("Score (%)")
    ax.set_title("Robustness: Metrics vs Data Quality")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 110)
    ax.axhline(y=66, color="gray", linestyle="--", alpha=0.4)

    # (1,2) — Sample size analysis
    ax = axes[1, 2]
    ax.plot(sample_size_df["n_sessions"], sample_size_df["nurse_sensitivity_pct"],
            "o-", color="#e74c3c", linewidth=2, markersize=8, label="Nurse sensitivity")
    ax.plot(sample_size_df["n_sessions"], sample_size_df["n_significant_findings"],
            "s-", color="#9b59b6", linewidth=2, markersize=8, label="# significant findings")
    ax.set_xlabel("Number of Sessions")
    ax.set_ylabel("Value")
    ax.set_title("Sample Size: Minimum Sessions Needed")
    ax.legend(fontsize=8)
    ax.axhline(y=100, color="#e74c3c", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(SIM_DIR / "validation_summary.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print("  Saved: validation_summary.png")


def plot_phase_comparison(gt_df: pd.DataFrame, tracker_df: pd.DataFrame) -> None:
    """Side-by-side: true vs recorded phase durations."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Ground Truth vs Tracker Recorded Durations by Phase",
                 fontsize=14, fontweight="bold")

    valid_tracker = tracker_df[~tracker_df["is_missing"]]

    for i, phase in enumerate(PHASE_KEYS):
        ax = axes[i // 4, i % 4]
        true_vals = gt_df[gt_df["phase"] == phase]["true_duration_min"]
        rec_vals = valid_tracker[valid_tracker["phase"] == phase]["recorded_duration_min"]

        ax.hist(true_vals, bins=20, alpha=0.5, color="#3498db", label="True", edgecolor="black")
        ax.hist(rec_vals, bins=20, alpha=0.5, color="#e74c3c", label="Recorded", edgecolor="black")
        ax.set_title(PHASE_LABELS[phase], fontsize=9)
        ax.set_xlabel("Minutes")
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(SIM_DIR / "phase_comparison.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print("  Saved: phase_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_validation(
    gt_df: pd.DataFrame,
    tracker_df: pd.DataFrame,
    metadata: Dict,
    real_df: pd.DataFrame | None = None,
) -> Dict:
    """Run full validation pipeline and produce outputs.

    Args:
        gt_df: ground-truth synthetic data
        tracker_df: simulated tracker observations
        metadata: simulation metadata
        real_df: actual procedure data with PT PREP/INTUBATION column (optional)

    Returns dict with all validation results.
    """
    SIM_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("CHECKLIST VALIDATION: Analyzing & Validating")
    print("=" * 60)

    has_real = real_df is not None and "PT PREP/INTUBATION" in real_df.columns
    n_steps = 8 if has_real else 6

    # ── Quantitative analysis ──
    print(f"\n  [1/{n_steps}] Phase variability analysis...")
    phase_stats = analyze_phase_variability(tracker_df)
    print(f"        Most variable phases: {', '.join(phase_stats.index[:3])}")

    print(f"  [2/{n_steps}] Nurse-phase performance analysis...")
    nurse_results = analyze_nurse_phase_performance(tracker_df)
    n_sig = nurse_results["significant"].sum() if len(nurse_results) > 0 else 0
    print(f"        Significant nurse-phase effects found: {n_sig}")

    print(f"  [3/{n_steps}] Delay cause attribution...")
    cause_results = analyze_delay_causes(tracker_df)
    print(f"        Reason codes analyzed: {len(cause_results)}")

    # ── Qualitative coding ──
    print(f"  [4/{n_steps}] Thematic coding of free-text notes...")
    coded_df = code_themes(tracker_df)
    print(f"        Notes coded: {len(coded_df)}")

    # ── Validation against ground truth ──
    print(f"  [5/{n_steps}] Validating against ground truth...")
    phase_val = validate_phase_variability(phase_stats, gt_df)
    nurse_val = validate_nurse_effects(nurse_results, gt_df)
    cause_val = validate_delay_causes(cause_results, metadata)
    qual_val = validate_qualitative_coding(coded_df, gt_df)

    print(f"        Phase variability: top-3 overlap = {phase_val['top3_overlap']}/3, "
          f"rank ρ = {phase_val['rank_correlation_rho']:.3f}")
    print(f"        Nurse effects: {nurse_val['detected']}/{nurse_val['planted']} detected "
          f"({nurse_val['sensitivity']:.0f}% sensitivity), "
          f"{nurse_val['false_positives']} false positives")
    print(f"        Delay causes: {cause_val['within_tolerance']}/{cause_val['planted']} "
          f"within tolerance ({cause_val['accuracy_tolerant_pct']:.0f}%)")
    print(f"        Qualitative coding: {qual_val['accuracy_pct']:.0f}% accuracy")

    # ── Real vs Simulated comparison ──
    real_vs_sim = None
    outlier_cls = None
    if has_real:
        print(f"  [6/{n_steps}] Real vs simulated comparison (significance tests + confusion matrix)...")
        real_vs_sim = validate_real_vs_simulated(real_df, gt_df)
        outlier_cls = validate_outlier_classification(real_df, gt_df)

        tests = real_vs_sim["tests"]
        print(f"        KS test: D={tests['ks_test']['statistic']:.3f}, "
              f"p={tests['ks_test']['p_value']:.4f} — {tests['ks_test']['interpretation']}")
        print(f"        Mann-Whitney U: p={tests['mann_whitney_u']['p_value']:.4f} — "
              f"{tests['mann_whitney_u']['interpretation']}")
        print(f"        Levene's test: p={tests['levene_variance']['p_value']:.4f} — "
              f"{tests['levene_variance']['interpretation']}")
        print(f"        Cohen's d: {tests['cohens_d']:.3f} ({tests['effect_size_label']})")
        cm = outlier_cls["confusion_matrix"]
        print(f"        Confusion matrix: accuracy={cm['accuracy_pct']}%, "
              f"precision={cm['precision_pct']}%, recall={cm['recall_pct']}%, "
              f"F1={cm['f1_pct']}%, Cohen's κ={cm['cohens_kappa']}")
        print(f"        Outlier rates: real={outlier_cls['outlier_rates']['real_pct']}% vs "
              f"sim={outlier_cls['outlier_rates']['sim_pct']}% "
              f"(p={outlier_cls['outlier_rates']['z_p_value']:.4f})")

    # ── Baseline vs Tracker model comparison ──
    model_comparison = None
    print(f"  [{'7' if has_real else '6'}/{n_steps}] Baseline vs tracker model comparison...")
    model_comparison = compare_baseline_vs_tracker(gt_df, tracker_df)
    if "error" not in model_comparison:
        comp = model_comparison["comparison"]
        print(f"        Baseline F1: {model_comparison['baseline']['f1_pct']}% | "
              f"AUC: {model_comparison['baseline']['auc']}")
        print(f"        Tracker  F1: {model_comparison['tracker']['f1_pct']}% | "
              f"AUC: {model_comparison['tracker']['auc']}")
        print(f"        F1 improvement: {comp['f1_improvement_pct']:+.1f}% — "
              f"Tracker {'outperforms' if comp['tracker_is_better'] else 'does not outperform'} baseline")
    else:
        print(f"        Skipped: {model_comparison['error']}")

    # ── Robustness & sample size ──
    step_robust = 8 if has_real else 6
    print(f"  [{step_robust}/{n_steps}] Robustness & sample size analysis...")
    sessions, _ = generate_ground_truth(n_sessions=metadata["n_sessions"])
    robustness_df = run_robustness_analysis(sessions, gt_df)
    sample_size_df = run_sample_size_analysis(gt_df, sessions)
    print(f"        Tested {len(robustness_df)} noise levels, {len(sample_size_df)} sample sizes")

    # ── Visualizations ──
    print("\n  Generating validation plots...")
    plot_validation_summary(phase_val, nurse_val, cause_val, qual_val,
                            robustness_df, sample_size_df)
    plot_phase_comparison(gt_df, tracker_df)
    if has_real:
        plot_real_vs_simulated(real_df, gt_df, real_vs_sim, outlier_cls)
    if model_comparison is not None and "error" not in model_comparison:
        plot_model_comparison(model_comparison)

    # ── Save results ──
    verdicts = {
        "phase_detection": "PASS" if phase_val["top3_overlap"] >= 2 else "MARGINAL" if phase_val["top3_overlap"] >= 1 else "FAIL",
        "nurse_detection": "PASS" if nurse_val["sensitivity"] >= 50 else "FAIL",
        "cause_attribution": "PASS" if cause_val["accuracy_tolerant_pct"] >= 66 else "MARGINAL" if cause_val["accuracy_tolerant_pct"] >= 33 else "FAIL",
        "qualitative_coding": "PASS" if qual_val["accuracy_pct"] >= 70 else "MARGINAL" if qual_val["accuracy_pct"] >= 50 else "FAIL",
        "min_sessions_for_reliability": int(sample_size_df[sample_size_df["nurse_sensitivity_pct"] >= 100]["n_sessions"].min())
            if (sample_size_df["nurse_sensitivity_pct"] >= 100).any() else ">150",
    }
    if real_vs_sim is not None:
        verdicts["real_vs_sim_distribution"] = real_vs_sim["verdict"]
    if outlier_cls is not None:
        verdicts["outlier_classification"] = outlier_cls["verdict"]
    if model_comparison is not None and "error" not in model_comparison:
        verdicts["tracker_vs_baseline"] = model_comparison["comparison"]["verdict"]

    validation_report = {
        "phase_variability": phase_val,
        "nurse_effects": nurse_val,
        "delay_causes": cause_val,
        "qualitative_coding": qual_val,
        "robustness": robustness_df.to_dict("records"),
        "sample_size": sample_size_df.to_dict("records"),
        "overall_verdict": verdicts,
    }
    if real_vs_sim is not None:
        validation_report["real_vs_simulated"] = real_vs_sim
    if outlier_cls is not None:
        validation_report["outlier_classification"] = outlier_cls
    if model_comparison is not None:
        validation_report["baseline_vs_tracker"] = model_comparison

    # Save CSVs
    phase_stats.to_csv(SIM_DIR / "analysis_phase_variability.csv")
    nurse_results.to_csv(SIM_DIR / "analysis_nurse_performance.csv", index=False)
    cause_results.to_csv(SIM_DIR / "analysis_delay_causes.csv", index=False)
    coded_df.to_csv(SIM_DIR / "analysis_qualitative_coding.csv", index=False)
    robustness_df.to_csv(SIM_DIR / "robustness_results.csv", index=False)
    sample_size_df.to_csv(SIM_DIR / "sample_size_results.csv", index=False)

    with open(SIM_DIR / "validation_report.json", "w") as f:
        json.dump(validation_report, f, indent=2, default=str)
    print("  Saved: validation_report.json + 6 analysis CSVs")

    # Print summary
    print("\n" + "─" * 50)
    print("  VALIDATION VERDICTS:")
    for key, verdict in validation_report["overall_verdict"].items():
        if verdict == "PASS":
            icon = "✓"
        elif verdict == "MARGINAL":
            icon = "~"
        elif verdict == "FAIL":
            icon = "✗"
        else:
            icon = "i"  # informational (e.g., sample size)
        print(f"    [{icon}] {key}: {verdict}")
    print("─" * 50)

    return validation_report
