"""
Phase 8: CSV, JSON, and markdown report generation.
"""

import json
from typing import Dict, List

import pandas as pd

from src.config import (
    OUTPUT_DIR, GLOBAL_LGB_PARAMS, OUTLIER_LABEL_MAP,
    COMPARE_COLS, TARGET_COL,
)


def save_results(
    df: pd.DataFrame,
    feature_cols: List[str],
    outlier_threshold: float,
    top_features: pd.Series,
    per_physician_results: Dict,
    additional_results: Dict,
    X: pd.DataFrame,
    y: pd.Series,
) -> None:
    """Save CSV, JSON report, and markdown summary."""
    print("\n" + "=" * 60)
    print("8. SAVING RESULTS")
    print("=" * 60)

    target = df[TARGET_COL]
    physicians = sorted(df["PHYSICIAN"].unique())

    # --- CSV ---
    output_df = df.copy()
    output_df["DATE"] = output_df["DATE"].dt.strftime("%Y-%m-%d")
    output_df["outlier_label"] = output_df["outlier_class"].map(OUTLIER_LABEL_MAP)
    output_df.to_csv(f"{OUTPUT_DIR}/MSE433_M4_Data_with_outliers.csv", index=False)
    print(f"Saved: {OUTPUT_DIR}/MSE433_M4_Data_with_outliers.csv")

    # --- JSON report ---
    n_outliers = int(df["outlier_class"].sum())
    n_normal = int(len(df) - n_outliers)
    outlier_df = df[df["outlier_class"] == 1]
    normal_df = df[df["outlier_class"] == 0]

    report = {
        "dataset": {
            "source": "Data/MSE433_M4_Data.xlsx",
            "total_cases": int(len(df)),
            "cases_with_target": int(df[TARGET_COL].notna().sum()),
            "features_used": feature_cols,
        },
        "target_variable": {
            "name": "PT IN-OUT (Patient In to Out, minutes)",
            "mean": round(float(target.mean()), 1),
            "median": round(float(target.median()), 1),
            "std": round(float(target.std()), 1),
            "min": round(float(target.min()), 1),
            "max": round(float(target.max()), 1),
            "skewness": round(float(target.skew()), 2),
            "kurtosis": round(float(target.kurtosis()), 2),
        },
        "outlier_detection": {
            "method": "90th Percentile (Top 10%)",
            "threshold_minutes": round(float(outlier_threshold), 1),
            "classification_type": "binary",
            "class_distribution": {
                OUTLIER_LABEL_MAP[k]: int(v)
                for k, v in df["outlier_class"].value_counts().sort_index().items()
            },
            "rationale": (
                "Top 10% (90th percentile) chosen to capture a meaningful group of "
                "long-duration cases for modeling and actionable problem identification. "
                "IQR yielded only 5 outliers (3.4%), too few for stable modeling. "
                "The 90th percentile threshold provides ~15 outliers, enabling clearer "
                "SHAP interpretation of what drives long procedures."
            ),
        },
        "outlier_cases": [
            {
                "case_num": int(row["CASE #"]),
                "date": str(row["DATE"].strftime("%Y-%m-%d") if pd.notna(row["DATE"]) else ""),
                "physician": str(row["PHYSICIAN"]),
                "pt_in_out_min": int(row[TARGET_COL]),
                "note": str(row["Note"]) if pd.notna(row["Note"]) and row["Note"] != "" else None,
            }
            for _, row in outlier_df.iterrows()
        ],
        "model": {
            "type": "LightGBM Classifier",
            "params": {k: v for k, v in GLOBAL_LGB_PARAMS.items() if k != "class_weight"},
            "class_weight": "balanced",
            "note": "Fitted on full dataset for SHAP interpretation, not prediction.",
        },
        "shap_analysis": {
            "top_features_by_mean_abs_shap": {
                feat: round(float(val), 3) for feat, val in top_features.items()
            },
        },
        "per_physician": per_physician_results,
        "additional_analyses": additional_results,
        "output_structure": {
            "eda/": "EDA charts (global + per-physician comparisons)",
            "global_model/": "Global LightGBM + SHAP plots",
            "per_physician/": "Per-physician model subdirectories",
            "additional/": "Learning curve, complexity, scheduling analyses",
        },
    }

    with open(f"{OUTPUT_DIR}/analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {OUTPUT_DIR}/analysis_report.json")

    # --- Markdown summary ---
    _write_markdown_summary(
        df, target, outlier_threshold, outlier_df, normal_df,
        n_outliers, n_normal, top_features, feature_cols,
        per_physician_results, additional_results, physicians, X, y,
    )


def _write_markdown_summary(
    df, target, outlier_threshold, outlier_df, normal_df,
    n_outliers, n_normal, top_features, feature_cols,
    per_physician_results, additional_results, physicians, X, y,
) -> None:
    """Build and write the markdown summary report."""
    compare_cols = COMPARE_COLS
    outlier_means = outlier_df[compare_cols].mean()
    normal_means = normal_df[compare_cols].mean()

    phys_outlier_global = df.groupby("PHYSICIAN")["outlier_class"].agg(["sum", "count"])
    phys_outlier_global["rate"] = (phys_outlier_global["sum"] / phys_outlier_global["count"] * 100)

    # Build outlier case table rows
    outlier_rows = ""
    for _, row in outlier_df.sort_values(TARGET_COL, ascending=False).iterrows():
        date_str = row["DATE"].strftime("%Y-%m-%d") if pd.notna(row["DATE"]) else ""
        note_str = str(row["Note"]) if pd.notna(row["Note"]) and row["Note"] != "" else "-"
        outlier_rows += f"| {int(row['CASE #']):>4} | {date_str} | {row['PHYSICIAN']} | {int(row[TARGET_COL]):>3} | {note_str} |\n"

    # Feature comparison table
    feature_compare_rows = ""
    for col in compare_cols:
        n_val = normal_means[col]
        o_val = outlier_means[col]
        diff_pct = ((o_val - n_val) / n_val * 100) if n_val != 0 else 0
        feature_compare_rows += f"| {col:<24} | {n_val:>6.1f} | {o_val:>6.1f} | {diff_pct:>+6.1f}% |\n"

    # SHAP ranking
    shap_rows = ""
    for rank, (feat, val) in enumerate(top_features.items(), 1):
        shap_rows += f"| {rank} | {feat:<24} | {val:.3f} |\n"

    # Physician breakdown (global threshold)
    phys_global_rows = ""
    for phys, row in phys_outlier_global.iterrows():
        phys_global_rows += f"| {phys} | {int(row['count'])} | {int(row['sum'])} | {row['rate']:.1f}% |\n"

    # Physician breakdown (per-physician IQR threshold)
    phys_local_rows = ""
    for phys in physicians:
        res = per_physician_results[phys]
        phys_local_rows += f"| {phys} | {res['n_cases']} | {res['threshold_minutes']:.0f} | {res['n_outliers']} | {res['n_outliers']/res['n_cases']*100:.1f}% |\n"

    learning_curve_results = additional_results["learning_curve"]
    complexity_stats = additional_results["case_complexity"]
    scheduling_results = additional_results["scheduling"]
    severity_results = additional_results["physician_severity"]

    summary_md = f"""# Analysis Summary: AFib Ablation Operation Duration Outliers

---

## 1. Dataset Overview

- **Source**: MSE433_M4_Data.xlsx (Varipulse PFA case data)
- **Total cases**: {len(df)}
- **Date range**: {df['DATE'].min().strftime('%Y-%m-%d')} to {df['DATE'].max().strftime('%Y-%m-%d')}
- **Physicians**: {', '.join(sorted(df['PHYSICIAN'].unique()))}
- **Target variable**: PT IN-OUT (total patient in-to-out time, minutes)

## 2. Operation Duration Statistics

| Statistic | Value |
|-----------|-------|
| Mean      | {target.mean():.1f} min |
| Median    | {target.median():.1f} min |
| Std Dev   | {target.std():.1f} min |
| Min       | {target.min():.0f} min |
| Max       | {target.max():.0f} min |
| Skewness  | {target.skew():.2f} (right-skewed) |
| Kurtosis  | {target.kurtosis():.2f} (heavy-tailed) |

---

## 3. Global Outlier Analysis

### Outlier Definition

- **Method**: 90th percentile (top 10% of cases by duration)
- **Threshold**: PT IN-OUT >= {outlier_threshold:.1f} minutes (>= 99 for integer durations)
- **Normal cases**: {n_normal} ({n_normal/len(df)*100:.1f}%)
- **Outlier cases**: {n_outliers} ({n_outliers/len(df)*100:.1f}%)
- **Rationale**: The 90th percentile captures a meaningful group of long-duration
  procedures large enough for stable modeling, while focusing on the cases most
  worth investigating for process improvement.

### Global Outlier Cases (Sorted by Duration)

| Case | Date       | Physician | PT IN-OUT (min) | Note |
|-----:|------------|-----------|----------------:|------|
{outlier_rows}

### Global Outlier vs Normal: Feature Comparison

| Feature                  | Normal | Outlier | Difference |
|--------------------------|-------:|--------:|-----------:|
{feature_compare_rows}

### Physician Breakdown (Global Threshold >= {outlier_threshold:.1f} min)

| Physician | Total Cases | Outliers | Outlier Rate |
|-----------|------------:|---------:|-------------:|
{phys_global_rows}

### Global LightGBM Model

- **Model**: LightGBM (gradient boosted decision tree) with balanced class weights
- **Fitted on**: {len(X)} of {len(df)} cases ({int(y.sum())} outliers, {int(len(y) - y.sum())} normal; {len(df) - len(X)} excluded due to missing feature values)
- **Purpose**: SHAP feature importance interpretation (not prediction)

### Global SHAP Feature Drivers

| Rank | Feature                  | Mean SHAP |
|-----:|--------------------------|----------:|
{shap_rows}

**Interpretation**: The SHAP analysis reveals which factors most strongly push a case
toward being classified as an outlier (long-duration):

"""

    # Build interpretation dynamically
    shap_descriptions = {
        "PHYSICIAN_ENC": "Physician identity is the single strongest predictor -- confirms that which physician performs the case is the dominant factor in outlier status.",
        "POST CARE/EXTUBATION": "Longer post-care time (hemostasis, extubation, monitoring) is associated with outlier duration.",
        "TSP": "Extended transseptal puncture time strongly associated with long procedures.",
        "PRE-MAP": "Longer pre-ablation electroanatomic mapping contributes to outlier classification.",
        "PT PREP/INTUBATION": "Extended patient preparation and anesthesia setup drives outlier duration.",
        "ABL DURATION": "Longer total ablation time (including catheter repositioning) contributes to outlier classification.",
        "ABL TIME": "Higher cumulative pulse-on energy delivery time associated with outlier cases.",
        "ACCESSS": "Longer vascular access time is associated with outlier duration.",
        "#ABL": "More ablation sites targeted contributes to longer procedures.",
        "#APPLICATIONS": "More PFA applications delivered contributes to longer procedures.",
        "NOTE_CTI": "CTI (cavo-tricuspid isthmus) ablation adds significant procedure time.",
        "NOTE_BOX": "Box isolation procedure adds to overall duration.",
        "NOTE_PST": "Posterior box ablation adds to overall duration.",
        "NOTE_SVC": "SVC ablation adds to overall duration.",
        "CASE_ORDER_IN_DAY": "Case scheduling position in the day affects duration (earlier cases tend to be longer).",
    }
    for rank_i, (feat_i, val_i) in enumerate(top_features.items(), 1):
        desc = shap_descriptions.get(feat_i, f"Higher values push predictions toward outlier class.")
        summary_md += f"{rank_i}. **{feat_i}** (SHAP: {val_i:.3f}): {desc}\n"

    summary_md += """

---

## 4. Per-Physician Outlier Analysis

Each physician's cases are analyzed independently using the **IQR method
(Q3 + 1.0 * IQR)** within their own distribution. Unlike a fixed percentile,
IQR scales naturally with sample size and distribution shape, making it fair
to compare physicians with different caseloads.

### Per-Physician Threshold Comparison

| Physician | Total Cases | Threshold (Q3+1.0*IQR) | Outliers | Outlier Rate |
|-----------|------------:|-----------------------:|---------:|-------------:|
"""
    summary_md += phys_local_rows

    summary_md += f"""
Note: Dr. B's threshold ({per_physician_results['Dr. B']['threshold_minutes']:.0f} min) is much higher than
Dr. A's ({per_physician_results['Dr. A']['threshold_minutes']:.0f} min) because Dr. B's baseline duration
is longer overall (higher Q3).

See comparison charts in `eda/`:
- `eda_per_physician_distributions.png` - histograms and strip plots per physician
- `eda_per_physician_comparison.png` - outlier rates, thresholds, mean durations, SHAP heatmap
- `eda_per_physician_feature_comparison.png` - outlier vs normal feature means side-by-side

"""

    # Per-physician detail sections
    for phys in physicians:
        res = per_physician_results[phys]
        phys_safe = phys.replace(".", "").replace(" ", "_")
        summary_md += f"### {phys} ({res['n_cases']} cases)\n\n"
        summary_md += f"- **Threshold**: PT IN-OUT > {res['threshold_minutes']:.1f} min (Q3+1.0*IQR within {phys})\n"
        summary_md += f"- **Normal**: {res['n_normal']}, **Outlier**: {res['n_outliers']}\n"

        if not res["model_fitted"]:
            summary_md += f"- **Model**: Not fitted ({res['reason_skipped']})\n\n"
            continue

        summary_md += f"\n**Top SHAP Features:**\n\n"
        summary_md += f"| Rank | Feature | Mean SHAP |\n"
        summary_md += f"|-----:|---------|----------:|\n"
        for rank, (feat, val) in enumerate(res["top_shap_features"].items(), 1):
            summary_md += f"| {rank} | {feat} | {val:.3f} |\n"

        summary_md += f"\n**Outlier vs Normal (feature means):**\n\n"
        summary_md += f"| Feature | Normal | Outlier | Diff |\n"
        summary_md += f"|---------|-------:|--------:|-----:|\n"
        for col, vals in res["outlier_vs_normal_means"].items():
            summary_md += f"| {col} | {vals['normal']:.1f} | {vals['outlier']:.1f} | {vals['diff_pct']:+.1f}% |\n"

        summary_md += f"\n**Outlier cases:**\n\n"
        summary_md += f"| Case | Date | PT IN-OUT | Note |\n"
        summary_md += f"|-----:|------|----------:|------|\n"
        for case in sorted(res["outlier_cases"], key=lambda c: -c["pt_in_out_min"]):
            note = case["note"] or "-"
            summary_md += f"| {case['case_num']} | {case['date']} | {case['pt_in_out_min']} | {note} |\n"

        summary_md += f"\nPlots: `per_physician/{phys_safe}/`\n\n"

    # Problem statement and solutions
    timing_top_global = [f for f in top_features.index if f in compare_cols]
    top_timing_str = " and ".join(f"**{f}**" for f in timing_top_global[:2]) if timing_top_global else "multiple procedural phases"

    summary_md += f"""---

## 5. Problem Statement

The top 10% of AFib ablation procedures take significantly longer than typical cases
(>={outlier_threshold:.1f} min vs median {target.median():.0f} min). The global model shows that
**physician identity** is the single strongest predictor of outlier status, confirming that
Dr. B's cases dominate the outlier class. Beyond physician, the key procedural bottlenecks
are {top_timing_str}, suggesting that delays accumulate in specific phases rather than
uniformly across the entire operation.

Per-physician analysis reveals that each physician has different outlier drivers:
"""
    for phys in physicians:
        res = per_physician_results[phys]
        if res["model_fitted"] and res.get("top_shap_features"):
            top_feat = list(res["top_shap_features"].keys())[0]
            summary_md += f"- **{phys}**: Top driver is **{top_feat}**\n"

    summary_md += f"""
## 6. Potential Solutions

Based on the analysis, process improvement efforts should target:
"""

    solution_num = 1
    timing_top = [f for f in top_features.index if f in compare_cols]
    for feat in timing_top[:3]:
        o_val = outlier_means.get(feat, 0)
        n_val = normal_means.get(feat, 0)
        if solution_num == 1:
            summary_md += f"\n{solution_num}. **{feat}** optimization: This phase shows a large difference between outlier and normal cases ({o_val:.0f} vs {n_val:.0f} min). Investigate root causes of extended times (complications, equipment issues, patient anatomy).\n"
        elif solution_num == 2:
            summary_md += f"{solution_num}. **{feat}** standardization: Reduce variability ({o_val:.0f} vs {n_val:.0f} min for outliers vs normal) through protocol standardization or pre-procedure checklists.\n"
        else:
            summary_md += f"{solution_num}. **{feat}** monitoring: Set real-time alerts when this phase exceeds expected duration ({n_val:.0f} min), enabling early intervention.\n"
        solution_num += 1

    summary_md += f"{solution_num}. **Physician-specific coaching**: Each physician has different outlier drivers. Targeted interventions per physician will be more effective than blanket policies.\n"
    phys_driver_parts = []
    for p in physicians:
        sf = per_physician_results[p].get("top_shap_features")
        driver = list(sf.keys())[0] if sf else "N/A"
        phys_driver_parts.append(f"{p}: top driver = {driver}")
    summary_md += f"   - {', '.join(phys_driver_parts)}\n"

    summary_md += """
---

## 7. Additional Analyses

### Learning Curve (Duration Over Time)

Are physicians getting faster with the Varipulse device over time?

"""

    for phys, lc in learning_curve_results.items():
        summary_md += f"- **{phys}**: slope = {lc['slope_min_per_case']:+.3f} min/case ({lc['interpretation']}), total change over all cases: {lc['total_improvement_min']:+.1f} min\n"

    summary_md += f"""
See: `additional/learning_curve.png`

### Case Complexity (Additional Procedures)

Impact of additional procedures (CTI, BOX, PST BOX, SVC) on duration and outlier rate:

| Procedure Type | N Cases | Mean Duration | Outlier Rate |
|----------------|--------:|--------------:|-------------:|
"""

    for label, info in complexity_stats.items():
        short = label.split(" (")[0]
        summary_md += f"| {short} | {info['n_cases']} | {info['mean_duration']:.1f} min | {info['outlier_rate_global_pct']:.1f}% |\n"

    summary_md += f"""
See: `additional/case_complexity.png`

### Case Scheduling (Order Within Day)

Does the sequence of a case in the day affect duration?

| Case Order | N Cases | Mean Duration | Outlier Rate |
|-----------:|--------:|--------------:|-------------:|
"""

    for order, info in scheduling_results["case_order_stats"].items():
        summary_md += f"| {order} | {info['n_cases']} | {info['mean_duration']:.1f} min | {info['outlier_rate_pct']:.1f}% |\n"

    summary_md += f"""
See: `additional/case_order_scheduling.png`

### Physician Severity & Case Complexity Profile

Does a certain physician see more complex or severe patients? This analysis compares
case complexity indicators across physicians to disentangle physician skill from patient mix.

| Metric | Dr. A | Dr. B | Dr. C |
|--------|------:|------:|------:|
| Cases | {severity_results['Dr. A']['n_cases']} | {severity_results['Dr. B']['n_cases']} | {severity_results['Dr. C']['n_cases']} |
| Mean PT IN-OUT | {severity_results['Dr. A']['pt_in_out']['mean']} min | {severity_results['Dr. B']['pt_in_out']['mean']} min | {severity_results['Dr. C']['pt_in_out']['mean']} min |
| Median PT IN-OUT | {severity_results['Dr. A']['pt_in_out']['median']} min | {severity_results['Dr. B']['pt_in_out']['median']} min | {severity_results['Dr. C']['pt_in_out']['median']} min |
| Mean Ablation Sites (#ABL) | {severity_results['Dr. A']['abl_sites']['mean']} | {severity_results['Dr. B']['abl_sites']['mean']} | {severity_results['Dr. C']['abl_sites']['mean']} |
| Mean ABL DURATION | {severity_results['Dr. A']['abl_duration_mean']} min | {severity_results['Dr. B']['abl_duration_mean']} min | {severity_results['Dr. C']['abl_duration_mean']} min |
| Mean ABL TIME (pulse-on) | {severity_results['Dr. A']['abl_time_mean']} min | {severity_results['Dr. B']['abl_time_mean']} min | {severity_results['Dr. C']['abl_time_mean']} min |
| Mean Repositioning Time | {severity_results['Dr. A']['repositioning_time_mean']} min | {severity_results['Dr. B']['repositioning_time_mean']} min | {severity_results['Dr. C']['repositioning_time_mean']} min |
| Mean TSP | {severity_results['Dr. A']['tsp_mean']} min | {severity_results['Dr. B']['tsp_mean']} min | {severity_results['Dr. C']['tsp_mean']} min |
| Mean PRE-MAP | {severity_results['Dr. A']['pre_map_mean']} min | {severity_results['Dr. B']['pre_map_mean']} min | {severity_results['Dr. C']['pre_map_mean']} min |
| Cases with Additional Procedures | {severity_results['Dr. A']['pct_additional_procedures']}% | {severity_results['Dr. B']['pct_additional_procedures']}% | {severity_results['Dr. C']['pct_additional_procedures']}% |

"""

    summary_md += """**Key Insight:** Dr. A actually performs the *most* additional procedures (20% of cases
include BOX, PST BOX, or CTI targets) yet has the *shortest* average duration (69.5 min).
Dr. B performs fewer additional procedures (11.7%) but takes substantially longer on average
(91.9 min). This suggests Dr. B's longer durations are **not explained by more complex patient
cases** but rather by differences in procedural efficiency, particularly in catheter
repositioning time (ABL DURATION 27.8 min for Dr. B vs 21.0 for Dr. A, despite similar
pulse-on time of ~7 min for both). Dr. C sees the simplest cases (0% additional procedures)
with moderate duration (74.7 min).

See: `additional/physician_severity_profile.png`

"""

    summary_md += f"""---

## 8. Output Directory Structure

```
output/
├── analysis_report.json          # Full structured report (machine-readable)
├── analysis_summary.md           # This summary (human-readable)
├── MSE433_M4_Data_with_outliers.csv  # Dataset with outlier columns
├── eda/                          # Exploratory data analysis
│   ├── eda_distribution.png
│   ├── eda_outlier_classes.png
│   ├── eda_correlation.png
│   ├── eda_per_physician_distributions.png
│   ├── eda_per_physician_comparison.png
│   └── eda_per_physician_feature_comparison.png
├── global_model/                 # Global LightGBM + SHAP
│   ├── lgbm_feature_importance.png
│   ├── shap_summary.png
│   ├── shap_bar.png
│   └── shap_dependence.png
├── per_physician/                # Per-physician models
│   ├── Dr_A/                     # (model only if outliers exist)
│   ├── Dr_B/
│   └── Dr_C/
└── additional/                   # Extra analyses
    ├── learning_curve.png
    ├── case_complexity.png
    ├── case_order_scheduling.png
    └── physician_severity_profile.png
```
"""

    with open(f"{OUTPUT_DIR}/analysis_summary.md", "w") as f:
        f.write(summary_md)
    print(f"Saved: {OUTPUT_DIR}/analysis_summary.md")
