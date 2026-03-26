"""
Phase 7: Additional analyses - learning curves, complexity, severity profiles,
prep time variability, and repositioning efficiency.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from src.config import EXTRA_DIR, TARGET_COL
from src.viz import PHYS_COLORS, SAVE_DPI


def run_additional_analyses(
    df: pd.DataFrame,
    per_physician_results: Dict,
) -> Dict:
    """Run additional analyses and return results dict with keys:
    learning_curve, case_complexity, scheduling, physician_severity.
    """
    print("\n" + "=" * 60)
    print("7. ADDITIONAL ANALYSES")
    print("=" * 60)

    physicians = sorted(df["PHYSICIAN"].unique())

    learning_curve_results = _learning_curve(df, physicians)
    complexity_stats = _case_complexity(df)
    severity_results = _physician_severity(df, physicians)
    prep_time_results = _prep_time_variability(df, physicians)
    repo_results = _repositioning_efficiency(df, physicians)

    return {
        "learning_curve": learning_curve_results,
        "case_complexity": complexity_stats,
        "physician_severity": severity_results,
        "prep_time": prep_time_results,
        "repositioning": repo_results,
    }


def _learning_curve(df: pd.DataFrame, physicians: list) -> Dict:
    """7a. Learning curve / time trend."""
    print("\n--- 7a. Learning Curve (Duration Over Time) ---")
    fig, axes = plt.subplots(1, len(physicians), figsize=(7 * len(physicians), 5))
    fig.suptitle("Learning Curve: Are Procedures Getting Faster Over Time?", fontsize=14, fontweight="bold")
    if len(physicians) == 1:
        axes = [axes]

    learning_curve_results = {}
    for idx, phys in enumerate(physicians):
        ax = axes[idx]
        phys_data = df[df["PHYSICIAN"] == phys].sort_values("DATE").reset_index(drop=True)
        case_seq = np.arange(1, len(phys_data) + 1)
        durations = phys_data[TARGET_COL].values

        ax.scatter(case_seq, durations, alpha=0.5, color=PHYS_COLORS.get(phys, "gray"),
                   edgecolor="black", s=30)

        mask = ~np.isnan(durations)
        if mask.sum() > 2:
            z = np.polyfit(case_seq[mask], durations[mask], 1)
            p = np.poly1d(z)
            ax.plot(case_seq, p(case_seq), color="red", linewidth=2,
                    label=f"Linear Trend ({z[0]:+.2f} min/case)")
            if len(durations) >= 10:
                rolling = pd.Series(durations).rolling(10, min_periods=5).mean()
                ax.plot(case_seq, rolling, color="darkblue", linewidth=1.5, alpha=0.7,
                        linestyle="--", label="10-Case Moving Average")
            learning_curve_results[phys] = {
                "slope_min_per_case": round(float(z[0]), 3),
                "total_improvement_min": round(float(z[0] * len(phys_data)), 1),
                "interpretation": "improving" if z[0] < -0.05 else "stable" if abs(z[0]) <= 0.05 else "worsening",
            }
            print(f"  {phys}: slope = {z[0]:+.3f} min/case ({learning_curve_results[phys]['interpretation']})")

        ax.set_title(f"{phys}: Duration Trend (n={len(phys_data)})")
        ax.set_xlabel("Case Number (Chronological Order)")
        ax.set_ylabel("Total Procedure Duration (minutes)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{EXTRA_DIR}/learning_curve.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print("Saved: learning_curve.png")

    return learning_curve_results


def _case_complexity(df: pd.DataFrame) -> Dict:
    """7b. Case complexity breakdown."""
    print("\n--- 7b. Case Complexity Analysis ---")

    # Use negative lookbehind so "BOX" row excludes "PST BOX" (avoids double-counting)
    box_only = df["Note"].astype(str).str.contains(r"(?<!PST )BOX", na=False) & (df["NOTE_PST"] == 0)
    complexity_flags = {
        "Standard PFA": ~df["HAS_NOTE"] | (df["Note"].astype(str).str.strip() == ""),
        "CTI (Cavo-tricuspid isthmus)": df["NOTE_CTI"] == 1,
        "BOX (Box isolation, excl. PST)": box_only,
        "PST BOX (Posterior box)": df["NOTE_PST"] == 1,
        "SVC (Superior vena cava)": df["NOTE_SVC"] == 1,
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Case Complexity: How Additional Procedures Affect Duration", fontsize=14, fontweight="bold")

    complexity_data = []
    complexity_labels = []
    complexity_stats = {}
    for label, mask in complexity_flags.items():
        vals = df.loc[mask, TARGET_COL].dropna()
        if len(vals) > 0:
            complexity_data.append(vals.values)
            short_label = label.split(" (")[0]
            complexity_labels.append(f"{short_label}\n(n={len(vals)})")
            outlier_rate = df.loc[mask, "outlier_class"].mean() * 100
            complexity_stats[label] = {
                "n_cases": int(len(vals)),
                "mean_duration": round(float(vals.mean()), 1),
                "median_duration": round(float(vals.median()), 1),
                "outlier_rate_global_pct": round(float(outlier_rate), 1),
            }
            print(f"  {label}: n={len(vals)}, mean={vals.mean():.1f} min, outlier rate={outlier_rate:.1f}%")

    bp = axes[0].boxplot(complexity_data, labels=complexity_labels, patch_artist=True, vert=True)
    colors_bp = ["#2196F3", "#FF9800", "#E91E63", "#9C27B0", "#4CAF50"]
    for patch, color in zip(bp["boxes"], colors_bp[:len(bp["boxes"])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    axes[0].set_ylabel("Total Procedure Duration (minutes)")
    axes[0].set_title("Duration Distribution by Procedure Type")
    axes[0].tick_params(axis="x", labelsize=9)

    comp_names = []
    comp_outlier_rates = []
    for label, info in complexity_stats.items():
        comp_names.append(label.split(" (")[0])
        comp_outlier_rates.append(info["outlier_rate_global_pct"])

    bars = axes[1].bar(range(len(comp_names)), comp_outlier_rates,
                       color=colors_bp[:len(comp_names)], alpha=0.7, edgecolor="black")
    for i, rate in enumerate(comp_outlier_rates):
        axes[1].text(i, rate + 0.5, f"{rate:.1f}%", ha="center", fontweight="bold")
    axes[1].set_xticks(range(len(comp_names)))
    axes[1].set_xticklabels(comp_names, fontsize=9, rotation=15, ha="right")
    axes[1].set_ylabel("Outlier Rate (%)")
    axes[1].set_title("Outlier Rate by Procedure Type")

    plt.tight_layout()
    plt.savefig(f"{EXTRA_DIR}/case_complexity.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print("Saved: case_complexity.png")

    return complexity_stats


def _physician_severity(df: pd.DataFrame, physicians: list) -> Dict:
    """7d. Physician severity & case complexity profile."""
    print("\n--- 7d. Physician Severity & Case Complexity Profile ---")

    df["NOTE_AAFL"] = df["Note"].astype(str).str.contains("AAFL", na=False).astype(int)
    df["NOTE_TROUBLESHOOT"] = df["Note"].astype(str).str.contains("TROUBLESHOOT", na=False).astype(int)
    df["HAS_ADDITIONAL"] = (df["NOTE_CTI"] | df["NOTE_BOX"] | df["NOTE_PST"] | df["NOTE_SVC"] | df["NOTE_AAFL"]).astype(int)
    df["N_ADDITIONAL"] = df["NOTE_CTI"] + df["NOTE_BOX"] + df["NOTE_PST"] + df["NOTE_SVC"] + df["NOTE_AAFL"]

    severity_results = {}
    for phys in physicians:
        sub = df[df["PHYSICIAN"] == phys]
        severity_results[phys] = {
            "n_cases": int(len(sub)),
            "pt_in_out": {"mean": round(float(sub[TARGET_COL].mean()), 1), "median": round(float(sub[TARGET_COL].median()), 1), "std": round(float(sub[TARGET_COL].std()), 1)},
            "abl_sites": {"mean": round(float(sub["#ABL"].mean()), 1), "median": round(float(sub["#ABL"].median()), 1)},
            "applications": round(float(sub["#APPLICATIONS"].mean()), 1),
            "abl_duration_mean": round(float(sub["ABL DURATION"].mean()), 1),
            "abl_time_mean": round(float(sub["ABL TIME"].mean()), 1),
            "repositioning_time_mean": round(float((sub["ABL DURATION"] - sub["ABL TIME"]).mean()), 1),
            "pre_map_mean": round(float(sub["PRE-MAP"].mean()), 1),
            "tsp_mean": round(float(sub["TSP"].mean()), 1),
            "pct_additional_procedures": round(float(sub["HAS_ADDITIONAL"].mean() * 100), 1),
            "additional_breakdown": {
                "CTI": int(sub["NOTE_CTI"].sum()),
                "BOX": int(sub["NOTE_BOX"].sum()),
                "PST_BOX": int(sub["NOTE_PST"].sum()),
                "SVC": int(sub["NOTE_SVC"].sum()),
                "AAFL": int(sub["NOTE_AAFL"].sum()),
            },
            "cases_2plus_additional": int((sub["N_ADDITIONAL"] >= 2).sum()),
            "troubleshoot_cases": int(sub["NOTE_TROUBLESHOOT"].sum()),
        }
        s = severity_results[phys]
        print(f"  {phys}: mean PT IN-OUT={s['pt_in_out']['mean']} min, "
              f"mean #ABL={s['abl_sites']['mean']}, "
              f"additional procedures={s['pct_additional_procedures']}%, "
              f"repositioning time={s['repositioning_time_mean']} min")

    # Figure: Physician severity comparison (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Physician Case Severity & Complexity Profile Comparison", fontsize=15, fontweight="bold")

    x_pos = np.arange(len(physicians))
    bar_colors = [PHYS_COLORS.get(p, "gray") for p in physicians]

    # (0,0) Duration distribution
    ax = axes[0, 0]
    phys_data_list = [df[df["PHYSICIAN"] == p][TARGET_COL].dropna().values for p in physicians]
    bp = ax.boxplot(phys_data_list, labels=physicians, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], bar_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel("Duration (minutes)")
    ax.set_title("Total Procedure Duration Distribution")

    # (0,1) Ablation sites
    ax = axes[0, 1]
    abl_means = [severity_results[p]["abl_sites"]["mean"] for p in physicians]
    abl_medians = [severity_results[p]["abl_sites"]["median"] for p in physicians]
    ax.bar(x_pos - 0.15, abl_means, 0.3, label="Mean", color=bar_colors, alpha=0.7, edgecolor="black")
    ax.bar(x_pos + 0.15, abl_medians, 0.3, label="Median", color=bar_colors, alpha=0.4, edgecolor="black", hatch="//")
    for i in range(len(physicians)):
        ax.text(i - 0.15, abl_means[i] + 0.2, f"{abl_means[i]:.1f}", ha="center", fontsize=9)
        ax.text(i + 0.15, abl_medians[i] + 0.2, f"{abl_medians[i]:.0f}", ha="center", fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(physicians)
    ax.set_ylabel("Number of Ablation Sites")
    ax.set_title("Ablation Sites Targeted per Case")
    ax.legend()

    # (0,2) ABL DURATION vs ABL TIME
    ax = axes[0, 2]
    abl_dur = [severity_results[p]["abl_duration_mean"] for p in physicians]
    abl_time = [severity_results[p]["abl_time_mean"] for p in physicians]
    repo_time = [severity_results[p]["repositioning_time_mean"] for p in physicians]
    ax.bar(x_pos, abl_time, 0.5, label="Active Pulse-On Time", color=bar_colors, alpha=0.7, edgecolor="black")
    ax.bar(x_pos, repo_time, 0.5, bottom=abl_time, label="Catheter Repositioning Time",
           color=bar_colors, alpha=0.3, edgecolor="black", hatch="xx")
    for i in range(len(physicians)):
        ax.text(i, abl_dur[i] + 0.5, f"{abl_dur[i]:.0f} min total", ha="center", fontsize=9, fontweight="bold")
        ax.text(i, abl_time[i] / 2, f"{abl_time[i]:.1f}", ha="center", fontsize=8, color="white", fontweight="bold")
        ax.text(i, abl_time[i] + repo_time[i] / 2, f"{repo_time[i]:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(physicians)
    ax.set_ylabel("Duration (minutes)")
    ax.set_title("Ablation Time Breakdown: Pulse-On vs Repositioning")
    ax.legend(fontsize=8)

    # (1,0) PRE-MAP and TSP
    ax = axes[1, 0]
    pre_map = [severity_results[p]["pre_map_mean"] for p in physicians]
    tsp = [severity_results[p]["tsp_mean"] for p in physicians]
    ax.bar(x_pos - 0.18, tsp, 0.35, label="TSP (Transseptal Puncture)", color=bar_colors, alpha=0.7, edgecolor="black")
    ax.bar(x_pos + 0.18, pre_map, 0.35, label="PRE-MAP (Electroanatomic Mapping)", color=bar_colors, alpha=0.4, edgecolor="black", hatch="//")
    for i in range(len(physicians)):
        ax.text(i - 0.18, tsp[i] + 0.2, f"{tsp[i]:.1f}", ha="center", fontsize=9)
        ax.text(i + 0.18, pre_map[i] + 0.1, f"{pre_map[i]:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(physicians)
    ax.set_ylabel("Duration (minutes)")
    ax.set_title("Pre-Ablation Complexity: TSP and Mapping Times")
    ax.legend(fontsize=8)

    # (1,1) Additional procedure rates
    ax = axes[1, 1]
    add_rates = [severity_results[p]["pct_additional_procedures"] for p in physicians]
    bars = ax.bar(x_pos, add_rates, 0.5, color=bar_colors, alpha=0.7, edgecolor="black")
    for i, rate in enumerate(add_rates):
        ax.text(i, rate + 0.5, f"{rate:.1f}%", ha="center", fontweight="bold", fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(physicians)
    ax.set_ylabel("Percentage of Cases (%)")
    ax.set_title("Cases with Additional Procedures Beyond Standard PVI")

    # (1,2) Additional procedure type breakdown
    ax = axes[1, 2]
    proc_types = ["CTI", "BOX", "PST_BOX", "SVC", "AAFL"]
    proc_labels = ["CTI", "BOX", "PST BOX", "SVC", "AAFL"]
    proc_colors = ["#FF9800", "#E91E63", "#9C27B0", "#4CAF50", "#795548"]
    bottom = np.zeros(len(physicians))
    for proc_type, proc_label, proc_color in zip(proc_types, proc_labels, proc_colors):
        counts = [severity_results[p]["additional_breakdown"][proc_type] for p in physicians]
        ax.bar(x_pos, counts, 0.5, bottom=bottom, label=proc_label, color=proc_color, alpha=0.7, edgecolor="black")
        for i, c in enumerate(counts):
            if c > 0:
                ax.text(i, bottom[i] + c / 2, str(c), ha="center", fontsize=9, fontweight="bold", color="white")
        bottom += np.array(counts)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(physicians)
    ax.set_ylabel("Number of Cases")
    ax.set_title("Additional Procedure Types by Physician")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{EXTRA_DIR}/physician_severity_profile.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print("Saved: physician_severity_profile.png")

    return severity_results


def _prep_time_variability(df: pd.DataFrame, physicians: list) -> Dict:
    """7e. Deep dive into PT PREP/INTUBATION variability and its link to PT IN-OUT."""
    print("\n--- 7e. Prep Time (PT PREP/INTUBATION) Variability Deep Dive ---")

    prep_col = "PT PREP/INTUBATION"
    df_valid = df.dropna(subset=[prep_col, TARGET_COL])

    # Global stats
    r_global, p_global = stats.pearsonr(df_valid[prep_col], df_valid[TARGET_COL])
    r2_global = r_global ** 2
    print(f"  Correlation with PT IN-OUT: r={r_global:.3f} (R²={r2_global:.3f}, p={p_global:.4f})")

    outlier = df_valid[df_valid["outlier_class"] == 1]
    normal = df_valid[df_valid["outlier_class"] == 0]
    print(f"  Outlier mean prep: {outlier[prep_col].mean():.1f} min vs Normal: {normal[prep_col].mean():.1f} min "
          f"(+{outlier[prep_col].mean() - normal[prep_col].mean():.1f} min)")

    # Per-physician stats
    phys_prep_stats = {}
    for phys in physicians:
        sub = df_valid[df_valid["PHYSICIAN"] == phys]
        prep = sub[prep_col]
        sub_out = sub[sub["outlier_class"] == 1]
        sub_norm = sub[sub["outlier_class"] == 0]
        r_phys, p_phys = stats.pearsonr(sub[prep_col], sub[TARGET_COL]) if len(sub) > 2 else (0, 1)
        phys_prep_stats[phys] = {
            "n": int(len(sub)),
            "mean": round(float(prep.mean()), 1),
            "median": round(float(prep.median()), 1),
            "std": round(float(prep.std()), 1),
            "cv_pct": round(float(prep.std() / prep.mean() * 100), 1) if prep.mean() > 0 else 0,
            "min": round(float(prep.min()), 1),
            "max": round(float(prep.max()), 1),
            "outlier_mean": round(float(sub_out[prep_col].mean()), 1) if len(sub_out) > 0 else None,
            "normal_mean": round(float(sub_norm[prep_col].mean()), 1) if len(sub_norm) > 0 else None,
            "r_with_pt_inout": round(float(r_phys), 3),
            "p_with_pt_inout": round(float(p_phys), 4),
        }
        s = phys_prep_stats[phys]
        print(f"  {phys}: mean={s['mean']}, std={s['std']}, CV={s['cv_pct']}%, "
              f"range=[{s['min']}-{s['max']}], r={s['r_with_pt_inout']:.3f}")

    # Prep time by procedure type
    proc_flags = {
        "Standard PFA": ~df_valid["HAS_NOTE"] | (df_valid["Note"].astype(str).str.strip() == ""),
        "CTI": df_valid["NOTE_CTI"] == 1,
        "BOX/PST BOX": (df_valid["NOTE_BOX"] == 1) | (df_valid["NOTE_PST"] == 1),
    }
    prep_by_type = {}
    for label, mask in proc_flags.items():
        vals = df_valid.loc[mask, prep_col].dropna()
        if len(vals) > 0:
            prep_by_type[label] = {
                "n": int(len(vals)),
                "mean": round(float(vals.mean()), 1),
                "std": round(float(vals.std()), 1),
            }

    # === Figure: 2x2 prep time deep dive ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("PT PREP/INTUBATION: Variability Deep Dive", fontsize=15, fontweight="bold")

    # (0,0) Distribution by physician
    ax = axes[0, 0]
    for phys in physicians:
        sub = df_valid[df_valid["PHYSICIAN"] == phys][prep_col]
        ax.hist(sub, bins=15, alpha=0.5, label=f"{phys} (n={len(sub)}, μ={sub.mean():.1f})",
                color=PHYS_COLORS.get(phys, "gray"), edgecolor="black")
    ax.set_xlabel("PT PREP/INTUBATION (minutes)")
    ax.set_ylabel("Number of Cases")
    ax.set_title("Prep Time Distribution by Physician")
    ax.legend(fontsize=8)

    # (0,1) Scatter: Prep time vs PT IN-OUT colored by physician
    ax = axes[0, 1]
    for phys in physicians:
        sub = df_valid[df_valid["PHYSICIAN"] == phys]
        ax.scatter(sub[prep_col], sub[TARGET_COL], alpha=0.5,
                   color=PHYS_COLORS.get(phys, "gray"), edgecolor="black", s=40, label=phys)
    # Outliers marked with red ring
    ax.scatter(outlier[prep_col], outlier[TARGET_COL], facecolors="none",
               edgecolors="red", s=120, linewidths=2, label="Outlier case", zorder=5)
    # Regression line
    z = np.polyfit(df_valid[prep_col], df_valid[TARGET_COL], 1)
    x_line = np.linspace(df_valid[prep_col].min(), df_valid[prep_col].max(), 50)
    ax.plot(x_line, np.poly1d(z)(x_line), "r--", linewidth=2,
            label=f"r={r_global:.3f}, R²={r2_global:.1%}")
    ax.set_xlabel("PT PREP/INTUBATION (minutes)")
    ax.set_ylabel("PT IN-OUT (minutes)")
    ax.set_title("Prep Time vs Total Duration (with Outliers)")
    ax.legend(fontsize=7)

    # (1,0) Boxplot: Outlier vs Normal prep time by physician
    ax = axes[1, 0]
    positions = []
    box_data = []
    box_colors = []
    tick_labels = []
    for i, phys in enumerate(physicians):
        sub_norm = df_valid[(df_valid["PHYSICIAN"] == phys) & (df_valid["outlier_class"] == 0)][prep_col]
        sub_out = df_valid[(df_valid["PHYSICIAN"] == phys) & (df_valid["outlier_class"] == 1)][prep_col]
        if len(sub_norm) > 0:
            box_data.append(sub_norm.values)
            positions.append(i * 3)
            box_colors.append(PHYS_COLORS.get(phys, "gray"))
            tick_labels.append(f"{phys}\nNormal (n={len(sub_norm)})")
        if len(sub_out) > 0:
            box_data.append(sub_out.values)
            positions.append(i * 3 + 1)
            box_colors.append("red")
            tick_labels.append(f"{phys}\nOutlier (n={len(sub_out)})")
    bp = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.7)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_ylabel("PT PREP/INTUBATION (minutes)")
    ax.set_title("Prep Time: Normal vs Outlier Cases by Physician")

    # (1,1) Prep time trend over time per physician
    ax = axes[1, 1]
    for phys in physicians:
        sub = df_valid[df_valid["PHYSICIAN"] == phys].sort_values("DATE").reset_index(drop=True)
        case_seq = np.arange(1, len(sub) + 1)
        prep_vals = sub[prep_col].values
        ax.scatter(case_seq, prep_vals, alpha=0.4, color=PHYS_COLORS.get(phys, "gray"),
                   edgecolor="black", s=25)
        mask = ~np.isnan(prep_vals)
        if mask.sum() > 2:
            z_trend = np.polyfit(case_seq[mask], prep_vals[mask], 1)
            ax.plot(case_seq, np.poly1d(z_trend)(case_seq), color=PHYS_COLORS.get(phys, "gray"),
                    linewidth=2, label=f"{phys} ({z_trend[0]:+.2f} min/case)")
    ax.set_xlabel("Case Number (Chronological per Physician)")
    ax.set_ylabel("PT PREP/INTUBATION (minutes)")
    ax.set_title("Prep Time Trend Over Time")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{EXTRA_DIR}/prep_time_deep_dive.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print("Saved: prep_time_deep_dive.png")

    return {
        "global_r": round(float(r_global), 3),
        "global_r2_pct": round(float(r2_global * 100), 1),
        "global_p": round(float(p_global), 6),
        "outlier_mean": round(float(outlier[prep_col].mean()), 1),
        "normal_mean": round(float(normal[prep_col].mean()), 1),
        "diff_min": round(float(outlier[prep_col].mean() - normal[prep_col].mean()), 1),
        "per_physician": phys_prep_stats,
        "by_procedure_type": prep_by_type,
    }


def _repositioning_efficiency(df: pd.DataFrame, physicians: list) -> Dict:
    """7f. Deep dive into catheter repositioning time (ABL DURATION - ABL TIME)."""
    print("\n--- 7f. Catheter Repositioning Efficiency Deep Dive ---")

    df_valid = df.dropna(subset=["ABL DURATION", "ABL TIME", TARGET_COL]).copy()
    df_valid["REPO_TIME"] = df_valid["ABL DURATION"] - df_valid["ABL TIME"]

    # Global correlation
    r_repo, p_repo = stats.pearsonr(df_valid["REPO_TIME"], df_valid[TARGET_COL])
    r_abl_dur, _ = stats.pearsonr(df_valid["ABL DURATION"], df_valid[TARGET_COL])
    r_abl_time, _ = stats.pearsonr(df_valid["ABL TIME"], df_valid[TARGET_COL])
    print(f"  Correlation with PT IN-OUT:")
    print(f"    ABL DURATION (total):     r={r_abl_dur:.3f} (R²={r_abl_dur**2:.1%})")
    print(f"    Repositioning time:       r={r_repo:.3f} (R²={r_repo**2:.1%})")
    print(f"    ABL TIME (pulse-on only): r={r_abl_time:.3f} (R²={r_abl_time**2:.1%})")
    print(f"  -> Repositioning (not pulse-on) drives ABL DURATION's correlation with PT IN-OUT")

    outlier = df_valid[df_valid["outlier_class"] == 1]
    normal = df_valid[df_valid["outlier_class"] == 0]
    print(f"  Outlier mean repo: {outlier['REPO_TIME'].mean():.1f} min vs Normal: {normal['REPO_TIME'].mean():.1f} min "
          f"(+{outlier['REPO_TIME'].mean() - normal['REPO_TIME'].mean():.1f} min)")

    # Per-physician stats
    phys_repo_stats = {}
    for phys in physicians:
        sub = df_valid[df_valid["PHYSICIAN"] == phys]
        repo = sub["REPO_TIME"]
        sub_out = sub[sub["outlier_class"] == 1]
        sub_norm = sub[sub["outlier_class"] == 0]
        r_phys, p_phys = stats.pearsonr(sub["REPO_TIME"], sub[TARGET_COL]) if len(sub) > 2 else (0, 1)
        # Repositioning per ablation site
        repo_per_site = repo / sub["#ABL"]
        phys_repo_stats[phys] = {
            "n": int(len(sub)),
            "mean": round(float(repo.mean()), 1),
            "median": round(float(repo.median()), 1),
            "std": round(float(repo.std()), 1),
            "cv_pct": round(float(repo.std() / repo.mean() * 100), 1) if repo.mean() > 0 else 0,
            "outlier_mean": round(float(sub_out["REPO_TIME"].mean()), 1) if len(sub_out) > 0 else None,
            "normal_mean": round(float(sub_norm["REPO_TIME"].mean()), 1) if len(sub_norm) > 0 else None,
            "repo_per_site_mean": round(float(repo_per_site.mean()), 2),
            "repo_per_site_std": round(float(repo_per_site.std()), 2),
            "r_with_pt_inout": round(float(r_phys), 3),
        }
        s = phys_repo_stats[phys]
        print(f"  {phys}: mean={s['mean']} min, std={s['std']}, CV={s['cv_pct']}%, "
              f"per-site={s['repo_per_site_mean']:.2f} min/site, r={s['r_with_pt_inout']:.3f}")

    # Correlation: repo time vs #ABL (does more sites = proportionally more repo time?)
    r_sites, p_sites = stats.pearsonr(df_valid["#ABL"], df_valid["REPO_TIME"])
    print(f"  #ABL vs Repositioning time: r={r_sites:.3f}, p={p_sites:.4f}")

    # === Figure: 2x2 repositioning deep dive ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Catheter Repositioning Efficiency Deep Dive\n(Repositioning = ABL DURATION - ABL TIME)",
                 fontsize=14, fontweight="bold")

    # (0,0) Scatter: Repositioning time vs PT IN-OUT
    ax = axes[0, 0]
    for phys in physicians:
        sub = df_valid[df_valid["PHYSICIAN"] == phys]
        ax.scatter(sub["REPO_TIME"], sub[TARGET_COL], alpha=0.5,
                   color=PHYS_COLORS.get(phys, "gray"), edgecolor="black", s=40, label=phys)
    ax.scatter(outlier["REPO_TIME"], outlier[TARGET_COL], facecolors="none",
               edgecolors="red", s=120, linewidths=2, label="Outlier case", zorder=5)
    z = np.polyfit(df_valid["REPO_TIME"], df_valid[TARGET_COL], 1)
    x_line = np.linspace(df_valid["REPO_TIME"].min(), df_valid["REPO_TIME"].max(), 50)
    ax.plot(x_line, np.poly1d(z)(x_line), "r--", linewidth=2,
            label=f"r={r_repo:.3f}, R²={r_repo**2:.1%}")
    ax.set_xlabel("Repositioning Time (minutes)")
    ax.set_ylabel("PT IN-OUT (minutes)")
    ax.set_title("Repositioning Time vs Total Duration")
    ax.legend(fontsize=7)

    # (0,1) Scatter: #ABL vs Repositioning time (efficiency)
    ax = axes[0, 1]
    for phys in physicians:
        sub = df_valid[df_valid["PHYSICIAN"] == phys]
        ax.scatter(sub["#ABL"], sub["REPO_TIME"], alpha=0.5,
                   color=PHYS_COLORS.get(phys, "gray"), edgecolor="black", s=40, label=phys)
    z2 = np.polyfit(df_valid["#ABL"], df_valid["REPO_TIME"], 1)
    x_abl = np.linspace(df_valid["#ABL"].min(), df_valid["#ABL"].max(), 50)
    ax.plot(x_abl, np.poly1d(z2)(x_abl), "r--", linewidth=2,
            label=f"r={r_sites:.3f} ({z2[0]:+.2f} min/site)")
    ax.set_xlabel("Number of Ablation Sites (#ABL)")
    ax.set_ylabel("Repositioning Time (minutes)")
    ax.set_title("Do More Sites = More Repositioning Time?")
    ax.legend(fontsize=8)

    # (1,0) Boxplot: Repositioning time by physician (normal vs outlier)
    ax = axes[1, 0]
    positions = []
    box_data = []
    box_colors = []
    tick_labels = []
    for i, phys in enumerate(physicians):
        sub_norm = df_valid[(df_valid["PHYSICIAN"] == phys) & (df_valid["outlier_class"] == 0)]["REPO_TIME"]
        sub_out = df_valid[(df_valid["PHYSICIAN"] == phys) & (df_valid["outlier_class"] == 1)]["REPO_TIME"]
        if len(sub_norm) > 0:
            box_data.append(sub_norm.values)
            positions.append(i * 3)
            box_colors.append(PHYS_COLORS.get(phys, "gray"))
            tick_labels.append(f"{phys}\nNormal (n={len(sub_norm)})")
        if len(sub_out) > 0:
            box_data.append(sub_out.values)
            positions.append(i * 3 + 1)
            box_colors.append("red")
            tick_labels.append(f"{phys}\nOutlier (n={len(sub_out)})")
    bp = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.7)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_ylabel("Repositioning Time (minutes)")
    ax.set_title("Repositioning Time: Normal vs Outlier by Physician")

    # (1,1) Per-site repositioning efficiency by physician
    ax = axes[1, 1]
    x_pos = np.arange(len(physicians))
    bar_colors = [PHYS_COLORS.get(p, "gray") for p in physicians]
    means = [phys_repo_stats[p]["repo_per_site_mean"] for p in physicians]
    stds = [phys_repo_stats[p]["repo_per_site_std"] for p in physicians]
    ax.bar(x_pos, means, 0.5, yerr=stds, color=bar_colors, alpha=0.7, edgecolor="black",
           capsize=5, error_kw={"linewidth": 1.5})
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.02, f"{m:.2f}±{s:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(physicians)
    ax.set_ylabel("Minutes per Ablation Site")
    ax.set_title("Repositioning Efficiency: Time per Ablation Site")

    plt.tight_layout()
    plt.savefig(f"{EXTRA_DIR}/repositioning_deep_dive.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print("Saved: repositioning_deep_dive.png")

    return {
        "global_r": round(float(r_repo), 3),
        "global_r2_pct": round(float(r_repo ** 2 * 100), 1),
        "abl_duration_r": round(float(r_abl_dur), 3),
        "abl_time_r": round(float(r_abl_time), 3),
        "outlier_mean": round(float(outlier["REPO_TIME"].mean()), 1),
        "normal_mean": round(float(normal["REPO_TIME"].mean()), 1),
        "diff_min": round(float(outlier["REPO_TIME"].mean() - normal["REPO_TIME"].mean()), 1),
        "sites_vs_repo_r": round(float(r_sites), 3),
        "per_physician": phys_repo_stats,
    }
