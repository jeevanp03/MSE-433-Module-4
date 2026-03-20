"""
Phase 7: Additional analyses - learning curves, complexity, scheduling, severity profiles.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    scheduling_results = _case_scheduling(df)
    severity_results = _physician_severity(df, physicians)

    return {
        "learning_curve": learning_curve_results,
        "case_complexity": complexity_stats,
        "scheduling": scheduling_results,
        "physician_severity": severity_results,
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


def _case_scheduling(df: pd.DataFrame) -> Dict:
    """7c. Day-of scheduling / case order analysis."""
    print("\n--- 7c. Case Order / Scheduling Analysis ---")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Day-of Scheduling: Effect of Case Order on Procedure Duration", fontsize=14, fontweight="bold")

    df_sched = df.dropna(subset=["CASE_ORDER_IN_DAY", TARGET_COL])
    case_order_groups = df_sched.groupby("CASE_ORDER_IN_DAY")[TARGET_COL]
    orders = sorted(df_sched["CASE_ORDER_IN_DAY"].dropna().unique().astype(int))
    order_means = [case_order_groups.get_group(o).mean() for o in orders]
    order_counts = [len(case_order_groups.get_group(o)) for o in orders]

    ax = axes[0]
    ax.bar(orders, order_means, color="steelblue", alpha=0.7, edgecolor="black")
    for i, (mean, count) in enumerate(zip(order_means, order_counts)):
        ax.text(orders[i], mean + 1, f"{mean:.0f}\n(n={count})", ha="center", fontsize=8)
    ax.set_xlabel("Case Position in Daily Schedule")
    ax.set_ylabel("Average Duration (minutes)")
    ax.set_title("Average Procedure Duration by Daily Case Order")
    ax.set_xticks(orders)

    order_outlier_rates = [df_sched[df_sched["CASE_ORDER_IN_DAY"] == o]["outlier_class"].mean() * 100 for o in orders]
    ax = axes[1]
    ax.bar(orders, order_outlier_rates, color="orange", alpha=0.7, edgecolor="black")
    for i, rate in enumerate(order_outlier_rates):
        ax.text(orders[i], rate + 0.5, f"{rate:.1f}%", ha="center", fontsize=8)
    ax.set_xlabel("Case Position in Daily Schedule")
    ax.set_ylabel("Outlier Rate (%)")
    ax.set_title("Outlier Rate by Daily Case Order")
    ax.set_xticks(orders)

    scheduling_results = {
        "case_order_stats": {
            int(o): {
                "n_cases": int(c),
                "mean_duration": round(float(m), 1),
                "outlier_rate_pct": round(float(r), 1),
            }
            for o, c, m, r in zip(orders, order_counts, order_means, order_outlier_rates)
        }
    }
    print(f"  Case order vs mean duration: {', '.join(f'{o}={m:.0f}min' for o, m in zip(orders, order_means))}")

    plt.tight_layout()
    plt.savefig(f"{EXTRA_DIR}/case_order_scheduling.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print("Saved: case_order_scheduling.png")

    return scheduling_results


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
