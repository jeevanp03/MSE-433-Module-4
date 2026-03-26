"""
Phase 2: Exploratory data analysis and distribution plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.config import EDA_DIR, NUM_COLS, TARGET_COL, CLASS_LABELS
from src.viz import CLASS_COLORS, SAVE_DPI


def run_eda(df: pd.DataFrame) -> float:
    """Run EDA and return the outlier threshold (90th percentile).

    Also sets df['outlier_class'] in-place.
    """
    print("=" * 60)
    print("2. EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    target = df[TARGET_COL]

    # --- 2a. Distribution analysis ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Patient In-Out Duration: Distribution Analysis", fontsize=14, fontweight="bold")

    axes[0, 0].hist(target, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0, 0].set_title("Distribution of Total Procedure Time")
    axes[0, 0].set_xlabel("Duration (minutes)")
    axes[0, 0].set_ylabel("Number of Cases")

    axes[0, 1].boxplot(target.dropna(), vert=True)
    axes[0, 1].set_title("Spread of Procedure Duration")
    axes[0, 1].set_ylabel("Duration (minutes)")

    stats.probplot(target.dropna(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Normality Check (Q-Q Plot)")

    physician_groups = df.groupby("PHYSICIAN")[TARGET_COL].apply(list)
    axes[1, 1].boxplot([v for v in physician_groups.values], labels=physician_groups.index)
    axes[1, 1].set_title("Duration Comparison Across Physicians")
    axes[1, 1].set_ylabel("Duration (minutes)")

    plt.tight_layout()
    plt.savefig(f"{EDA_DIR}/eda_distribution.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print("Saved: eda_distribution.png")

    # --- 2b. Statistical outlier detection methods ---
    print("\n--- Outlier Detection Methods Comparison ---")

    Q1 = target.quantile(0.25)
    Q3 = target.quantile(0.75)
    IQR = Q3 - Q1
    iqr_upper = Q3 + 1.5 * IQR
    iqr_outliers = (target > iqr_upper)
    print(f"\nIQR Method (1.5*IQR): Q1={Q1}, Q3={Q3}, IQR={IQR}")
    print(f"  Upper bound: {iqr_upper:.1f} min -> {iqr_outliers.sum()} outliers ({iqr_outliers.mean()*100:.1f}%)")

    z_scores = np.abs(stats.zscore(target.dropna()))
    z2_outliers = z_scores > 2
    print(f"\nZ-Score Method (|Z|>2): {z2_outliers.sum()} outliers ({z2_outliers.mean()*100:.1f}%)")

    p90 = target.quantile(0.90)
    p95 = target.quantile(0.95)
    print(f"\nPercentile Method:")
    print(f"  > 90th percentile ({p90:.0f} min): {(target > p90).sum()} cases ({(target > p90).mean()*100:.1f}%)")
    print(f"  > 95th percentile ({p95:.0f} min): {(target > p95).sum()} cases ({(target > p95).mean()*100:.1f}%)")

    print(f"\nTop 15 longest operations:")
    print(target.nlargest(15).to_string())

    print(f"\nSkewness: {target.skew():.2f}")
    print(f"Kurtosis: {target.kurtosis():.2f}")

    # --- 2c. Decision: Top 10% (90th percentile) as outliers ---
    outlier_threshold = p90

    print(f"\n--- DECISION: Top 10% Outlier Classification (90th Percentile) ---")
    print(f"  Threshold: PT IN-OUT > {outlier_threshold:.0f} min")
    print(f"  0 = Normal (bottom 90%)")
    print(f"  1 = Outlier (top 10%, long-duration cases)")

    df["outlier_class"] = (df[TARGET_COL] > outlier_threshold).astype(int)

    print(f"\nFinal classification type: binary")
    print(f"Final class distribution:")
    print(df["outlier_class"].value_counts().sort_index())

    # Print outlier case numbers
    outlier_cases = df[df["outlier_class"] == 1].sort_values(TARGET_COL, ascending=False)
    case_nums = outlier_cases["CASE #"].astype(int).tolist()
    print(f"\nOutlier case numbers: {case_nums}")

    # --- 2d. Visualize outlier classes ---
    class_labels = CLASS_LABELS
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for cls in sorted(df["outlier_class"].unique()):
        subset = df[df["outlier_class"] == cls][TARGET_COL]
        axes[0].hist(subset, bins=20, alpha=0.6, label=class_labels[cls],
                     color=CLASS_COLORS[cls], edgecolor="black")
    axes[0].set_title("Procedure Duration: Normal vs Outlier")
    axes[0].set_xlabel("Duration (minutes)")
    axes[0].set_ylabel("Number of Cases")
    axes[0].axvline(outlier_threshold, color="red", linestyle="--",
                    label=f"Outlier Threshold ({outlier_threshold:.0f} min)")
    axes[0].legend()

    for cls in sorted(df["outlier_class"].unique()):
        subset = df[df["outlier_class"] == cls]
        axes[1].scatter(subset["CASE #"].astype(float), subset[TARGET_COL],
                        alpha=0.7, label=class_labels[cls], color=CLASS_COLORS[cls],
                        edgecolor="black", s=40)
    axes[1].set_title("Outlier Cases Over Time (by Case Number)")
    axes[1].set_xlabel("Case Number (Chronological)")
    axes[1].set_ylabel("Total Procedure Duration (minutes)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{EDA_DIR}/eda_outlier_classes.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print("Saved: eda_outlier_classes.png")

    # --- 2e. Correlation heatmap ---
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_matrix = df[NUM_COLS].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Between Procedural Timing Features")
    plt.tight_layout()
    plt.savefig(f"{EDA_DIR}/eda_correlation.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print("Saved: eda_correlation.png")

    return outlier_threshold
