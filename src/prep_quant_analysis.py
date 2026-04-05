"""
Prep Phase Quantitative Analysis
=================================
Analyzes AI-generated prep-tracker data to identify sub-phase variability
drivers within the PT PREP/INTUBATION phase.

Mirrors the original analysis approach (variability decomposition, outlier
detection, feature importance via LightGBM + SHAP) but applied to the
granular sub-phases captured by the EP Voice Tracker.

Analyses:
  1. Sub-phase duration descriptive stats & coefficient of variation
  2. Variability decomposition: which sub-phase contributes most to total variance
  3. Outlier detection (90th percentile on total prep time)
  4. LightGBM + SHAP to identify what drives long prep cases
  5. Physician and nurse effect comparison
  6. Delay type impact analysis
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder

from src.config import RANDOM_STATE, OUTPUT_DIR
from src.viz import PHYS_COLORS, SAVE_DPI

PREP_DIR = OUTPUT_DIR / "prep_quant_analysis"

# Sub-phase column definitions (computed from timestamps)
SUBPHASE_COLS = [
    "Entry_to_Anesthesia",
    "Anesthesia_to_Intubation",
    "Intubation_to_Draping",
    "Draping_to_Access",
]
SUBPHASE_LABELS = {
    "Entry_to_Anesthesia": "Patient Entry\n→ Anesthesia",
    "Anesthesia_to_Intubation": "Anesthesia\n→ Intubation",
    "Intubation_to_Draping": "Intubation\n→ Draping",
    "Draping_to_Access": "Draping\n→ Access",
}


def compute_subphases(df: pd.DataFrame) -> pd.DataFrame:
    """Compute sub-phase durations (minutes) from timestamps."""
    df = df.copy()
    for col in ["Patient Entry", "Anesthesia", "Intubation", "Draping", "Access"]:
        df[col] = pd.to_datetime(df[col])

    df["Entry_to_Anesthesia"] = (df["Anesthesia"] - df["Patient Entry"]).dt.total_seconds() / 60
    df["Anesthesia_to_Intubation"] = (df["Intubation"] - df["Anesthesia"]).dt.total_seconds() / 60
    df["Intubation_to_Draping"] = (df["Draping"] - df["Intubation"]).dt.total_seconds() / 60
    df["Draping_to_Access"] = (df["Access"] - df["Draping"]).dt.total_seconds() / 60
    df["Total_Prep"] = (df["Access"] - df["Patient Entry"]).dt.total_seconds() / 60

    return df


def plot_subphase_distributions(df: pd.DataFrame) -> None:
    """Plot 1: Sub-phase duration distributions with variability metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Sub-Phase Duration Distributions\n(Simulated EP Voice Tracker Data)",
                 fontsize=14, fontweight="bold")

    for ax, col in zip(axes.flat, SUBPHASE_COLS):
        vals = df[col]
        cv = vals.std() / vals.mean() * 100
        ax.hist(vals, bins=15, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(vals.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean={vals.mean():.1f}")
        ax.axvline(vals.quantile(0.9), color="orange", linestyle=":", linewidth=1.5,
                   label=f"P90={vals.quantile(0.9):.1f}")
        ax.set_xlabel("Duration (min)")
        ax.set_ylabel("Count")
        ax.set_title(f"{SUBPHASE_LABELS[col]}\nCV={cv:.1f}%", fontsize=11)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(PREP_DIR / "subphase_distributions.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_variability_decomposition(df: pd.DataFrame) -> None:
    """Plot 2: Which sub-phase contributes most to total prep time variance."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Variability Decomposition: What Drives Total Prep Time?",
                 fontsize=14, fontweight="bold")

    # (a) Stacked bar of mean durations
    means = [df[c].mean() for c in SUBPHASE_COLS]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
    bottom = 0
    for i, (col, m) in enumerate(zip(SUBPHASE_COLS, means)):
        axes[0].bar("Mean Total", m, bottom=bottom, color=colors[i],
                     label=SUBPHASE_LABELS[col].replace("\n", " "), edgecolor="white")
        bottom += m
    axes[0].set_ylabel("Duration (min)")
    axes[0].set_title("(a) Mean Duration Breakdown")
    axes[0].legend(fontsize=8, loc="upper right")

    # (b) Variance contribution (proportion of total variance explained)
    variances = [df[c].var() for c in SUBPHASE_COLS]
    # Also compute covariance contribution via correlation with total
    corr_contributions = []
    for col in SUBPHASE_COLS:
        r = df[col].corr(df["Total_Prep"])
        contrib = r * df[col].std() / df["Total_Prep"].std()
        corr_contributions.append(contrib)

    total_var = sum(variances)
    var_pct = [v / total_var * 100 for v in variances]
    bars = axes[1].barh(
        [SUBPHASE_LABELS[c].replace("\n", " ") for c in SUBPHASE_COLS],
        var_pct, color=colors, edgecolor="white"
    )
    for bar, pct in zip(bars, var_pct):
        axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                     f"{pct:.1f}%", va="center", fontsize=9)
    axes[1].set_xlabel("% of Total Variance")
    axes[1].set_title("(b) Variance Contribution")

    # (c) Correlation with total prep time
    corrs = [df[c].corr(df["Total_Prep"]) for c in SUBPHASE_COLS]
    bars = axes[2].barh(
        [SUBPHASE_LABELS[c].replace("\n", " ") for c in SUBPHASE_COLS],
        corrs, color=colors, edgecolor="white"
    )
    for bar, r in zip(bars, corrs):
        axes[2].text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                     f"r={r:.2f}", va="center", fontsize=9)
    axes[2].set_xlabel("Pearson r with Total Prep")
    axes[2].set_title("(c) Correlation with Total")
    axes[2].set_xlim(0, 1.0)

    plt.tight_layout()
    fig.savefig(PREP_DIR / "variability_decomposition.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_physician_nurse_effects(df: pd.DataFrame) -> None:
    """Plot 3: Physician and nurse effects on sub-phase durations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Physician & Nurse Effects on Sub-Phase Durations",
                 fontsize=14, fontweight="bold")

    # (a) By physician
    phys_data = df.groupby("Doctor")[SUBPHASE_COLS].mean()
    phys_data.columns = [SUBPHASE_LABELS[c].replace("\n", " ") for c in SUBPHASE_COLS]
    phys_data.plot(kind="bar", ax=axes[0], color=["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"],
                   edgecolor="white", width=0.7)
    axes[0].set_ylabel("Mean Duration (min)")
    axes[0].set_title("(a) By Physician")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].tick_params(axis="x", rotation=0)

    # (b) By nurse
    nurse_data = df.groupby("Nurse")[SUBPHASE_COLS].mean()
    nurse_data.columns = [SUBPHASE_LABELS[c].replace("\n", " ") for c in SUBPHASE_COLS]
    nurse_data.plot(kind="bar", ax=axes[1], color=["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"],
                    edgecolor="white", width=0.7)
    axes[1].set_ylabel("Mean Duration (min)")
    axes[1].set_title("(b) By Nurse")
    axes[1].legend(fontsize=7, loc="upper right")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    fig.savefig(PREP_DIR / "physician_nurse_effects.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_delay_impact(df: pd.DataFrame) -> None:
    """Plot 4: Impact of delay types on total prep time."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Delay Type Impact on Prep Duration",
                 fontsize=14, fontweight="bold")

    # (a) Total prep by delay type (boxplot)
    delay_col = df["Delay Type"].fillna("No Delay")
    plot_df = df.copy()
    plot_df["Delay Category"] = delay_col
    order = ["No Delay", "Equipment", "Cable", "Positioning", "Staff Wait"]
    order = [o for o in order if o in plot_df["Delay Category"].values]
    sns.boxplot(data=plot_df, x="Delay Category", y="Total_Prep", order=order,
                ax=axes[0], palette="Set2")
    axes[0].set_ylabel("Total Prep Time (min)")
    axes[0].set_title("(a) Total Prep by Delay Type")

    # (b) Delay frequency and mean extra time
    delay_df = df[df["Delay Type"].notna()].copy()
    if len(delay_df) > 0:
        freq = delay_df["Delay Type"].value_counts()
        mean_dur = delay_df.groupby("Delay Type")["Delay Duration (min)"].mean()
        combined = pd.DataFrame({"Count": freq, "Mean Delay (min)": mean_dur}).sort_values("Count", ascending=True)

        ax2 = axes[1]
        bars = ax2.barh(combined.index, combined["Count"], color="#2196F3", edgecolor="white", alpha=0.7)
        ax2.set_xlabel("Number of Cases")
        ax2.set_title("(b) Delay Frequency & Severity")

        # Annotate mean delay duration directly on each bar
        for i, (idx, row) in enumerate(combined.iterrows()):
            ax2.text(row["Count"] + 0.1, i, f'{row["Mean Delay (min)"]:.1f} min avg',
                     va="center", fontsize=9, color="#FF5722", fontweight="bold")

    plt.tight_layout()
    fig.savefig(PREP_DIR / "delay_impact.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_shap_analysis(df: pd.DataFrame) -> None:
    """Plot 5: LightGBM + SHAP feature importance for prep outlier classification."""
    import shap

    # Prepare features
    features = SUBPHASE_COLS.copy()
    le_doc = LabelEncoder()
    le_nurse = LabelEncoder()
    df_model = df.copy()
    df_model["Doctor_enc"] = le_doc.fit_transform(df_model["Doctor"])
    df_model["Nurse_enc"] = le_nurse.fit_transform(df_model["Nurse"])
    df_model["Has_Delay"] = df_model["Delay Type"].notna().astype(int)
    df_model["Delay_Duration"] = df_model["Delay Duration (min)"].fillna(0)

    feature_cols = features + ["Doctor_enc", "Nurse_enc", "Has_Delay", "Delay_Duration"]
    feature_labels = {
        "Entry_to_Anesthesia": "Entry → Anesthesia",
        "Anesthesia_to_Intubation": "Anesthesia → Intubation",
        "Intubation_to_Draping": "Intubation → Draping",
        "Draping_to_Access": "Draping → Access",
        "Doctor_enc": "Physician",
        "Nurse_enc": "Nurse",
        "Has_Delay": "Has Delay Flag",
        "Delay_Duration": "Delay Duration",
    }

    # Outlier label: 90th percentile of total prep
    p90 = df_model["Total_Prep"].quantile(0.9)
    y = (df_model["Total_Prep"] >= p90).astype(int)
    X = df_model[feature_cols]

    # Train LightGBM
    model = lgb.LGBMClassifier(
        n_estimators=75, max_depth=3, num_leaves=10,
        learning_rate=0.05, class_weight="balanced",
        random_state=RANDOM_STATE, verbosity=-1,
    )
    model.fit(X, y)

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For binary classification, shap_values may be [class0, class1]
    if isinstance(shap_values, list):
        sv = shap_values[1]  # outlier class
    else:
        sv = shap_values

    # Rename features for display
    X_display = X.rename(columns=feature_labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("SHAP Analysis: What Drives Long Prep Cases?\n(LightGBM on simulated tracker data)",
                 fontsize=14, fontweight="bold")

    # (a) Bar plot of mean |SHAP|
    mean_abs_shap = np.abs(sv).mean(axis=0)
    feat_importance = pd.Series(mean_abs_shap, index=X_display.columns).sort_values(ascending=True)
    feat_importance.plot(kind="barh", ax=axes[0], color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Mean |SHAP value|")
    axes[0].set_title("(a) Feature Importance (Mean |SHAP|)")

    # (b) Beeswarm-style dot plot
    ax1 = axes[1]
    shap.summary_plot(sv, X_display, plot_type="dot", show=False, max_display=8)
    ax1.set_title("(b) SHAP Beeswarm")

    plt.tight_layout()
    fig.savefig(PREP_DIR / "shap_analysis.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close("all")

    return model, sv, X, y, p90


def plot_outlier_comparison(df: pd.DataFrame) -> None:
    """Plot 6: Outlier vs normal cases across sub-phases."""
    p90 = df["Total_Prep"].quantile(0.9)
    df = df.copy()
    df["Outlier"] = np.where(df["Total_Prep"] >= p90, "Outlier", "Normal")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Outlier vs Normal Prep Cases (threshold: {p90:.0f} min)",
                 fontsize=14, fontweight="bold")

    # (a) Mean sub-phase comparison
    comp = df.groupby("Outlier")[SUBPHASE_COLS].mean().T
    comp.index = [SUBPHASE_LABELS[c].replace("\n", " ") for c in SUBPHASE_COLS]
    comp.plot(kind="barh", ax=axes[0], color=["steelblue", "orange"], edgecolor="white")
    axes[0].set_xlabel("Mean Duration (min)")
    axes[0].set_title("(a) Mean Sub-Phase Duration")
    axes[0].legend(title="Class")

    # (b) Total prep distribution split
    for label, color in [("Normal", "steelblue"), ("Outlier", "orange")]:
        subset = df[df["Outlier"] == label]["Total_Prep"]
        axes[1].hist(subset, bins=12, color=color, alpha=0.7, label=f"{label} (n={len(subset)})",
                     edgecolor="white")
    axes[1].axvline(p90, color="red", linestyle="--", linewidth=2, label=f"P90={p90:.0f} min")
    axes[1].set_xlabel("Total Prep Time (min)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("(b) Total Prep Distribution")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(PREP_DIR / "outlier_comparison.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def generate_stats_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate descriptive statistics table for the report."""
    stats_rows = []
    for col in SUBPHASE_COLS + ["Total_Prep"]:
        vals = df[col]
        stats_rows.append({
            "Sub-Phase": SUBPHASE_LABELS.get(col, "Total Prep").replace("\n", " "),
            "Mean (min)": round(vals.mean(), 1),
            "Median (min)": round(vals.median(), 1),
            "Std Dev": round(vals.std(), 1),
            "CV (%)": round(vals.std() / vals.mean() * 100, 1),
            "Min": round(vals.min(), 1),
            "Max": round(vals.max(), 1),
            "P90": round(vals.quantile(0.9), 1),
        })
    return pd.DataFrame(stats_rows)


def run_prep_quant_analysis(df: pd.DataFrame) -> dict:
    """Run the full prep phase quantitative analysis.

    Args:
        df: Raw prep tracker data (from prep_data_gen or loaded from Excel/CSV)

    Returns:
        Dictionary of analysis results for export
    """
    PREP_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("PREP PHASE: Quantitative Analysis")
    print("=" * 60)

    # Compute sub-phase durations
    df = compute_subphases(df)

    # Descriptive stats
    stats_df = generate_stats_summary(df)
    stats_df.to_csv(PREP_DIR / "subphase_statistics.csv", index=False)
    print("\n  Sub-Phase Statistics:")
    print(stats_df.to_string(index=False))

    # Statistical tests: Kruskal-Wallis for physician differences
    print("\n  Kruskal-Wallis tests (physician effect):")
    kw_results = {}
    for col in SUBPHASE_COLS:
        groups = [group[col].values for _, group in df.groupby("Doctor")]
        if len(groups) >= 2:
            stat, p = stats.kruskal(*groups)
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "ns"
            kw_results[col] = {"H": round(stat, 2), "p": round(p, 4), "sig": sig}
            print(f"    {SUBPHASE_LABELS[col].replace(chr(10), ' ')}: H={stat:.2f}, p={p:.4f} {sig}")

    # Correlation matrix
    corr = df[SUBPHASE_COLS + ["Total_Prep"]].corr()

    # Generate all plots
    print("\n  Generating visualizations...")
    plot_subphase_distributions(df)
    print("    [1/6] subphase_distributions.png")

    plot_variability_decomposition(df)
    print("    [2/6] variability_decomposition.png")

    plot_physician_nurse_effects(df)
    print("    [3/6] physician_nurse_effects.png")

    plot_delay_impact(df)
    print("    [4/6] delay_impact.png")

    model, sv, X, y, p90 = plot_shap_analysis(df)
    print("    [5/6] shap_analysis.png")

    plot_outlier_comparison(df)
    print("    [6/6] outlier_comparison.png")

    # Summary results
    n_outliers = int((df["Total_Prep"] >= p90).sum())
    highest_cv_phase = stats_df.loc[stats_df["CV (%)"].idxmax(), "Sub-Phase"]

    results = {
        "n_cases": len(df),
        "total_prep_mean": round(df["Total_Prep"].mean(), 1),
        "total_prep_std": round(df["Total_Prep"].std(), 1),
        "p90_threshold": round(p90, 1),
        "n_outliers": n_outliers,
        "highest_cv_phase": highest_cv_phase,
        "kruskal_wallis": kw_results,
        "stats_table": stats_df.to_dict("records"),
    }

    print(f"\n  Summary:")
    print(f"    Cases analyzed: {len(df)}")
    print(f"    Total prep: {results['total_prep_mean']} +/- {results['total_prep_std']} min")
    print(f"    Outlier threshold (P90): {p90:.1f} min")
    print(f"    Outliers: {n_outliers} / {len(df)}")
    print(f"    Highest variability: {highest_cv_phase}")

    return results
