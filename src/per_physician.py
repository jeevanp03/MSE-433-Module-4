"""
Phase 6: Per-physician outlier detection and modeling.
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import shap

from src.config import PHYS_DIR, EDA_DIR, PHYS_LGB_PARAMS, COMPARE_COLS, TARGET_COL
from src.viz import PHYS_COLORS, SAVE_DPI


def run_per_physician_analysis(
    df: pd.DataFrame,
    feature_cols: List[str],
    outlier_threshold: float,
    per_physician_results: Dict,
) -> Dict:
    """Run per-physician outlier detection, LightGBM+SHAP, and EDA comparison charts.

    Modifies df in-place (adds phys_outlier_class, phys_outlier_label).
    Returns per_physician_results dict.
    """
    print("\n" + "=" * 60)
    print("6. PER-PHYSICIAN OUTLIER ANALYSIS")
    print("=" * 60)

    # Features for per-physician models (exclude physician encoding)
    phys_feature_cols = [c for c in feature_cols if c != "PHYSICIAN_ENC"]
    physicians = sorted(df["PHYSICIAN"].unique())

    for phys in physicians:
        phys_safe = phys.replace(".", "").replace(" ", "_")
        phys_dir = f"{PHYS_DIR}/{phys_safe}"
        os.makedirs(phys_dir, exist_ok=True)

        phys_df = df[df["PHYSICIAN"] == phys].copy()
        phys_target = phys_df[TARGET_COL]
        n_cases = len(phys_df)

        print(f"\n--- {phys} ({n_cases} cases) ---")

        # Per-physician IQR-based outlier threshold
        phys_q1 = phys_target.quantile(0.25)
        phys_q3 = phys_target.quantile(0.75)
        phys_iqr = phys_q3 - phys_q1
        phys_threshold = phys_q3 + 1.0 * phys_iqr
        phys_df["phys_outlier_class"] = (phys_target > phys_threshold).astype(int)
        n_outliers_phys = int(phys_df["phys_outlier_class"].sum())
        n_normal_phys = n_cases - n_outliers_phys

        print(f"  IQR threshold: Q1={phys_q1:.0f}, Q3={phys_q3:.0f}, IQR={phys_iqr:.0f}")
        print(f"  Upper bound (Q3+1.0*IQR): {phys_threshold:.0f} min")
        print(f"  Normal: {n_normal_phys}, Outlier: {n_outliers_phys}")

        # Store back on main df
        df.loc[phys_df.index, "phys_outlier_class"] = phys_df["phys_outlier_class"]

        # Prepare model data
        phys_model_df = phys_df[phys_feature_cols + ["phys_outlier_class"]].dropna()
        X_phys = phys_model_df[phys_feature_cols]
        y_phys = phys_model_df["phys_outlier_class"]

        if y_phys.sum() < 2:
            print(f"  Skipping model: only {int(y_phys.sum())} outlier(s), need >= 2")
            per_physician_results[phys] = {
                "n_cases": n_cases,
                "method": "IQR (Q3+1.0*IQR)",
                "Q1": round(float(phys_q1), 1),
                "Q3": round(float(phys_q3), 1),
                "IQR": round(float(phys_iqr), 1),
                "threshold_minutes": round(float(phys_threshold), 1),
                "n_outliers": n_outliers_phys,
                "n_normal": n_normal_phys,
                "model_fitted": False,
                "reason_skipped": f"Only {int(y_phys.sum())} outlier(s)",
            }
            continue

        # LightGBM for this physician
        phys_model = lgb.LGBMClassifier(**PHYS_LGB_PARAMS)
        phys_model.fit(X_phys, y_phys)
        print(f"  Model fitted on {len(X_phys)} cases ({int(y_phys.sum())} outliers, {int(len(y_phys) - y_phys.sum())} normal)")

        # Feature importance plot
        fig, ax = plt.subplots(figsize=(10, 6))
        lgb.plot_importance(phys_model, ax=ax, importance_type="gain", max_num_features=15)
        ax.set_title(f"{phys}: Feature Importance by Information Gain")
        plt.tight_layout()
        plt.savefig(f"{phys_dir}/lgbm_feature_importance.png", dpi=SAVE_DPI, bbox_inches="tight")
        plt.close()

        # SHAP
        phys_explainer = shap.TreeExplainer(phys_model)
        phys_shap_values = phys_explainer.shap_values(X_phys)
        if isinstance(phys_shap_values, list):
            phys_sv = phys_shap_values[1]
        else:
            phys_sv = phys_shap_values

        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(phys_sv, X_phys, show=False, max_display=15)
        plt.title(f"{phys}: Feature Impact on Outlier Prediction (SHAP)")
        plt.tight_layout()
        plt.savefig(f"{phys_dir}/shap_summary.png", dpi=SAVE_DPI, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(phys_sv, X_phys, plot_type="bar", show=False, max_display=15)
        plt.title(f"{phys}: Average Feature Importance (SHAP)")
        plt.tight_layout()
        plt.savefig(f"{phys_dir}/shap_bar.png", dpi=SAVE_DPI, bbox_inches="tight")
        plt.close()

        # Top features
        phys_mean_abs_shap = np.abs(phys_sv).mean(axis=0)
        phys_top = pd.Series(phys_mean_abs_shap, index=phys_feature_cols).nlargest(4)
        print(f"  Top 4 SHAP features:")
        for feat, val in phys_top.items():
            print(f"    {feat}: {val:.3f}")

        # Dependence for top 2
        if len(phys_top) >= 2:
            fig, axes_dep = plt.subplots(1, 2, figsize=(14, 5))
            for idx, (feat, _) in enumerate(list(phys_top.items())[:2]):
                shap.dependence_plot(feat, phys_sv, X_phys, ax=axes_dep[idx], show=False)
                axes_dep[idx].set_title(f"Effect of {feat} on Outlier Prediction")
            plt.suptitle(f"{phys}: How Top Features Influence Outlier Classification", fontsize=13, fontweight="bold")
            plt.tight_layout()
            plt.savefig(f"{phys_dir}/shap_dependence.png", dpi=SAVE_DPI, bbox_inches="tight")
            plt.close()

        # Outlier vs normal comparison
        phys_outlier_df = phys_df[phys_df["phys_outlier_class"] == 1]
        phys_normal_df = phys_df[phys_df["phys_outlier_class"] == 0]
        phys_outlier_means = phys_outlier_df[COMPARE_COLS].mean()
        phys_normal_means = phys_normal_df[COMPARE_COLS].mean()

        print(f"  Saved plots to {phys_dir}/")

        per_physician_results[phys] = {
            "n_cases": n_cases,
            "method": "IQR (Q3+1.0*IQR)",
            "Q1": round(float(phys_q1), 1),
            "Q3": round(float(phys_q3), 1),
            "IQR": round(float(phys_iqr), 1),
            "threshold_minutes": round(float(phys_threshold), 1),
            "n_outliers": n_outliers_phys,
            "n_normal": n_normal_phys,
            "model_fitted": True,
            "top_shap_features": {feat: round(float(val), 3) for feat, val in phys_top.items()},
            "outlier_vs_normal_means": {
                col: {
                    "normal": round(float(phys_normal_means[col]), 1),
                    "outlier": round(float(phys_outlier_means[col]), 1),
                    "diff_pct": round(float((phys_outlier_means[col] - phys_normal_means[col]) / phys_normal_means[col] * 100), 1) if phys_normal_means[col] != 0 else 0,
                }
                for col in COMPARE_COLS
            },
            "outlier_cases": [
                {
                    "case_num": int(row["CASE #"]),
                    "date": str(row["DATE"].strftime("%Y-%m-%d") if pd.notna(row["DATE"]) else ""),
                    "pt_in_out_min": int(row[TARGET_COL]),
                    "note": str(row["Note"]) if pd.notna(row["Note"]) and row["Note"] != "" else None,
                }
                for _, row in phys_outlier_df.iterrows()
            ],
            "output_dir": phys_dir,
        }

    # Add label
    df["phys_outlier_label"] = df["phys_outlier_class"].map({0: "Normal", 1: "Outlier (IQR for Physician)"})

    # --- Per-physician EDA comparison charts ---
    _plot_per_physician_eda(df, physicians, per_physician_results, outlier_threshold)

    return per_physician_results


def _plot_per_physician_eda(
    df: pd.DataFrame,
    physicians: list,
    per_physician_results: Dict,
    outlier_threshold: float,
) -> None:
    """Generate per-physician EDA comparison charts."""
    print("\n--- Generating per-physician EDA comparison charts ---")

    # Figure 1: Distribution & outlier comparison (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Per-Physician: Procedure Duration Distribution & Outlier Detection", fontsize=15, fontweight="bold")

    for idx, phys in enumerate(physicians):
        phys_data = df[df["PHYSICIAN"] == phys]
        phys_target_vals = phys_data[TARGET_COL]
        phys_p90_val = per_physician_results[phys]["threshold_minutes"]
        color = PHYS_COLORS.get(phys, "gray")

        ax = axes[0, idx]
        normal_vals = phys_data[phys_data["phys_outlier_class"] == 0][TARGET_COL]
        outlier_vals = phys_data[phys_data["phys_outlier_class"] == 1][TARGET_COL]
        ax.hist(normal_vals, bins=15, alpha=0.6, color=color, edgecolor="black", label="Normal Duration")
        ax.hist(outlier_vals, bins=8, alpha=0.7, color="red", edgecolor="black", label="Long Duration (Outlier)")
        ax.axvline(phys_p90_val, color="red", linestyle="--", linewidth=1.5,
                   label=f"IQR bound = {phys_p90_val:.0f} min")
        ax.set_title(f"{phys}: Duration Distribution (n={len(phys_data)})")
        ax.set_xlabel("Duration (minutes)")
        ax.set_ylabel("Number of Cases")
        ax.legend(fontsize=8)

        ax = axes[1, idx]
        bp = ax.boxplot(phys_target_vals.dropna(), vert=True, widths=0.4,
                        patch_artist=True, positions=[0.3])
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.3)
        rng = np.random.RandomState(42)
        jitter = rng.normal(0.7, 0.04, size=len(normal_vals))
        ax.scatter(jitter[:len(normal_vals)], normal_vals, alpha=0.5, color=color, s=20, label="Normal Duration")
        jitter_out = rng.normal(0.7, 0.04, size=len(outlier_vals))
        ax.scatter(jitter_out[:len(outlier_vals)], outlier_vals, alpha=0.8, color="red",
                   s=50, edgecolor="black", zorder=5, label="Long Duration (Outlier)")
        ax.axhline(phys_p90_val, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(f"{phys}: Individual Case Durations")
        ax.set_ylabel("Duration (minutes)")
        ax.set_xticks([])
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{EDA_DIR}/eda_per_physician_distributions.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print("Saved: eda_per_physician_distributions.png")

    # Figure 2: Outlier rate comparison & feature breakdown (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Per-Physician: Outlier Rates, Thresholds & Feature Drivers", fontsize=15, fontweight="bold")

    x_pos = np.arange(len(physicians))
    phys_names = []
    outlier_rates = []
    outlier_counts = []
    normal_counts = []
    for phys in physicians:
        res = per_physician_results[phys]
        phys_names.append(phys)
        outlier_rates.append(res["n_outliers"] / res["n_cases"] * 100)
        outlier_counts.append(res["n_outliers"])
        normal_counts.append(res["n_normal"])

    # (0,0) Outlier rate bar chart
    ax = axes[0, 0]
    ax.bar(x_pos, normal_counts, color=[PHYS_COLORS[p] for p in phys_names],
           alpha=0.5, label="Normal Duration", edgecolor="black")
    ax.bar(x_pos, outlier_counts, bottom=normal_counts,
           color="red", alpha=0.7, label="Long Duration (Outlier)", edgecolor="black")
    for i, (rate, total) in enumerate(zip(outlier_rates, [per_physician_results[p]["n_cases"] for p in physicians])):
        ax.text(i, total + 1, f"{rate:.1f}%", ha="center", fontweight="bold", fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(phys_names)
    ax.set_ylabel("Total Number of Cases")
    ax.set_title("Case Volume & Outlier Rate by Physician")
    ax.legend()

    # (0,1) Per-physician thresholds vs global
    ax = axes[0, 1]
    thresholds = [per_physician_results[p]["threshold_minutes"] for p in physicians]
    bar_colors = [PHYS_COLORS[p] for p in physicians]
    ax.bar(x_pos, thresholds, color=bar_colors, edgecolor="black", alpha=0.7)
    ax.axhline(float(outlier_threshold), color="black", linestyle="--", linewidth=2,
               label=f"Global Threshold ({outlier_threshold:.0f} min)")
    for i, (t, phys) in enumerate(zip(thresholds, phys_names)):
        ax.text(i, t + 1, f"{t:.0f} min", ha="center", fontweight="bold", fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(phys_names)
    ax.set_ylabel("IQR Outlier Threshold (minutes)")
    ax.set_title("Per-Physician Outlier Threshold vs Global Threshold")
    ax.legend()

    # (1,0) Mean PT IN-OUT: Outlier vs Normal per physician
    ax = axes[1, 0]
    bar_width = 0.35
    normal_means_list = []
    outlier_means_list = []
    for phys in physicians:
        phys_data = df[df["PHYSICIAN"] == phys]
        normal_means_list.append(phys_data[phys_data["phys_outlier_class"] == 0][TARGET_COL].mean())
        outlier_means_list.append(phys_data[phys_data["phys_outlier_class"] == 1][TARGET_COL].mean())

    ax.bar(x_pos - bar_width/2, normal_means_list, bar_width, label="Normal Duration",
           color=[PHYS_COLORS[p] for p in physicians], alpha=0.5, edgecolor="black")
    ax.bar(x_pos + bar_width/2, outlier_means_list, bar_width, label="Long Duration (Outlier)",
           color="red", alpha=0.7, edgecolor="black")
    for i in range(len(physicians)):
        ax.text(i - bar_width/2, normal_means_list[i] + 1, f"{normal_means_list[i]:.0f}",
                ha="center", fontsize=9)
        ax.text(i + bar_width/2, outlier_means_list[i] + 1, f"{outlier_means_list[i]:.0f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(phys_names)
    ax.set_ylabel("Average Duration (minutes)")
    ax.set_title("Average Procedure Duration: Normal vs Outlier")
    ax.legend()

    # (1,1) Top SHAP feature comparison across physicians (heatmap)
    ax = axes[1, 1]
    all_top_feats = []
    for phys in physicians:
        res = per_physician_results[phys]
        if res["model_fitted"] and res.get("top_shap_features"):
            all_top_feats.extend(res["top_shap_features"].keys())
    unique_feats = list(dict.fromkeys(all_top_feats))

    shap_matrix = []
    for phys in physicians:
        res = per_physician_results[phys]
        row = []
        if res["model_fitted"] and res.get("top_shap_features"):
            shap_dict = res["top_shap_features"]
            for feat in unique_feats:
                row.append(shap_dict.get(feat, 0.0))
        else:
            row = [0.0] * len(unique_feats)
        shap_matrix.append(row)

    shap_df = pd.DataFrame(shap_matrix, index=physicians, columns=unique_feats)
    sns.heatmap(shap_df, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Mean SHAP Value"})
    ax.set_title("SHAP Feature Importance Comparison Across Physicians")
    ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(f"{EDA_DIR}/eda_per_physician_comparison.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print("Saved: eda_per_physician_comparison.png")

    # Figure 3: Per-physician feature means (outlier vs normal) side-by-side
    compare_cols_short = COMPARE_COLS
    n_feats = len(compare_cols_short)
    fig, axes = plt.subplots(1, len(physicians), figsize=(7 * len(physicians), 8))
    fig.suptitle("Per-Physician: Feature Duration Breakdown (Normal vs Outlier)", fontsize=15, fontweight="bold")

    if len(physicians) == 1:
        axes = [axes]

    for idx, phys in enumerate(physicians):
        ax = axes[idx]
        phys_data = df[df["PHYSICIAN"] == phys]
        n_means = phys_data[phys_data["phys_outlier_class"] == 0][compare_cols_short].mean()
        o_means = phys_data[phys_data["phys_outlier_class"] == 1][compare_cols_short].mean()

        y_pos = np.arange(n_feats)
        ax.barh(y_pos + 0.2, n_means.values, 0.35, label="Normal Duration",
                color=PHYS_COLORS.get(phys, "gray"), alpha=0.5, edgecolor="black")
        ax.barh(y_pos - 0.2, o_means.values, 0.35, label="Long Duration (Outlier)",
                color="red", alpha=0.7, edgecolor="black")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(compare_cols_short, fontsize=9)
        ax.set_xlabel("Duration (minutes)")
        ax.set_title(f"{phys}: Feature Comparison (n={len(phys_data)})")
        ax.legend(fontsize=8)
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(f"{EDA_DIR}/eda_per_physician_feature_comparison.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print("Saved: eda_per_physician_feature_comparison.png")
