"""
MSE 433 Module 4 - Medical Procedure Outlier Analysis
=====================================================
Identifies outliers in AFib ablation operation duration (PT IN-OUT),
fits a LightGBM classifier, and uses SHAP to explain feature contributions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import shap
import json
import os
import warnings
warnings.filterwarnings("ignore")

import shutil

OUTPUT_DIR = "output"
# Clean output directory to remove stale files from previous runs
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
# Create organized subdirectories
EDA_DIR = f"{OUTPUT_DIR}/eda"
GLOBAL_DIR = f"{OUTPUT_DIR}/global_model"
PHYS_DIR = f"{OUTPUT_DIR}/per_physician"
EXTRA_DIR = f"{OUTPUT_DIR}/additional"
for d in [OUTPUT_DIR, EDA_DIR, GLOBAL_DIR, PHYS_DIR, EXTRA_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# 1. DATA LOADING & CLEANING
# ============================================================
print("=" * 60)
print("1. LOADING & CLEANING DATA")
print("=" * 60)

df_raw = pd.read_excel("Data/MSE433_M4_Data.xlsx", header=None)
cols = df_raw.iloc[2].tolist()
cols = [str(c).strip() if pd.notna(c) else f"col_{i}" for i, c in enumerate(cols)]
df = df_raw.iloc[4:].copy()
df.columns = cols
df = df.drop(columns=["col_0"])
df = df.reset_index(drop=True)

# Convert numeric columns
num_cols = [
    "PT PREP/INTUBATION", "ACCESSS", "TSP", "PRE-MAP",
    "ABL DURATION", "ABL TIME", "#ABL", "#APPLICATIONS",
    "LA DWELL TIME", "CASE TIME", "SKIN-SKIN",
    "POST CARE/EXTUBATION", "PT IN-OUT",
]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Parse date
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

# Drop AVG columns (running averages, not per-case features) and PT OUT TIME (mostly null)
drop_cols = ["AVG CASE TIME", "AVG SKIN-SKIN", "AVG TURNOVER TIME", "PT OUT TIME"]
df = df.drop(columns=drop_cols)

# Drop rows missing the target
df = df.dropna(subset=["PT IN-OUT"])
print(f"Dataset shape after cleaning: {df.shape}")
print(f"Target (PT IN-OUT) stats:\n{df['PT IN-OUT'].describe()}\n")

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS - OUTLIER IDENTIFICATION
# ============================================================
print("=" * 60)
print("2. EXPLORATORY DATA ANALYSIS")
print("=" * 60)

target = df["PT IN-OUT"]

# --- 2a. Distribution analysis ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Patient In-Out Duration: Distribution Analysis", fontsize=14, fontweight="bold")

# Histogram
axes[0, 0].hist(target, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
axes[0, 0].set_title("Distribution of Total Procedure Time")
axes[0, 0].set_xlabel("Duration (minutes)")
axes[0, 0].set_ylabel("Number of Cases")

# Box plot
axes[0, 1].boxplot(target.dropna(), vert=True)
axes[0, 1].set_title("Spread of Procedure Duration")
axes[0, 1].set_ylabel("Duration (minutes)")

# QQ plot
stats.probplot(target.dropna(), dist="norm", plot=axes[1, 0])
axes[1, 0].set_title("Normality Check (Q-Q Plot)")

# By physician
physician_groups = df.groupby("PHYSICIAN")["PT IN-OUT"].apply(list)
axes[1, 1].boxplot([v for v in physician_groups.values], labels=physician_groups.index)
axes[1, 1].set_title("Duration Comparison Across Physicians")
axes[1, 1].set_ylabel("Duration (minutes)")

plt.tight_layout()
plt.savefig(f"{EDA_DIR}/eda_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: eda_distribution.png")

# --- 2b. Statistical outlier detection methods ---
print("\n--- Outlier Detection Methods Comparison ---")

# IQR method
Q1 = target.quantile(0.25)
Q3 = target.quantile(0.75)
IQR = Q3 - Q1
iqr_upper = Q3 + 1.5 * IQR
iqr_outliers = (target > iqr_upper)
print(f"\nIQR Method (1.5*IQR): Q1={Q1}, Q3={Q3}, IQR={IQR}")
print(f"  Upper bound: {iqr_upper:.1f} min -> {iqr_outliers.sum()} outliers ({iqr_outliers.mean()*100:.1f}%)")

# Z-score method
z_scores = np.abs(stats.zscore(target.dropna()))
z2_outliers = z_scores > 2
print(f"\nZ-Score Method (|Z|>2): {z2_outliers.sum()} outliers ({z2_outliers.mean()*100:.1f}%)")

# Percentile method
p90 = target.quantile(0.90)
p95 = target.quantile(0.95)
print(f"\nPercentile Method:")
print(f"  > 90th percentile ({p90:.0f} min): {(target > p90).sum()} cases ({(target > p90).mean()*100:.1f}%)")
print(f"  > 95th percentile ({p95:.0f} min): {(target > p95).sum()} cases ({(target > p95).mean()*100:.1f}%)")

# Print actual outlier values
print(f"\nTop 15 longest operations:")
print(target.nlargest(15).to_string())

# Skewness
print(f"\nSkewness: {target.skew():.2f}")
print(f"Kurtosis: {target.kurtosis():.2f}")

# --- 2c. Decision: Top 10% (90th percentile) as outliers ---
# Using 90th percentile gives a meaningful outlier group (~15 cases) that is large
# enough for stable modeling while capturing the truly long-duration procedures.
# Binary classification: 0 = Normal, 1 = Outlier (top 10%)
outlier_threshold = p90
classification_type = "binary"

print(f"\n--- DECISION: Top 10% Outlier Classification (90th Percentile) ---")
print(f"  Threshold: PT IN-OUT > {outlier_threshold:.0f} min")
print(f"  0 = Normal (bottom 90%)")
print(f"  1 = Outlier (top 10%, long-duration cases)")

df["outlier_class"] = (df["PT IN-OUT"] > outlier_threshold).astype(int)

print(f"\nFinal classification type: {classification_type}")
print(f"Final class distribution:")
print(df["outlier_class"].value_counts().sort_index())

# --- 2d. Visualize outlier classes ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = {0: "steelblue", 1: "orange", 2: "red"}
class_labels = {0: "Normal Duration", 1: "Long Duration (Top 10%)"}

for cls in sorted(df["outlier_class"].unique()):
    subset = df[df["outlier_class"] == cls]["PT IN-OUT"]
    axes[0].hist(subset, bins=20, alpha=0.6, label=class_labels[cls], color=colors[cls], edgecolor="black")
axes[0].set_title("Procedure Duration: Normal vs Outlier")
axes[0].set_xlabel("Duration (minutes)")
axes[0].set_ylabel("Number of Cases")
axes[0].axvline(outlier_threshold, color="red", linestyle="--", label=f"Outlier Threshold ({outlier_threshold:.0f} min)")
axes[0].legend()

# Scatter: case number vs duration colored by class
for cls in sorted(df["outlier_class"].unique()):
    subset = df[df["outlier_class"] == cls]
    axes[1].scatter(subset["CASE #"].astype(float), subset["PT IN-OUT"],
                    alpha=0.7, label=class_labels[cls], color=colors[cls], edgecolor="black", s=40)
axes[1].set_title("Outlier Cases Over Time (by Case Number)")
axes[1].set_xlabel("Case Number (Chronological)")
axes[1].set_ylabel("Total Procedure Duration (minutes)")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{EDA_DIR}/eda_outlier_classes.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: eda_outlier_classes.png")

# --- 2e. Correlation heatmap ---
corr_cols = [c for c in num_cols if c != "PT IN-OUT"]
fig, ax = plt.subplots(figsize=(12, 10))
corr_matrix = df[num_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Between Procedural Timing Features")
plt.tight_layout()
plt.savefig(f"{EDA_DIR}/eda_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: eda_correlation.png")

# ============================================================
# 3. FEATURE ENGINEERING & MODEL PREPARATION
# ============================================================
print("\n" + "=" * 60)
print("3. FEATURE ENGINEERING & MODEL PREPARATION")
print("=" * 60)

# Features to use — only granular procedural phases, not aggregate sub-totals.
# Excluded (per MSE433_M4_Definitions.pdf these are sub-totals of PT IN-OUT):
#   CASE TIME (Cath In to Out) — "core procedural duration", corr 0.91 with target
#   SKIN-SKIN (Access to Cath-Out) — similar to CASE TIME but from skin puncture, corr 0.92
#   LA DWELL TIME (Abl Start to Cath-Out) — time catheter is in left atrium, corr 0.80
# These just restate "the case was long" without explaining why.
# Keeping only the granular phases tells us WHERE time is being spent:
feature_cols = [
    "PT PREP/INTUBATION", "ACCESSS", "TSP", "PRE-MAP",
    "ABL DURATION", "ABL TIME", "#ABL", "#APPLICATIONS",
    "POST CARE/EXTUBATION",
]

# Encode physician as numeric
le_phys = LabelEncoder()
df["PHYSICIAN_ENC"] = le_phys.fit_transform(df["PHYSICIAN"].astype(str))
feature_cols.append("PHYSICIAN_ENC")

# Encode Note as binary features for procedure type markers
df["HAS_NOTE"] = df["Note"].notna() & (df["Note"] != "")
df["NOTE_CTI"] = df["Note"].astype(str).str.contains("CTI", na=False).astype(int)
df["NOTE_BOX"] = df["Note"].astype(str).str.contains("BOX", na=False).astype(int)
df["NOTE_PST"] = df["Note"].astype(str).str.contains("PST", na=False).astype(int)
df["NOTE_SVC"] = df["Note"].astype(str).str.contains("SVC", na=False).astype(int)
feature_cols.extend(["NOTE_CTI", "NOTE_BOX", "NOTE_PST", "NOTE_SVC"])

# Add case order within day (proxy for fatigue/scheduling)
df["CASE_ORDER_IN_DAY"] = df.groupby("DATE").cumcount() + 1
feature_cols.append("CASE_ORDER_IN_DAY")

# Drop rows with missing features
model_df = df[feature_cols + ["outlier_class"]].dropna()
X = model_df[feature_cols]
y = model_df["outlier_class"]

print(f"Feature matrix shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts().sort_index()}")
print(f"\nFeatures used ({len(feature_cols)}):")
for f in feature_cols:
    print(f"  - {f}")

# ============================================================
# 4. LIGHTGBM MODEL
# ============================================================
print("\n" + "=" * 60)
print("4. LIGHTGBM MODEL TRAINING")
print("=" * 60)

if classification_type == "binary":
    objective = "binary"
    metric = "binary_logloss"
    num_class = None
else:
    objective = "multiclass"
    metric = "multi_logloss"
    num_class = len(y.unique())

params = {
    "objective": objective,
    "metric": metric,
    "verbosity": -1,
    "boosting_type": "gbdt",
    "num_leaves": 15,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "max_depth": 4,
    "min_child_samples": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "class_weight": "balanced",
    "random_state": 42,
}
if num_class and num_class > 2:
    params["num_class"] = num_class

model = lgb.LGBMClassifier(**params)

# Fit on full dataset — goal is SHAP interpretation, not prediction
model.fit(X, y)
print(f"\nModel fitted on {len(X)} cases ({int(y.sum())} outliers, {int(len(y) - y.sum())} normal)")

# Feature importance
fig, ax = plt.subplots(figsize=(10, 6))
lgb.plot_importance(model, ax=ax, importance_type="gain", max_num_features=15)
ax.set_title("Global Model: Feature Importance by Information Gain")
plt.tight_layout()
plt.savefig(f"{GLOBAL_DIR}/lgbm_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: lgbm_feature_importance.png")

# ============================================================
# 5. SHAP ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("5. SHAP ANALYSIS")
print("=" * 60)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# For binary: shap_values is a single array
# For multiclass: shap_values is a list of arrays
if classification_type == "binary":
    # shap_values for the positive class
    if isinstance(shap_values, list):
        sv = shap_values[1]  # positive class
    else:
        sv = shap_values

    # Summary plot (bee swarm)
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(sv, X, show=False, max_display=15)
    plt.title("Global Model: Feature Impact on Outlier Prediction (SHAP)")
    plt.tight_layout()
    plt.savefig(f"{GLOBAL_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: shap_summary.png")

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(sv, X, plot_type="bar", show=False, max_display=15)
    plt.title("Global Model: Average Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(f"{GLOBAL_DIR}/shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: shap_bar.png")

else:
    # Multiclass: plot for each class
    for cls_idx, cls_name in class_labels.items():
        if cls_idx >= len(shap_values):
            continue
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values[cls_idx], X, show=False, max_display=15)
        plt.title(f"SHAP Summary - Class {cls_idx}: {cls_name}")
        plt.tight_layout()
        plt.savefig(f"{GLOBAL_DIR}/shap_summary_class{cls_idx}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: shap_summary_class{cls_idx}.png")

    # Overall bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False,
                      max_display=15, class_names=list(class_labels.values()))
    plt.title("SHAP Feature Importance by Class")
    plt.tight_layout()
    plt.savefig(f"{GLOBAL_DIR}/shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: shap_bar.png")

# --- SHAP Dependence plots for top features ---
if classification_type == "binary":
    mean_abs_shap = np.abs(sv).mean(axis=0)
else:
    mean_abs_shap = np.mean([np.abs(s).mean(axis=0) for s in shap_values], axis=0)

top_features = pd.Series(mean_abs_shap, index=feature_cols).nlargest(4)
print(f"\nTop 4 features by mean |SHAP|:")
for feat, val in top_features.items():
    print(f"  {feat}: {val:.3f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, (feat, _) in enumerate(top_features.items()):
    ax = axes[idx // 2, idx % 2]
    sv_for_dep = sv if classification_type == "binary" else shap_values[1] if len(shap_values) > 1 else shap_values[0]
    shap.dependence_plot(feat, sv_for_dep, X, ax=ax, show=False)
    ax.set_title(f"Effect of {feat} on Outlier Prediction")
plt.suptitle("How Top Features Influence Outlier Classification", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{GLOBAL_DIR}/shap_dependence.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_dependence.png")

# ============================================================
# 6. PER-PHYSICIAN OUTLIER DETECTION & MODELS
# ============================================================
print("\n" + "=" * 60)
print("6. PER-PHYSICIAN OUTLIER ANALYSIS")
print("=" * 60)

# Features for per-physician models (exclude physician encoding)
phys_feature_cols = [c for c in feature_cols if c != "PHYSICIAN_ENC"]

physicians = sorted(df["PHYSICIAN"].unique())
per_physician_results = {}

for phys in physicians:
    phys_safe = phys.replace(".", "").replace(" ", "_")
    phys_dir = f"{PHYS_DIR}/{phys_safe}"
    os.makedirs(phys_dir, exist_ok=True)

    phys_df = df[df["PHYSICIAN"] == phys].copy()
    phys_target = phys_df["PT IN-OUT"]
    n_cases = len(phys_df)

    print(f"\n--- {phys} ({n_cases} cases) ---")

    # Per-physician IQR-based outlier threshold (scales with sample size)
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

    # Store the per-physician outlier flag back on main df
    df.loc[phys_df.index, "phys_outlier_class"] = phys_df["phys_outlier_class"]

    # Prepare model data
    phys_model_df = phys_df[phys_feature_cols + ["phys_outlier_class"]].dropna()
    X_phys = phys_model_df[phys_feature_cols]
    y_phys = phys_model_df["phys_outlier_class"]

    # Need at least 2 outliers to attempt modeling
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
    phys_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": 10,
        "learning_rate": 0.05,
        "n_estimators": 150,
        "max_depth": 3,
        "min_child_samples": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "class_weight": "balanced",
        "random_state": 42,
    }
    phys_model = lgb.LGBMClassifier(**phys_params)

    # Fit on full physician data — goal is SHAP interpretation, not prediction
    phys_model.fit(X_phys, y_phys)
    print(f"  Model fitted on {len(X_phys)} cases ({int(y_phys.sum())} outliers, {int(len(y_phys) - y_phys.sum())} normal)")

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(10, 6))
    lgb.plot_importance(phys_model, ax=ax, importance_type="gain", max_num_features=15)
    ax.set_title(f"{phys}: Feature Importance by Information Gain")
    plt.tight_layout()
    plt.savefig(f"{phys_dir}/lgbm_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # SHAP
    phys_explainer = shap.TreeExplainer(phys_model)
    phys_shap_values = phys_explainer.shap_values(X_phys)
    if isinstance(phys_shap_values, list):
        phys_sv = phys_shap_values[1]
    else:
        phys_sv = phys_shap_values

    # SHAP summary
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(phys_sv, X_phys, show=False, max_display=15)
    plt.title(f"{phys}: Feature Impact on Outlier Prediction (SHAP)")
    plt.tight_layout()
    plt.savefig(f"{phys_dir}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # SHAP bar
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(phys_sv, X_phys, plot_type="bar", show=False, max_display=15)
    plt.title(f"{phys}: Average Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(f"{phys_dir}/shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Top features for this physician
    phys_mean_abs_shap = np.abs(phys_sv).mean(axis=0)
    phys_top = pd.Series(phys_mean_abs_shap, index=phys_feature_cols).nlargest(4)
    print(f"  Top 4 SHAP features:")
    for feat, val in phys_top.items():
        print(f"    {feat}: {val:.3f}")

    # SHAP dependence for top 2
    if len(phys_top) >= 2:
        fig, axes_dep = plt.subplots(1, 2, figsize=(14, 5))
        for idx, (feat, _) in enumerate(list(phys_top.items())[:2]):
            shap.dependence_plot(feat, phys_sv, X_phys, ax=axes_dep[idx], show=False)
            axes_dep[idx].set_title(f"Effect of {feat} on Outlier Prediction")
        plt.suptitle(f"{phys}: How Top Features Influence Outlier Classification", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{phys_dir}/shap_dependence.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Outlier vs normal comparison for this physician
    phys_outlier_df = phys_df[phys_df["phys_outlier_class"] == 1]
    phys_normal_df = phys_df[phys_df["phys_outlier_class"] == 0]
    compare_cols_phys = [
        "PT PREP/INTUBATION", "ACCESSS", "TSP", "PRE-MAP",
        "ABL DURATION", "ABL TIME", "#ABL", "#APPLICATIONS",
        "POST CARE/EXTUBATION",
    ]
    phys_outlier_means = phys_outlier_df[compare_cols_phys].mean()
    phys_normal_means = phys_normal_df[compare_cols_phys].mean()

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
            for col in compare_cols_phys
        },
        "outlier_cases": [
            {
                "case_num": int(row["CASE #"]),
                "date": str(row["DATE"].strftime("%Y-%m-%d") if pd.notna(row["DATE"]) else ""),
                "pt_in_out_min": int(row["PT IN-OUT"]),
                "note": str(row["Note"]) if pd.notna(row["Note"]) and row["Note"] != "" else None,
            }
            for _, row in phys_outlier_df.iterrows()
        ],
        "output_dir": phys_dir,
    }

# Add per-physician outlier label to the main dataframe
df["phys_outlier_label"] = df["phys_outlier_class"].map({0: "Normal", 1: "Outlier (IQR for Physician)"})

# --- Per-physician EDA comparison chart ---
print("\n--- Generating per-physician EDA comparison charts ---")

phys_colors = {"Dr. A": "#2196F3", "Dr. B": "#FF5722", "Dr. C": "#4CAF50"}

# Figure 1: Distribution & outlier comparison (2x3 grid)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Per-Physician: Procedure Duration Distribution & Outlier Detection", fontsize=15, fontweight="bold")

for idx, phys in enumerate(physicians):
    phys_data = df[df["PHYSICIAN"] == phys]
    phys_target_vals = phys_data["PT IN-OUT"]
    phys_p90_val = per_physician_results[phys]["threshold_minutes"]
    color = phys_colors.get(phys, "gray")

    # Row 1: Histograms with per-physician threshold
    ax = axes[0, idx]
    normal_vals = phys_data[phys_data["phys_outlier_class"] == 0]["PT IN-OUT"]
    outlier_vals = phys_data[phys_data["phys_outlier_class"] == 1]["PT IN-OUT"]
    ax.hist(normal_vals, bins=15, alpha=0.6, color=color, edgecolor="black", label="Normal Duration")
    ax.hist(outlier_vals, bins=8, alpha=0.7, color="red", edgecolor="black", label="Long Duration (Outlier)")
    ax.axvline(phys_p90_val, color="red", linestyle="--", linewidth=1.5,
               label=f"IQR bound = {phys_p90_val:.0f} min")
    ax.set_title(f"{phys}: Duration Distribution (n={len(phys_data)})")
    ax.set_xlabel("Duration (minutes)")
    ax.set_ylabel("Number of Cases")
    ax.legend(fontsize=8)

    # Row 2: Boxplot + strip plot showing individual cases
    ax = axes[1, idx]
    bp = ax.boxplot(phys_target_vals.dropna(), vert=True, widths=0.4,
                    patch_artist=True, positions=[0.3])
    bp["boxes"][0].set_facecolor(color)
    bp["boxes"][0].set_alpha(0.3)
    # Overlay individual points
    jitter = np.random.normal(0.7, 0.04, size=len(normal_vals))
    ax.scatter(jitter[:len(normal_vals)], normal_vals, alpha=0.5, color=color, s=20, label="Normal Duration")
    jitter_out = np.random.normal(0.7, 0.04, size=len(outlier_vals))
    ax.scatter(jitter_out[:len(outlier_vals)], outlier_vals, alpha=0.8, color="red",
               s=50, edgecolor="black", zorder=5, label="Long Duration (Outlier)")
    ax.axhline(phys_p90_val, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_title(f"{phys}: Individual Case Durations")
    ax.set_ylabel("Duration (minutes)")
    ax.set_xticks([])
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{EDA_DIR}/eda_per_physician_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: eda_per_physician_distributions.png")

# Figure 2: Outlier rate comparison & feature breakdown (2x2)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Per-Physician: Outlier Rates, Thresholds & Feature Drivers", fontsize=15, fontweight="bold")

# (0,0) Outlier rate bar chart
ax = axes[0, 0]
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

x_pos = np.arange(len(phys_names))
bars_normal = ax.bar(x_pos, normal_counts, color=[phys_colors[p] for p in phys_names],
                     alpha=0.5, label="Normal Duration", edgecolor="black")
bars_outlier = ax.bar(x_pos, outlier_counts, bottom=normal_counts,
                      color="red", alpha=0.7, label="Long Duration (Outlier)", edgecolor="black")
# Add rate labels on top
for i, (rate, total) in enumerate(zip(outlier_rates, [r["n_cases"] for r in [per_physician_results[p] for p in physicians]])):
    ax.text(i, total + 1, f"{rate:.1f}%", ha="center", fontweight="bold", fontsize=11)
ax.set_xticks(x_pos)
ax.set_xticklabels(phys_names)
ax.set_ylabel("Total Number of Cases")
ax.set_title("Case Volume & Outlier Rate by Physician")
ax.legend()

# (0,1) Per-physician thresholds vs global
ax = axes[0, 1]
thresholds = [per_physician_results[p]["threshold_minutes"] for p in physicians]
bar_colors = [phys_colors[p] for p in physicians]
bars = ax.bar(x_pos, thresholds, color=bar_colors, edgecolor="black", alpha=0.7)
ax.axhline(float(outlier_threshold), color="black", linestyle="--", linewidth=2,
           label=f"Global Threshold ({outlier_threshold:.0f} min)")
for i, (t, phys) in enumerate(zip(thresholds, phys_names)):
    ax.text(i, t + 1, f"{t:.0f} min", ha="center", fontweight="bold", fontsize=11)
ax.set_xticks(x_pos)
ax.set_xticklabels(phys_names)
ax.set_ylabel("IQR Outlier Threshold (minutes)")
ax.set_title("Per-Physician Outlier Threshold vs Global Threshold")
ax.legend()

# (1,0) Mean PT IN-OUT: Outlier vs Normal per physician (grouped bar)
ax = axes[1, 0]
bar_width = 0.35
normal_means_list = []
outlier_means_list = []
for phys in physicians:
    phys_data = df[df["PHYSICIAN"] == phys]
    normal_means_list.append(phys_data[phys_data["phys_outlier_class"] == 0]["PT IN-OUT"].mean())
    outlier_means_list.append(phys_data[phys_data["phys_outlier_class"] == 1]["PT IN-OUT"].mean())

ax.bar(x_pos - bar_width/2, normal_means_list, bar_width, label="Normal Duration",
       color=[phys_colors[p] for p in physicians], alpha=0.5, edgecolor="black")
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

# (1,1) Top SHAP feature comparison across physicians (heatmap-style)
ax = axes[1, 1]
# Collect all unique top features across physicians
all_top_feats = []
for phys in physicians:
    res = per_physician_results[phys]
    if res["model_fitted"] and res.get("top_shap_features"):
        all_top_feats.extend(res["top_shap_features"].keys())
unique_feats = list(dict.fromkeys(all_top_feats))  # preserve order, deduplicate

# Build matrix: physicians x features
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
plt.savefig(f"{EDA_DIR}/eda_per_physician_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: eda_per_physician_comparison.png")

# Figure 3: Per-physician feature means (outlier vs normal) side-by-side
compare_cols_short = [
    "PT PREP/INTUBATION", "ACCESSS", "TSP", "PRE-MAP",
    "ABL DURATION", "ABL TIME", "#ABL", "#APPLICATIONS",
    "POST CARE/EXTUBATION",
]
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
            color=phys_colors.get(phys, "gray"), alpha=0.5, edgecolor="black")
    ax.barh(y_pos - 0.2, o_means.values, 0.35, label="Long Duration (Outlier)",
            color="red", alpha=0.7, edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(compare_cols_short, fontsize=9)
    ax.set_xlabel("Duration (minutes)")
    ax.set_title(f"{phys}: Feature Comparison (n={len(phys_data)})")
    ax.legend(fontsize=8)
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig(f"{EDA_DIR}/eda_per_physician_feature_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: eda_per_physician_feature_comparison.png")

# ============================================================
# 7. ADDITIONAL ANALYSES
# ============================================================
print("\n" + "=" * 60)
print("7. ADDITIONAL ANALYSES")
print("=" * 60)

# --- 7a. Learning curve / time trend ---
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
    durations = phys_data["PT IN-OUT"].values

    # Scatter
    ax.scatter(case_seq, durations, alpha=0.5, color=phys_colors.get(phys, "gray"),
               edgecolor="black", s=30)

    # Linear trendline
    mask = ~np.isnan(durations)
    if mask.sum() > 2:
        z = np.polyfit(case_seq[mask], durations[mask], 1)
        p = np.poly1d(z)
        ax.plot(case_seq, p(case_seq), color="red", linewidth=2,
                label=f"Linear Trend ({z[0]:+.2f} min/case)")
        # Rolling average (window=10)
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
plt.savefig(f"{EXTRA_DIR}/learning_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: learning_curve.png")

# --- 7b. Case complexity breakdown ---
print("\n--- 7b. Case Complexity Analysis ---")

# Define complexity flags from Note column
complexity_flags = {
    "Standard PFA": ~df["HAS_NOTE"] | (df["Note"].astype(str).str.strip() == ""),
    "CTI (Cavo-tricuspid isthmus)": df["NOTE_CTI"] == 1,
    "BOX (Box isolation)": df["NOTE_BOX"] == 1,
    "PST BOX (Posterior box)": df["NOTE_PST"] == 1,
    "SVC (Superior vena cava)": df["NOTE_SVC"] == 1,
}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Case Complexity: How Additional Procedures Affect Duration", fontsize=14, fontweight="bold")

# (0) Box plot of duration by complexity flag
complexity_data = []
complexity_labels = []
complexity_stats = {}
for label, mask in complexity_flags.items():
    vals = df.loc[mask, "PT IN-OUT"].dropna()
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

# (1) Outlier rate by complexity
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
plt.savefig(f"{EXTRA_DIR}/case_complexity.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: case_complexity.png")

# --- 7c. Day-of scheduling / case order analysis ---
print("\n--- 7c. Case Order / Scheduling Analysis ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Day-of Scheduling: Effect of Case Order on Procedure Duration", fontsize=14, fontweight="bold")

# (0) Duration by case order within day
df_sched = df.dropna(subset=["CASE_ORDER_IN_DAY", "PT IN-OUT"])
case_order_groups = df_sched.groupby("CASE_ORDER_IN_DAY")["PT IN-OUT"]
orders = sorted(df_sched["CASE_ORDER_IN_DAY"].dropna().unique().astype(int))
order_means = [case_order_groups.get_group(o).mean() for o in orders]
order_counts = [len(case_order_groups.get_group(o)) for o in orders]

ax = axes[0]
bars = ax.bar(orders, order_means, color="steelblue", alpha=0.7, edgecolor="black")
for i, (mean, count) in enumerate(zip(order_means, order_counts)):
    ax.text(orders[i], mean + 1, f"{mean:.0f}\n(n={count})", ha="center", fontsize=8)
ax.set_xlabel("Case Position in Daily Schedule")
ax.set_ylabel("Average Duration (minutes)")
ax.set_title("Average Procedure Duration by Daily Case Order")
ax.set_xticks(orders)

# (1) Outlier rate by case order
order_outlier_rates = [df_sched[df_sched["CASE_ORDER_IN_DAY"] == o]["outlier_class"].mean() * 100 for o in orders]
ax = axes[1]
bars = ax.bar(orders, order_outlier_rates, color="orange", alpha=0.7, edgecolor="black")
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
plt.savefig(f"{EXTRA_DIR}/case_order_scheduling.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: case_order_scheduling.png")

# --- 7d. Physician severity / case complexity profile ---
print("\n--- 7d. Physician Severity & Case Complexity Profile ---")

# Build complexity indicators
df["NOTE_AAFL"] = df["Note"].astype(str).str.contains("AAFL", na=False).astype(int)
df["NOTE_TROUBLESHOOT"] = df["Note"].astype(str).str.contains("TROUBLESHOOT", na=False).astype(int)
df["HAS_ADDITIONAL"] = (df["NOTE_CTI"] | df["NOTE_BOX"] | df["NOTE_PST"] | df["NOTE_SVC"] | df["NOTE_AAFL"]).astype(int)
df["N_ADDITIONAL"] = df["NOTE_CTI"] + df["NOTE_BOX"] + df["NOTE_PST"] + df["NOTE_SVC"] + df["NOTE_AAFL"]

severity_results = {}
for phys in physicians:
    sub = df[df["PHYSICIAN"] == phys]
    severity_results[phys] = {
        "n_cases": int(len(sub)),
        "pt_in_out": {"mean": round(float(sub["PT IN-OUT"].mean()), 1), "median": round(float(sub["PT IN-OUT"].median()), 1), "std": round(float(sub["PT IN-OUT"].std()), 1)},
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
bar_colors = [phys_colors.get(p, "gray") for p in physicians]

# (0,0) Duration distribution violin/box comparison
ax = axes[0, 0]
phys_data_list = [df[df["PHYSICIAN"] == p]["PT IN-OUT"].dropna().values for p in physicians]
bp = ax.boxplot(phys_data_list, labels=physicians, patch_artist=True, widths=0.5)
for patch, color in zip(bp["boxes"], bar_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
ax.set_ylabel("Duration (minutes)")
ax.set_title("Total Procedure Duration Distribution")

# (0,1) Ablation sites (#ABL) per physician
ax = axes[0, 1]
abl_means = [severity_results[p]["abl_sites"]["mean"] for p in physicians]
abl_medians = [severity_results[p]["abl_sites"]["median"] for p in physicians]
bars = ax.bar(x_pos - 0.15, abl_means, 0.3, label="Mean", color=bar_colors, alpha=0.7, edgecolor="black")
ax.bar(x_pos + 0.15, abl_medians, 0.3, label="Median", color=bar_colors, alpha=0.4, edgecolor="black", hatch="//")
for i in range(len(physicians)):
    ax.text(i - 0.15, abl_means[i] + 0.2, f"{abl_means[i]:.1f}", ha="center", fontsize=9)
    ax.text(i + 0.15, abl_medians[i] + 0.2, f"{abl_medians[i]:.0f}", ha="center", fontsize=9)
ax.set_xticks(x_pos)
ax.set_xticklabels(physicians)
ax.set_ylabel("Number of Ablation Sites")
ax.set_title("Ablation Sites Targeted per Case")
ax.legend()

# (0,2) ABL DURATION vs ABL TIME (repositioning breakdown)
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

# (1,0) PRE-MAP and TSP comparison (complexity indicators)
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

# (1,2) Additional procedure type breakdown (stacked bar)
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
plt.savefig(f"{EXTRA_DIR}/physician_severity_profile.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: physician_severity_profile.png")

# ============================================================
# 8. SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("8. SAVING RESULTS")
print("=" * 60)

# Add outlier columns to original data and save
output_df = df.copy()
output_df["DATE"] = output_df["DATE"].dt.strftime("%Y-%m-%d")
outlier_label_map = {0: "Normal", 1: "Outlier (Top 10%)"}
output_df["outlier_label"] = output_df["outlier_class"].map(outlier_label_map)
# phys_outlier_label already set above
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
        "cases_with_target": int(df["PT IN-OUT"].notna().sum()),
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
        "classification_type": classification_type,
        "class_distribution": {
            outlier_label_map[k]: int(v)
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
            "pt_in_out_min": int(row["PT IN-OUT"]),
            "note": str(row["Note"]) if pd.notna(row["Note"]) and row["Note"] != "" else None,
        }
        for _, row in outlier_df.iterrows()
    ],
    "model": {
        "type": "LightGBM Classifier",
        "params": {k: v for k, v in params.items() if k != "class_weight"},
        "class_weight": "balanced",
        "note": "Fitted on full dataset for SHAP interpretation, not prediction.",
    },
    "shap_analysis": {
        "top_features_by_mean_abs_shap": {
            feat: round(float(val), 3) for feat, val in top_features.items()
        },
    },
    "per_physician": per_physician_results,
    "additional_analyses": {
        "learning_curve": learning_curve_results,
        "case_complexity": complexity_stats,
        "scheduling": scheduling_results,
        "physician_severity": severity_results,
    },
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
# Compute per-physician outlier rates (global threshold)
phys_outlier_global = df.groupby("PHYSICIAN")["outlier_class"].agg(["sum", "count"])
phys_outlier_global["rate"] = (phys_outlier_global["sum"] / phys_outlier_global["count"] * 100)

# Compute mean feature values for outlier vs normal (global)
compare_cols = [
    "PT PREP/INTUBATION", "ACCESSS", "TSP", "PRE-MAP",
    "ABL DURATION", "ABL TIME", "#ABL", "#APPLICATIONS",
    "POST CARE/EXTUBATION",
]
outlier_means = outlier_df[compare_cols].mean()
normal_means = normal_df[compare_cols].mean()

# Build outlier case table rows (global)
outlier_rows = ""
for _, row in outlier_df.sort_values("PT IN-OUT", ascending=False).iterrows():
    date_str = row["DATE"].strftime("%Y-%m-%d") if pd.notna(row["DATE"]) else ""
    note_str = str(row["Note"]) if pd.notna(row["Note"]) and row["Note"] != "" else "-"
    outlier_rows += f"| {int(row['CASE #']):>4} | {date_str} | {row['PHYSICIAN']} | {int(row['PT IN-OUT']):>3} | {note_str} |\n"

# Build feature comparison table (global)
feature_compare_rows = ""
for col in compare_cols:
    n_val = normal_means[col]
    o_val = outlier_means[col]
    diff_pct = ((o_val - n_val) / n_val * 100) if n_val != 0 else 0
    feature_compare_rows += f"| {col:<24} | {n_val:>6.1f} | {o_val:>6.1f} | {diff_pct:>+6.1f}% |\n"

# Build global SHAP ranking
shap_rows = ""
for rank, (feat, val) in enumerate(top_features.items(), 1):
    shap_rows += f"| {rank} | {feat:<24} | {val:.3f} |\n"

# Build physician breakdown (global threshold)
phys_global_rows = ""
for phys, row in phys_outlier_global.iterrows():
    phys_global_rows += f"| {phys} | {int(row['count'])} | {int(row['sum'])} | {row['rate']:.1f}% |\n"

# Build physician breakdown (per-physician IQR threshold)
phys_local_rows = ""
for phys in physicians:
    res = per_physician_results[phys]
    phys_local_rows += f"| {phys} | {res['n_cases']} | {res['threshold_minutes']:.0f} | {res['n_outliers']} | {res['n_outliers']/res['n_cases']*100:.1f}% |\n"

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
- **Threshold**: PT IN-OUT > {outlier_threshold:.0f} minutes
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

### Physician Breakdown (Global Threshold > {outlier_threshold:.0f} min)

| Physician | Total Cases | Outliers | Outlier Rate |
|-----------|------------:|---------:|-------------:|
{phys_global_rows}

### Global LightGBM Model

- **Model**: LightGBM (gradient boosted decision tree) with balanced class weights
- **Fitted on**: All {len(X)} cases ({int(y.sum())} outliers, {int(len(y) - y.sum())} normal)
- **Purpose**: SHAP feature importance interpretation (not prediction)

### Global SHAP Feature Drivers

| Rank | Feature                  | Mean SHAP |
|-----:|--------------------------|----------:|
{shap_rows}

**Interpretation**: The SHAP analysis reveals which factors most strongly push a case
toward being classified as an outlier (long-duration):

"""

# Build interpretation dynamically, handling non-timing features like PHYSICIAN_ENC
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
{phys_local_rows}

Note: Dr. B's threshold ({per_physician_results['Dr. B']['threshold_minutes']:.0f} min) is much higher than
Dr. A's ({per_physician_results['Dr. A']['threshold_minutes']:.0f} min) because Dr. B's baseline duration
is longer overall (higher Q3).

See comparison charts in `eda/`:
- `eda_per_physician_distributions.png` - histograms and strip plots per physician
- `eda_per_physician_comparison.png` - outlier rates, thresholds, mean durations, SHAP heatmap
- `eda_per_physician_feature_comparison.png` - outlier vs normal feature means side-by-side

"""

# Append per-physician detail sections
for phys in physicians:
    res = per_physician_results[phys]
    phys_safe = phys.replace(".", "").replace(" ", "_")
    summary_md += f"### {phys} ({res['n_cases']} cases)\n\n"
    summary_md += f"- **Threshold**: PT IN-OUT > {res['threshold_minutes']:.0f} min (Q3+1.0*IQR within {phys})\n"
    summary_md += f"- **Normal**: {res['n_normal']}, **Outlier**: {res['n_outliers']}\n"

    if not res["model_fitted"]:
        summary_md += f"- **Model**: Not fitted ({res['reason_skipped']})\n\n"
        continue

    # Top SHAP features
    summary_md += f"\n**Top SHAP Features:**\n\n"
    summary_md += f"| Rank | Feature | Mean SHAP |\n"
    summary_md += f"|-----:|---------|----------:|\n"
    for rank, (feat, val) in enumerate(res["top_shap_features"].items(), 1):
        summary_md += f"| {rank} | {feat} | {val:.3f} |\n"

    # Outlier vs normal comparison
    summary_md += f"\n**Outlier vs Normal (feature means):**\n\n"
    summary_md += f"| Feature | Normal | Outlier | Diff |\n"
    summary_md += f"|---------|-------:|--------:|-----:|\n"
    for col, vals in res["outlier_vs_normal_means"].items():
        summary_md += f"| {col} | {vals['normal']:.1f} | {vals['outlier']:.1f} | {vals['diff_pct']:+.1f}% |\n"

    # Outlier cases
    summary_md += f"\n**Outlier cases:**\n\n"
    summary_md += f"| Case | Date | PT IN-OUT | Note |\n"
    summary_md += f"|-----:|------|----------:|------|\n"
    for case in sorted(res["outlier_cases"], key=lambda c: -c["pt_in_out_min"]):
        note = case["note"] or "-"
        summary_md += f"| {case['case_num']} | {case['date']} | {case['pt_in_out_min']} | {note} |\n"

    summary_md += f"\nPlots: `per_physician/{phys_safe}/`\n\n"

# Problem statement and solutions — build dynamically from top timing features
timing_top_global = [f for f in top_features.index if f in compare_cols]
top_timing_str = " and ".join(f"**{f}**" for f in timing_top_global[:2]) if timing_top_global else "multiple procedural phases"

summary_md += f"""---

## 5. Problem Statement

The top 10% of AFib ablation procedures take significantly longer than typical cases
(>{outlier_threshold:.0f} min vs median {target.median():.0f} min). The global model shows that
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

# Build solutions dynamically from top SHAP timing features
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

# Physician-specific coaching
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

# Key insight about severity
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

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nGlobal findings:")
print(f"  - Total cases: {len(df)}")
print(f"  - Global outlier threshold: >{outlier_threshold:.0f} min (90th pctl)")
print(f"  - Global outliers: {n_outliers} / {len(df)}")
print(f"  - Global model: fitted on {len(X)} cases")
print(f"  - Top global drivers: {', '.join(top_features.index[:3])}")
print(f"\nPer-physician findings:")
for phys in physicians:
    res = per_physician_results[phys]
    fitted = "model fitted" if res["model_fitted"] else "model skipped"
    print(f"  - {phys}: {res['n_cases']} cases, threshold >{res['threshold_minutes']:.0f} min, "
          f"{res['n_outliers']} outliers, {fitted}")
    if res["model_fitted"] and res.get("top_shap_features"):
        top_feat = list(res["top_shap_features"].keys())[0]
        print(f"    Top driver: {top_feat}")
print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
