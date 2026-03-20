"""
Export analysis data as JSON for the React dashboard.
Loads the saved model artifacts and existing output files, then writes dashboard_data.json.
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

# Add project root to path so we can import src.config
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUT_DIR, DATA_PATH, BASE_DIR
import joblib


def main() -> None:
    # ── Load saved model artifacts ──
    model_path = OUTPUT_DIR / "model" / "global_model.pkl"
    artifacts = joblib.load(model_path)
    model = artifacts['model']
    sv = artifacts['shap_values']
    X = artifacts['X']
    y = artifacts['y']
    feature_cols = artifacts['feature_cols']

    # ── Load existing outputs ──
    REPORT_PATH = OUTPUT_DIR / "analysis_report.json"
    CSV_PATH = OUTPUT_DIR / "MSE433_M4_Data_with_outliers.csv"
    OUT_PATH = PROJECT_ROOT / "app" / "frontend" / "src" / "data" / "dashboard_data.json"

    with open(REPORT_PATH) as f:
        report = json.load(f)

    df = pd.read_csv(CSV_PATH)

    # ── Derived values from saved artifacts ──
    mean_abs_shap = np.abs(sv).mean(axis=0)
    feature_importance_gain = model.feature_importances_

    # ── 1. cases ──
    cases = []
    for _, row in df.iterrows():
        case = {
            "caseNum": int(row["CASE #"]),
            "date": str(row["DATE"]),
            "physician": row["PHYSICIAN"],
            "ptInOut": float(row["PT IN-OUT"]),
            "note": row["Note"] if pd.notna(row.get("Note")) else None,
            "outlierClass": int(row["outlier_class"]),
            "outlierLabel": row.get("outlier_label", "Normal"),
            "physOutlierClass": int(row["phys_outlier_class"]) if pd.notna(row.get("phys_outlier_class")) else 0,
            "physOutlierLabel": row.get("phys_outlier_label", "Normal"),
            "features": {},
        }
        for feat in feature_cols:
            val = row.get(feat)
            case["features"][feat] = float(val) if pd.notna(val) else None
        cases.append(case)

    # ── 2. globalModel ──
    global_model = {
        "featureImportance": [],
        "threshold": report["outlier_detection"]["threshold_minutes"],
        "params": report["model"]["params"],
    }
    for i, feat in enumerate(feature_cols):
        global_model["featureImportance"].append({
            "feature": feat,
            "importance": float(feature_importance_gain[i]),
            "shapMean": round(float(mean_abs_shap[i]), 4),
        })
    global_model["featureImportance"].sort(key=lambda x: x["shapMean"], reverse=True)

    # ── 3. physicians ──
    physicians = {}
    for phys_name, phys_data in report["per_physician"].items():
        phys_cases = df[df["PHYSICIAN"] == phys_name]
        physicians[phys_name] = {
            "caseCount": int(phys_data["n_cases"]),
            "outlierCount": int(phys_data["n_outliers"]),
            "meanDuration": round(float(phys_cases["PT IN-OUT"].mean()), 1),
            "medianDuration": round(float(phys_cases["PT IN-OUT"].median()), 1),
            "iqrThreshold": phys_data["threshold_minutes"],
            "topDrivers": phys_data.get("top_shap_features", {}),
            "Q1": phys_data.get("Q1"),
            "Q3": phys_data.get("Q3"),
            "IQR": phys_data.get("IQR"),
            "modelFitted": phys_data.get("model_fitted", False),
        }

    # ── 4. shapValues (matrix: cases x features) ──
    shap_matrix = sv.tolist()

    # ── 5. featureStats ──
    feature_stats = {}
    for feat in feature_cols:
        col = df[feat].dropna()
        feature_stats[feat] = {
            "mean": round(float(col.mean()), 2),
            "std": round(float(col.std()), 2),
            "min": round(float(col.min()), 2),
            "max": round(float(col.max()), 2),
            "median": round(float(col.median()), 2),
            "q25": round(float(col.quantile(0.25)), 2),
            "q75": round(float(col.quantile(0.75)), 2),
        }

    # ── 6. distributions ──
    target = df["PT IN-OUT"]
    bins = np.linspace(float(target.min()), float(target.max()), 21)
    bin_labels = [round(float(b), 1) for b in bins]

    overall_counts, _ = np.histogram(target, bins=bins)
    distributions = {
        "overall": {
            "bins": bin_labels,
            "counts": [int(c) for c in overall_counts],
        },
        "byPhysician": {},
    }
    for phys in df["PHYSICIAN"].unique():
        phys_target = df[df["PHYSICIAN"] == phys]["PT IN-OUT"]
        counts, _ = np.histogram(phys_target, bins=bins)
        distributions["byPhysician"][phys] = {
            "bins": bin_labels,
            "counts": [int(c) for c in counts],
        }

    # ── 7. trends ──
    learning_curve = []
    for _, row in df.iterrows():
        learning_curve.append({
            "caseNum": int(row["CASE #"]),
            "duration": float(row["PT IN-OUT"]),
            "physician": row["PHYSICIAN"],
        })

    scheduling_effects = []
    for _, row in df.iterrows():
        co = row.get("CASE_ORDER_IN_DAY")
        if pd.notna(co):
            scheduling_effects.append({
                "caseOrder": int(co),
                "duration": float(row["PT IN-OUT"]),
                "physician": row["PHYSICIAN"],
            })

    trends = {
        "learningCurve": learning_curve,
        "schedulingEffects": scheduling_effects,
        "learningCurveStats": report["additional_analyses"]["learning_curve"],
        "schedulingStats": report["additional_analyses"]["scheduling"],
    }

    # ── 8. complexity ──
    complexity_data = report["additional_analyses"]["case_complexity"]
    procedure_types = []
    for ptype, pdata in complexity_data.items():
        procedure_types.append({
            "type": ptype,
            "totalCases": pdata["n_cases"],
            "outlierCases": round(pdata["n_cases"] * pdata["outlier_rate_global_pct"] / 100),
            "outlierRate": pdata["outlier_rate_global_pct"],
            "meanDuration": pdata["mean_duration"],
            "medianDuration": pdata["median_duration"],
        })
    complexity = {"procedureTypes": procedure_types}

    # ── 9. metadata ──
    target_stats = report["target_variable"]
    metadata = {
        "totalCases": report["dataset"]["total_cases"],
        "outlierCount": report["outlier_detection"]["class_distribution"]["Outlier (Top 10%)"],
        "threshold": report["outlier_detection"]["threshold_minutes"],
        "skewness": target_stats["skewness"],
        "kurtosis": target_stats["kurtosis"],
        "meanDuration": target_stats["mean"],
        "medianDuration": target_stats["median"],
        "stdDuration": target_stats["std"],
        "minDuration": target_stats["min"],
        "maxDuration": target_stats["max"],
        "dateGenerated": pd.Timestamp.now().isoformat(),
        "featuresUsed": feature_cols,
        "physicianSeverity": report["additional_analyses"]["physician_severity"],
    }

    # ── Assemble and write ──
    dashboard_data = {
        "cases": cases,
        "globalModel": global_model,
        "physicians": physicians,
        "shapValues": shap_matrix,
        "featureStats": feature_stats,
        "distributions": distributions,
        "trends": trends,
        "complexity": complexity,
        "metadata": metadata,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(dashboard_data, f, indent=2, default=str)

    print(f"Dashboard data exported to {OUT_PATH}")
    print(f"  Cases: {len(cases)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  SHAP matrix: {len(shap_matrix)} x {len(shap_matrix[0])}")
    print(f"  Physicians: {list(physicians.keys())}")


if __name__ == "__main__":
    main()
