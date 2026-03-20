"""
Pre-compute patient reassignment data for the React dashboard.

For every case, computes outlier probability under each physician (Dr A/B/C),
batch reassignment scenarios, and optimal assignment to minimize total outliers.
Exports reassignment_data.json consumed by the frontend.
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", category=UserWarning, module="shap")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUT_DIR
import joblib


PHYSICIAN_NAMES = {0: "Dr. A", 1: "Dr. B", 2: "Dr. C"}
OUTLIER_THRESHOLD = 0.5


def main() -> None:
    # ── Load model artifacts ──
    model_path = OUTPUT_DIR / "model" / "global_model.pkl"
    artifacts = joblib.load(model_path)
    model = artifacts["model"]
    X = artifacts["X"].copy()
    y = artifacts["y"].copy()
    feature_cols = artifacts["feature_cols"]

    # Load CSV for case metadata (case numbers, physician names, PT IN-OUT)
    csv_path = OUTPUT_DIR / "MSE433_M4_Data_with_outliers.csv"
    df = pd.read_csv(csv_path)

    explainer = shap.TreeExplainer(model)

    # ── 1. Per-case reassignment predictions ──
    # Use case_nums from model artifacts for safe alignment (no positional indexing)
    case_nums = artifacts.get("case_nums", None)
    df_by_case = df.set_index("CASE #")
    case_reassignments = []

    for i in range(len(X)):
        row_x = X.iloc[i]
        case_num = int(case_nums[i]) if case_nums is not None else int(df.iloc[i]["CASE #"])
        csv_row = df_by_case.loc[case_num]

        original_physician = csv_row["PHYSICIAN"]
        original_enc = int(row_x["PHYSICIAN_ENC"])
        pt_in_out = float(csv_row["PT IN-OUT"])
        outlier_class = int(y.iloc[i])

        features_dict = {feat: round(float(row_x[feat]), 4) for feat in feature_cols}

        predictions = {}
        for phys_enc in [0, 1, 2]:
            # Swap physician
            modified = pd.DataFrame([row_x.values], columns=feature_cols)
            modified["PHYSICIAN_ENC"] = phys_enc

            prob = float(model.predict_proba(modified)[0, 1])

            sv = explainer.shap_values(modified)
            if isinstance(sv, list):
                sv_arr = sv[1][0]
            else:
                sv_arr = sv[0]

            # Top 5 features by absolute SHAP
            shap_dict = {feat: float(sv_arr[i]) for i, feat in enumerate(feature_cols)}
            sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
            top5 = {k: round(v, 4) for k, v in sorted_shap[:5]}

            predictions[str(phys_enc)] = {
                "probability": round(prob, 4),
                "shapTop5": top5,
            }

        # Optimal physician = lowest probability
        optimal_enc = min(predictions, key=lambda k: predictions[k]["probability"])
        optimal_enc_int = int(optimal_enc)

        case_reassignments.append({
            "caseNum": case_num,
            "originalPhysician": original_physician,
            "originalPhysicianEnc": original_enc,
            "ptInOut": pt_in_out,
            "outlierClass": outlier_class,
            "features": features_dict,
            "predictions": predictions,
            "optimalPhysician": optimal_enc_int,
            "optimalProbability": round(predictions[optimal_enc]["probability"], 4),
        })

    print(f"Computed predictions for {len(case_reassignments)} cases x 3 physicians")

    # ── 2. Batch reassignment scenarios ──
    batch_scenarios = []
    physicians = [0, 1, 2]

    for source in physicians:
        for target in physicians:
            if source == target:
                continue

            # Find outlier cases belonging to source physician
            source_outliers = [
                c for c in case_reassignments
                if c["originalPhysicianEnc"] == source and c["outlierClass"] == 1
            ]

            if not source_outliers:
                batch_scenarios.append({
                    "sourcePhysician": PHYSICIAN_NAMES[source],
                    "targetPhysician": PHYSICIAN_NAMES[target],
                    "affectedCases": 0,
                    "outliersResolved": 0,
                    "outliersRemaining": 0,
                    "cases": [],
                })
                continue

            cases_detail = []
            resolved = 0
            for c in source_outliers:
                before_prob = c["predictions"][str(source)]["probability"]
                after_prob = c["predictions"][str(target)]["probability"]
                is_resolved = after_prob < OUTLIER_THRESHOLD
                if is_resolved:
                    resolved += 1
                cases_detail.append({
                    "caseNum": c["caseNum"],
                    "beforeProb": round(before_prob, 4),
                    "afterProb": round(after_prob, 4),
                    "resolved": is_resolved,
                })

            batch_scenarios.append({
                "sourcePhysician": PHYSICIAN_NAMES[source],
                "targetPhysician": PHYSICIAN_NAMES[target],
                "affectedCases": len(source_outliers),
                "outliersResolved": resolved,
                "outliersRemaining": len(source_outliers) - resolved,
                "cases": sorted(cases_detail, key=lambda x: x["beforeProb"], reverse=True),
            })

    print(f"Computed {len(batch_scenarios)} batch reassignment scenarios")

    # ── 3. Optimal assignment ──
    current_outliers = sum(1 for c in case_reassignments if c["outlierClass"] == 1)
    optimized_outliers = sum(
        1 for c in case_reassignments if c["optimalProbability"] >= OUTLIER_THRESHOLD
    )

    changes = []
    for c in case_reassignments:
        if c["optimalPhysician"] != c["originalPhysicianEnc"]:
            changes.append({
                "caseNum": c["caseNum"],
                "fromPhysician": c["originalPhysician"],
                "toPhysician": PHYSICIAN_NAMES[c["optimalPhysician"]],
                "beforeProb": round(
                    c["predictions"][str(c["originalPhysicianEnc"])]["probability"], 4
                ),
                "afterProb": round(c["optimalProbability"], 4),
            })

    reduction_pct = (
        round((current_outliers - optimized_outliers) / current_outliers * 100, 1)
        if current_outliers > 0
        else 0.0
    )

    optimal_assignment = {
        "currentOutliers": current_outliers,
        "optimizedOutliers": optimized_outliers,
        "reductionPercent": reduction_pct,
        "changes": sorted(changes, key=lambda x: x["beforeProb"], reverse=True),
    }

    print(
        f"Optimal assignment: {current_outliers} -> {optimized_outliers} outliers "
        f"({reduction_pct}% reduction, {len(changes)} cases change physician)"
    )

    # ── 4. Physician workload ──
    current_workload = {}
    optimal_workload = {}
    for enc, name in PHYSICIAN_NAMES.items():
        current_workload[name] = sum(
            1 for c in case_reassignments if c["originalPhysicianEnc"] == enc
        )
        optimal_workload[name] = sum(
            1 for c in case_reassignments if c["optimalPhysician"] == enc
        )

    physician_workload = {
        "current": current_workload,
        "optimal": optimal_workload,
    }

    print(f"Workload current: {current_workload}")
    print(f"Workload optimal: {optimal_workload}")

    # ── 5. Export JSON ──
    output = {
        "caseReassignments": case_reassignments,
        "batchScenarios": batch_scenarios,
        "optimalAssignment": optimal_assignment,
        "physicianWorkload": physician_workload,
    }

    out_path = PROJECT_ROOT / "app" / "frontend" / "src" / "data" / "reassignment_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nReassignment data exported to {out_path}")
    print(f"  Cases: {len(case_reassignments)}")
    print(f"  Batch scenarios: {len(batch_scenarios)}")
    print(f"  Optimal changes: {len(changes)}")


if __name__ == "__main__":
    main()
