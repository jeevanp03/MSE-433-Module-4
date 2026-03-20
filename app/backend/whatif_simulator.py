"""
What-If Simulator backend.
Loads the saved global LightGBM model, then pre-computes a response surface
by varying each top feature across 20 points while holding others at median.
Exports whatif_data.json for the React dashboard.
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

# Add project root to path so we can import src.config
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUT_DIR
import joblib


def main() -> None:
    # ── Load saved model artifacts ──
    model_path = OUTPUT_DIR / "model" / "global_model.pkl"
    artifacts = joblib.load(model_path)
    model = artifacts['model']
    X = artifacts['X']
    feature_cols = artifacts['feature_cols']

    OUT_PATH = PROJECT_ROOT / "app" / "frontend" / "src" / "data" / "whatif_data.json"

    # ── SHAP explainer ──
    explainer = shap.TreeExplainer(model)

    # ── Identify top 5 features by mean |SHAP| ──
    sv = artifacts['shap_values']
    mean_abs_shap = np.abs(sv).mean(axis=0)
    shap_series = pd.Series(mean_abs_shap, index=feature_cols)
    top5 = shap_series.nlargest(5)
    top5_names = list(top5.index)

    print(f"Top 5 features for what-if: {top5_names}")

    # ── Compute medians for baseline ──
    medians = X.median().to_dict()

    # ── Feature metadata ──
    features_meta = []
    for feat in top5_names:
        col = X[feat]
        feat_min = float(col.min())
        feat_max = float(col.max())
        feat_median = float(col.median())
        step = round((feat_max - feat_min) / 20, 4)
        features_meta.append({
            "name": feat,
            "min": round(feat_min, 2),
            "max": round(feat_max, 2),
            "median": round(feat_median, 2),
            "step": step if step > 0 else 0.1,
        })

    # ── Response surface: for each top feature, vary across 20 points ──
    response_surface = []
    for feat in top5_names:
        col = X[feat]
        feat_min = float(col.min())
        feat_max = float(col.max())
        points = np.linspace(feat_min, feat_max, 20)

        values = []
        for pt in points:
            row = pd.DataFrame([medians], columns=feature_cols)
            row[feat] = pt

            prob = float(model.predict_proba(row)[0, 1])

            sv_row = explainer.shap_values(row)
            if isinstance(sv_row, list):
                sv_single = sv_row[1][0]
            else:
                sv_single = sv_row[0]

            shap_contribs = {}
            for i, fc in enumerate(feature_cols):
                shap_contribs[fc] = round(float(sv_single[i]), 4)

            values.append({
                "featureValue": round(float(pt), 2),
                "probability": round(prob, 4),
                "shapContributions": shap_contribs,
            })

        response_surface.append({
            "featureName": feat,
            "values": values,
        })

    print("Response surface computed for all 5 features x 20 points")

    # ── Presets ──
    typical = {feat: round(float(medians[feat]), 2) for feat in feature_cols}

    complex_preset = dict(typical)
    for feat in ["ABL DURATION", "#ABL", "#APPLICATIONS", "TSP", "PRE-MAP"]:
        if feat in feature_cols:
            complex_preset[feat] = round(float(X[feat].quantile(0.90)), 2)
    for feat in ["NOTE_CTI", "NOTE_BOX"]:
        if feat in feature_cols:
            complex_preset[feat] = 1

    worst_case = {}
    for feat in feature_cols:
        col = X[feat]
        if feat == "PHYSICIAN_ENC":
            worst_case[feat] = 1.0  # Dr. B (the physician with most outliers)
        elif col.nunique() <= 2:
            worst_case[feat] = round(float(col.max()), 2)
        else:
            worst_case[feat] = round(float(col.quantile(0.95)), 2)

    optimized = {}
    for feat in feature_cols:
        col = X[feat]
        if col.nunique() <= 2:
            optimized[feat] = 0
        else:
            optimized[feat] = round(float(col.quantile(0.10)), 2)

    presets = {
        "typical": typical,
        "complex": complex_preset,
        "worstCase": worst_case,
        "optimized": optimized,
    }

    for name, preset_vals in presets.items():
        row = pd.DataFrame([preset_vals], columns=feature_cols)
        prob = float(model.predict_proba(row)[0, 1])
        presets[name] = {
            "featureValues": preset_vals,
            "outlierProbability": round(prob, 4),
        }

    # ── Assemble output ──
    whatif_data = {
        "features": features_meta,
        "responseSurface": response_surface,
        "presets": presets,
        "allFeatures": feature_cols,
        "medians": {k: round(float(v), 2) for k, v in medians.items()},
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(whatif_data, f, indent=2)

    print(f"What-if data exported to {OUT_PATH}")
    print(f"  Top features: {[f['name'] for f in features_meta]}")
    print(f"  Response surface: {len(response_surface)} features x 20 points each")
    print(f"  Presets: {list(presets.keys())}")


if __name__ == "__main__":
    main()
