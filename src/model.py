"""
Phases 4-5: LightGBM training and SHAP explainability.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap

from src.config import GLOBAL_DIR, GLOBAL_LGB_PARAMS
from src.viz import SAVE_DPI


def train_global_model(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: List[str],
) -> Tuple[lgb.LGBMClassifier, np.ndarray, pd.Series]:
    """Train global LightGBM model and run SHAP analysis.

    Returns (model, shap_values_positive_class, top_features_series).
    """
    # --- Phase 4: Train ---
    print("\n" + "=" * 60)
    print("4. LIGHTGBM MODEL TRAINING")
    print("=" * 60)

    params = dict(GLOBAL_LGB_PARAMS)
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)
    print(f"\nModel fitted on {len(X)} cases ({int(y.sum())} outliers, {int(len(y) - y.sum())} normal)")

    # Feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    lgb.plot_importance(model, ax=ax, importance_type="gain", max_num_features=15)
    ax.set_title("Global Model: Feature Importance by Information Gain")
    plt.tight_layout()
    plt.savefig(f"{GLOBAL_DIR}/lgbm_feature_importance.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print("Saved: lgbm_feature_importance.png")

    # --- Phase 5: SHAP ---
    print("\n" + "=" * 60)
    print("5. SHAP ANALYSIS")
    print("=" * 60)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # shap_values for the positive class
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # Summary plot (bee swarm)
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(sv, X, show=False, max_display=15)
    plt.title("Global Model: Feature Impact on Outlier Prediction (SHAP)")
    plt.tight_layout()
    plt.savefig(f"{GLOBAL_DIR}/shap_summary.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print("Saved: shap_summary.png")

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(sv, X, plot_type="bar", show=False, max_display=15)
    plt.title("Global Model: Average Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(f"{GLOBAL_DIR}/shap_bar.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print("Saved: shap_bar.png")

    # Dependence plots for top features
    mean_abs_shap = np.abs(sv).mean(axis=0)
    top_features = pd.Series(mean_abs_shap, index=feature_cols).nlargest(4)
    print(f"\nTop 4 features by mean |SHAP|:")
    for feat, val in top_features.items():
        print(f"  {feat}: {val:.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, (feat, _) in enumerate(top_features.items()):
        ax = axes[idx // 2, idx % 2]
        shap.dependence_plot(feat, sv, X, ax=ax, show=False)
        ax.set_title(f"Effect of {feat} on Outlier Prediction")
    plt.suptitle("How Top Features Influence Outlier Classification", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{GLOBAL_DIR}/shap_dependence.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print("Saved: shap_dependence.png")

    return model, sv, top_features
