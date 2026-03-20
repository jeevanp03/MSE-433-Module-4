"""
MSE 433 Module 4 - Medical Procedure Outlier Analysis
=====================================================
Entry point that runs the full analysis pipeline.
Equivalent to running outlier_analysis.py.
"""

import joblib

from src.config import init_output_dirs, MODEL_DIR
from src.data_loader import load_and_clean
from src.eda import run_eda
from src.feature_eng import engineer_features
from src.model import train_global_model
from src.per_physician import run_per_physician_analysis
from src.additional import run_additional_analyses
from src.export import save_results


def main() -> None:
    # Phase 0: Initialize output directories
    init_output_dirs()

    # Phase 1: Data loading & cleaning
    df = load_and_clean()

    # Phase 2: EDA & outlier identification
    outlier_threshold = run_eda(df)

    # Phase 3: Feature engineering
    X, y, feature_cols, le_phys = engineer_features(df)

    # Phase 4-5: Global LightGBM model + SHAP analysis
    model, sv, top_features = train_global_model(X, y, feature_cols)

    # Save trained model + artifacts for downstream scripts
    # Include case_nums aligned with X rows for safe joining in backend scripts
    case_nums = df.loc[X.index, 'CASE #'].values
    joblib.dump({
        'model': model,
        'shap_values': sv,
        'X': X,
        'y': y,
        'feature_cols': feature_cols,
        'le_phys': le_phys,
        'threshold': outlier_threshold,
        'case_nums': case_nums,
    }, MODEL_DIR / 'global_model.pkl')
    print(f"Saved: {MODEL_DIR / 'global_model.pkl'}")

    # Phase 6: Per-physician outlier detection & models
    per_physician_results = {}
    per_physician_results = run_per_physician_analysis(
        df, feature_cols, outlier_threshold, per_physician_results,
    )

    # Phase 7: Additional analyses
    additional_results = run_additional_analyses(df, per_physician_results)

    # Phase 8: Save results
    save_results(
        df, feature_cols, outlier_threshold, top_features,
        per_physician_results, additional_results, X, y,
    )

    # Final summary
    physicians = sorted(df["PHYSICIAN"].unique())
    n_outliers = int(df["outlier_class"].sum())

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nGlobal findings:")
    print(f"  - Total cases: {len(df)}")
    print(f"  - Global outlier threshold: >={outlier_threshold:.1f} min (90th pctl)")
    print(f"  - Global outliers: {n_outliers} / {len(df)}")
    print(f"  - Global model: fitted on {len(X)} of {len(df)} cases ({len(df) - len(X)} excluded for missing features)")
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
    print(f"\nAll outputs saved to: output/")


if __name__ == "__main__":
    main()
