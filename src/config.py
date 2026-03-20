"""
Configuration constants, paths, and hyperparameters.
"""

import shutil
from pathlib import Path

RANDOM_STATE = 42

# Absolute paths derived from project root
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
EDA_DIR = OUTPUT_DIR / "eda"
GLOBAL_DIR = OUTPUT_DIR / "global_model"
PHYS_DIR = OUTPUT_DIR / "per_physician"
EXTRA_DIR = OUTPUT_DIR / "additional"
MODEL_DIR = OUTPUT_DIR / "model"

# Data source
DATA_PATH = BASE_DIR / "Data" / "MSE433_M4_Data.xlsx"

# Numeric columns in the raw data
NUM_COLS = [
    "PT PREP/INTUBATION", "ACCESSS", "TSP", "PRE-MAP",
    "ABL DURATION", "ABL TIME", "#ABL", "#APPLICATIONS",
    "LA DWELL TIME", "CASE TIME", "SKIN-SKIN",
    "POST CARE/EXTUBATION", "PT IN-OUT",
]

# Columns to drop from the raw data
DROP_COLS = ["AVG CASE TIME", "AVG SKIN-SKIN", "AVG TURNOVER TIME", "PT OUT TIME"]

# Target variable
TARGET_COL = "PT IN-OUT"

# Granular feature columns (excludes aggregate sub-totals to avoid leakage)
# '#APPLICATIONS' excluded — deterministic function of #ABL (always 3×), r=1.0 correlation
BASE_FEATURE_COLS = [
    "PT PREP/INTUBATION", "ACCESSS", "TSP", "PRE-MAP",
    "ABL DURATION", "ABL TIME", "#ABL",
    "POST CARE/EXTUBATION",
]

# Comparison columns for outlier vs normal analysis
COMPARE_COLS = [
    "PT PREP/INTUBATION", "ACCESSS", "TSP", "PRE-MAP",
    "ABL DURATION", "ABL TIME", "#ABL", "#APPLICATIONS",
    "POST CARE/EXTUBATION",
]

# Global LightGBM parameters
GLOBAL_LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "num_leaves": 15,
    "learning_rate": 0.05,
    # n_estimators=100 (reduced from 200 to limit memorization; model achieves ~100% training
    # accuracy regardless, but lower capacity makes SHAP values more robust)
    "n_estimators": 100,
    "max_depth": 4,
    "min_child_samples": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

# Per-physician LightGBM parameters
PHYS_LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "num_leaves": 10,
    "learning_rate": 0.05,
    # n_estimators=75 (reduced from 150; same rationale as global model)
    "n_estimators": 75,
    "max_depth": 3,
    "min_child_samples": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

# Outlier class labels
OUTLIER_LABEL_MAP = {0: "Normal", 1: "Outlier (Top 10%)"}
CLASS_LABELS = {0: "Normal Duration", 1: "Long Duration (Top 10%)"}


def init_output_dirs() -> None:
    """Clean and recreate the output directory structure."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for d in [OUTPUT_DIR, EDA_DIR, GLOBAL_DIR, PHYS_DIR, EXTRA_DIR, MODEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)
