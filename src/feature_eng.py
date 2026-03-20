"""
Phase 3: Feature engineering, label encoding, and exclusions.
"""

from typing import List, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import BASE_FEATURE_COLS


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], LabelEncoder]:
    """Add engineered features and return (X, feature_cols, label_encoder)."""
    print("\n" + "=" * 60)
    print("3. FEATURE ENGINEERING & MODEL PREPARATION")
    print("=" * 60)

    feature_cols = list(BASE_FEATURE_COLS)

    # Note: Label encoding imposes ordinal relationship (Dr.A=0 < Dr.B=1 < Dr.C=2) which
    # doesn't exist clinically. LightGBM handles this via threshold splits, so the practical
    # impact is negligible. Native categorical support would be more principled but equivalent.
    le_phys = LabelEncoder()
    df["PHYSICIAN_ENC"] = le_phys.fit_transform(df["PHYSICIAN"].astype(str))
    feature_cols.append("PHYSICIAN_ENC")

    # Encode Note as binary features for procedure type markers
    df["HAS_NOTE"] = df["Note"].notna() & (df["Note"] != "")
    df["NOTE_CTI"] = df["Note"].astype(str).str.contains("CTI", na=False).astype(int)
    # NOTE_BOX matches any Note containing "BOX" (including "PST BOX"). This overlap with
    # NOTE_PST is intentional — it flags "any BOX procedure" as a model feature.
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

    return X, y, feature_cols, le_phys
