"""
Phase 1: Data loading and cleaning.
"""

import pandas as pd
from src.config import DATA_PATH, NUM_COLS, DROP_COLS, TARGET_COL


def load_and_clean() -> pd.DataFrame:
    """Load Excel data, parse columns, convert types, and drop incomplete rows."""
    print("=" * 60)
    print("1. LOADING & CLEANING DATA")
    print("=" * 60)

    df_raw = pd.read_excel(DATA_PATH, header=None)
    cols = df_raw.iloc[2].tolist()
    cols = [str(c).strip() if pd.notna(c) else f"col_{i}" for i, c in enumerate(cols)]
    df = df_raw.iloc[4:].copy()
    df.columns = cols
    df = df.drop(columns=["col_0"])
    df = df.reset_index(drop=True)

    # Convert numeric columns
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Parse date
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    # Drop AVG columns and PT OUT TIME
    df = df.drop(columns=DROP_COLS)

    # Drop rows missing the target
    df = df.dropna(subset=[TARGET_COL])
    print(f"Dataset shape after cleaning: {df.shape}")
    print(f"Target ({TARGET_COL}) stats:\n{df[TARGET_COL].describe()}\n")

    return df
