"""
data_preparation.py
-------------------
Prepares the heart disease dataset for model training.

Primary dataset:
    processed.cleveland.data — 303 rows, Cleveland Clinic Foundation
    Source: https://archive.ics.uci.edu/dataset/45/heart+disease

This script uses the Cleveland dataset only, which is the only UCI hospital
dataset with reliable measurements for all 13 features including ca and thal
— the two most important predictive features (~28% combined importance).

Investigation summary (see scripts/investigate_datasets.py):
    - Cleveland only (13 features, 302 rows):         ~88% RF accuracy
    - UCI 4 hospitals (11 features, 918 rows):        ~67% RF accuracy
    - Kaggle combined (11 features, 1220 rows):       ~67% RF accuracy

Adding more data from other hospitals requires dropping ca and thal due to
near-complete missingness (291/294 Hungarian, 118/123 Switzerland, 198/200 VA).
The accuracy loss from dropping these two features outweighs the gain from
more training data. Cleveland with all 13 features gives the best results.

Methodology:
1. Load processed.cleveland.data
2. Convert target to binary (0 = no disease, 1+ = disease present)
3. Impute the 6 missing values in ca and thal with column medians
4. Save to data/heart_combined.csv (used by DataProcessor by default)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# --- Path constants ---
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CLEVELAND_PATH = DATA_DIR / "processed.cleveland.data"
COMBINED_PATH = DATA_DIR / "heart_combined.csv"

# Column names for the processed UCI files
UCI_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]


def prepare_dataset() -> pd.DataFrame:
    """
    Load Cleveland dataset, convert target to binary,
    impute missing values, and save to heart_combined.csv.
    """
    if not CLEVELAND_PATH.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {CLEVELAND_PATH}\n"
            f"Download from https://archive.ics.uci.edu/dataset/45/heart+disease"
        )

    df = pd.read_csv(CLEVELAND_PATH, header=None, names=UCI_COLUMNS, na_values="?")

    print(f"Loaded Cleveland dataset: {len(df)} rows")
    print(f"Missing values before imputation: {df.isnull().sum().to_dict()}")

    # Convert target to binary
    df["target"] = (df["target"] > 0).astype(int)

    # Impute missing values (only 6 rows in ca and thal)
    missing_before = df.isnull().sum().sum()
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    print(f"Missing values imputed : {missing_before}")
    print(f"Final dataset          : {len(df)} rows, {len(df.columns)} columns")
    print(f"Target balance         : {df['target'].mean():.1%} positive")
    print(f"Saved to               : {COMBINED_PATH}")

    df.to_csv(COMBINED_PATH, index=False)
    return df


if __name__ == "__main__":
    prepare_dataset()
