"""
data_preparation.py
-------------------
Combines the original UCI Cleveland Heart Disease dataset (heart.csv)
with the Kaggle Heart Failure Prediction dataset (heart_kaggle.csv)
into a single cleaned dataset (heart_combined.csv).

Methodology:
1. Load both source files — originals are never modified
2. Convert Kaggle text labels to numeric values matching the UCI encoding
3. Replace physiologically impossible zero values (Cholesterol, RestingBP)
   with age-appropriate averages and flag imputed rows in 'chol_imputed'
4. Impute missing features (ca, thal) using median values from the original dataset
5. Rename Kaggle columns to match UCI column names
6. Concatenate both datasets and remove duplicates
7. Save to data/heart_combined.csv

Note on chol_imputed flag:
    The column 'chol_imputed' (1=imputed, 0=original) is included in the CSV
    for transparency and auditing only. It is NOT a clinical feature and must
    be dropped before model training. DataProcessor handles this automatically
    by selecting only the 13 known feature columns.

Column mapping (Kaggle -> UCI):
    Age           -> age
    Sex           -> sex           (M=1, F=0)
    ChestPainType -> cp            (ATA=1, NAP=2, ASY=0, TA=3)
    RestingBP     -> trestbps
    Cholesterol   -> chol
    FastingBS     -> fbs
    RestingECG    -> restecg       (Normal=1, ST=2, LVH=0)
    MaxHR         -> thalach
    ExerciseAngina-> exang         (Y=1, N=0)
    Oldpeak       -> oldpeak
    ST_Slope      -> slope         (Up=2, Flat=1, Down=0)
    HeartDisease  -> target

Missing features imputed from original dataset medians:
    ca   (number of major vessels, 0-4)
    thal (thalassemia type)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# --- Path constants ---
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ORIGINAL_PATH = DATA_DIR / "heart.csv"
KAGGLE_PATH = DATA_DIR / "heart_kaggle.csv"
COMBINED_PATH = DATA_DIR / "heart_combined.csv"

# --- Encoding maps ---
SEX_MAP = {"M": 1, "F": 0}

CHEST_PAIN_MAP = {
    "ASY": 0,   # Asymptomatic
    "ATA": 1,   # Atypical Angina
    "NAP": 2,   # Non-Anginal Pain
    "TA":  3,   # Typical Angina
}

RESTING_ECG_MAP = {
    "LVH":    0,   # Left ventricular hypertrophy
    "Normal": 1,
    "ST":     2,   # ST-T wave abnormality
}

EXERCISE_ANGINA_MAP = {"N": 0, "Y": 1}

ST_SLOPE_MAP = {
    "Down": 0,
    "Flat": 1,
    "Up":   2,
}

COLUMN_RENAME_MAP = {
    "Age":            "age",
    "Sex":            "sex",
    "ChestPainType":  "cp",
    "RestingBP":      "trestbps",
    "Cholesterol":    "chol",
    "FastingBS":      "fbs",
    "RestingECG":     "restecg",
    "MaxHR":          "thalach",
    "ExerciseAngina": "exang",
    "Oldpeak":        "oldpeak",
    "ST_Slope":       "slope",
    "HeartDisease":   "target",
}

# Age-based average cholesterol values (mg/dL)
# Source: general clinical reference ranges
AGE_CHOLESTEROL_MAP = [
    (20, 29, 170),
    (30, 39, 185),
    (40, 49, 200),
    (50, 59, 215),
    (60, 69, 210),
    (70, 79, 205),
    (80, 90, 195),
]


def get_age_based_cholesterol(age: int) -> float:
    """Return the age-appropriate average cholesterol value."""
    for low, high, avg in AGE_CHOLESTEROL_MAP:
        if low <= age <= high:
            return float(avg)
    # Fallback for ages outside defined ranges
    return 200.0


def load_original() -> pd.DataFrame:
    """Load and normalise the original UCI Cleveland dataset."""
    df = pd.read_csv(ORIGINAL_PATH)
    df.columns = [col.strip().lower() for col in df.columns]
    # Flag all original rows as not imputed
    df["chol_imputed"] = 0
    return df


def load_kaggle(ca_median: float, thal_median: float) -> pd.DataFrame:
    """
    Load the Kaggle dataset, convert text labels to numeric values,
    impute missing features, and rename columns to match UCI.

    Args:
        ca_median:   Median value of 'ca' from the original dataset.
        thal_median: Median value of 'thal' from the original dataset.
    """
    df = pd.read_csv(KAGGLE_PATH)

    # Convert text labels to numeric
    df["Sex"] = df["Sex"].map(SEX_MAP)
    df["ChestPainType"] = df["ChestPainType"].map(CHEST_PAIN_MAP)
    df["RestingECG"] = df["RestingECG"].map(RESTING_ECG_MAP)
    df["ExerciseAngina"] = df["ExerciseAngina"].map(EXERCISE_ANGINA_MAP)
    df["ST_Slope"] = df["ST_Slope"].map(ST_SLOPE_MAP)

    # Flag rows where cholesterol will be imputed
    df["chol_imputed"] = (df["Cholesterol"] == 0).astype(int)
    imputed_count = df["chol_imputed"].sum()

    # Replace zero cholesterol with age-appropriate averages
    zero_mask = df["Cholesterol"] == 0
    df.loc[zero_mask, "Cholesterol"] = df.loc[zero_mask, "Age"].apply(
        get_age_based_cholesterol
    )

    # Replace zero RestingBP with median of valid values
    bp_median = df.loc[df["RestingBP"] > 0, "RestingBP"].median()
    df.loc[df["RestingBP"] == 0, "RestingBP"] = bp_median

    # Rename columns to match UCI
    df = df.rename(columns=COLUMN_RENAME_MAP)

    # Impute missing features using original dataset medians
    df["ca"] = ca_median
    df["thal"] = thal_median

    print(f"Cholesterol imputed (age-based): {imputed_count} rows")
    print(f"RestingBP imputed (median)      : {(df['trestbps'] == bp_median).sum()} rows")

    return df


def combine_datasets() -> pd.DataFrame:
    """
    Combine the original and Kaggle datasets into a single cleaned dataframe.
    Saves to heart_combined.csv and returns the combined dataframe.
    """
    original = load_original()

    # Compute medians from original for imputation
    ca_median = original["ca"].median()
    thal_median = original["thal"].median()

    print(f"ca median (from original)  : {ca_median}")
    print(f"thal median (from original): {thal_median}")
    print()

    kaggle = load_kaggle(ca_median, thal_median)

    # Ensure column order matches original
    kaggle = kaggle[original.columns]

    # Concatenate and remove duplicates
    combined = pd.concat([original, kaggle], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(
        subset=[c for c in combined.columns if c != "chol_imputed"]
    )
    after = len(combined)

    print()
    print(f"Original dataset  : {len(original)} rows")
    print(f"Kaggle dataset    : {len(kaggle)} rows")
    print(f"Combined (raw)    : {before} rows")
    print(f"Duplicates removed: {before - after}")
    print(f"Final dataset     : {after} rows")
    print(f"Imputed chol rows : {combined['chol_imputed'].sum()}")
    print(f"Saved to          : {COMBINED_PATH}")

    combined.to_csv(COMBINED_PATH, index=False)
    return combined


if __name__ == "__main__":
    combine_datasets()
