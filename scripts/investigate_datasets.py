"""
investigate_datasets.py
-----------------------
Investigates model performance across different dataset combinations
to understand why accuracy dropped with the combined dataset.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data_processing import DataProcessor
from src.model_training import ModelTrainer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def train_and_report(name: str, df: pd.DataFrame) -> None:
    """Train all three models on a dataset and print results."""
    print(f"\n{'='*55}")
    print(f" Dataset: {name}  ({len(df)} rows)")
    print(f"{'='*55}")

    # Drop metadata column if present
    if "chol_imputed" in df.columns:
        df = df.drop(columns=["chol_imputed"])

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    trainer = ModelTrainer(random_state=42)
    trainer.train_models(X_train, y_train)
    trainer.evaluate(X_test, y_test)

    comparison = trainer.compare()
    print(comparison.to_string(index=False))


def main() -> None:
    # 1. Original UCI dataset only
    original = pd.read_csv(DATA_DIR / "heart.csv")
    original.columns = [c.strip().lower() for c in original.columns]
    train_and_report("Original UCI (heart.csv)", original)

    # 2. Kaggle dataset only (after conversion via combined file)
    combined = pd.read_csv(DATA_DIR / "heart_combined.csv")
    kaggle_only = combined[combined.index >= len(original)].copy()
    train_and_report("Kaggle only", kaggle_only)

    # 3. Combined dataset
    train_and_report("Combined (heart_combined.csv)", combined)

    # 4. Combined WITHOUT imputed ca/thal rows
    kaggle_rows = combined.iloc[len(original):]
    non_imputed = pd.concat([
        original.assign(chol_imputed=0),
        kaggle_rows[kaggle_rows["chol_imputed"] == 0]
    ], ignore_index=True)
    train_and_report("Combined — excluding imputed chol rows", non_imputed)


if __name__ == "__main__":
    main()
