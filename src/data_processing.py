from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# Default to the combined dataset
_DEFAULT_DATA_PATH = str(
    Path(__file__).resolve().parent.parent / "data" / "heart_combined.csv"
)

# Columns that are metadata/flags and should never be used as model features
_COLUMNS_TO_DROP = ["chol_imputed", "ca", "thal"]


class DataProcessor:
    """Load, clean and split the heart disease dataset."""

    def __init__(self, data_path: str = _DEFAULT_DATA_PATH) -> None:
        self.data_path = data_path
        self.df: pd.DataFrame = pd.DataFrame()

    def load_data(self) -> pd.DataFrame:
        """Load CSV data from the configured path."""
        self.df = pd.read_csv(self.data_path)
        return self.df

    def clean_data(self) -> pd.DataFrame:
        """Normalize column names, drop metadata columns,
        fill missing values and return cleaned data."""
        if self.df.empty:
            self.load_data()

        self.df = self.df.copy()
        self.df.columns = [col.strip().lower() for col in self.df.columns]

        # Drop metadata/flag columns and features not present in all datasets
        cols_to_drop = [c for c in _COLUMNS_TO_DROP if c in self.df.columns]
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)

        if self.df.isnull().any().any():
            numeric_cols = self.df.select_dtypes(include="number").columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(
                self.df[numeric_cols].median()
            )

        return self.df

    def get_features_and_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Return feature matrix X and target vector y."""
        if self.df.empty:
            self.clean_data()

        X = self.df.drop("target", axis=1)
        y = self.df["target"]
        return X, y

    def split_data(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split the data into training and test sets."""
        X, y = self.get_features_and_target()
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

    def summary(self) -> pd.DataFrame:
        """Return a summary of dataset statistics and missing values."""
        if self.df.empty:
            self.clean_data()

        summary = pd.DataFrame(
            {
                "dtype": self.df.dtypes,
                "missing": self.df.isna().sum(),
                "unique": self.df.nunique(),
            }
        )
        return summary

    def correlation_matrix(self) -> pd.DataFrame:
        """Compute and return the correlation matrix."""
        if self.df.empty:
            self.clean_data()
        return self.df.corr()
