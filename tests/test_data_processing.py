"""Tests for data_processing module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_processing import DataProcessor


class TestDataProcessor:
    """Test cases for DataProcessor class."""

    def test_initialization(self, sample_data_csv_path: str) -> None:
        """Test DataProcessor initialization."""
        processor = DataProcessor(sample_data_csv_path)
        assert processor.data_path == sample_data_csv_path
        assert processor.df.empty

    def test_load_data(self, sample_data_csv_path: str) -> None:
        """Test data loading."""
        processor = DataProcessor(sample_data_csv_path)
        df = processor.load_data()
        assert not df.empty
        assert len(df) == 100

    def test_clean_data(self, sample_data_csv_path: str) -> None:
        """Test data cleaning."""
        processor = DataProcessor(sample_data_csv_path)
        processor.load_data()
        df = processor.clean_data()
        assert not df.empty
        assert df.isnull().sum().sum() == 0

    def test_clean_data_drops_chol_imputed(self, sample_data_csv_path: str) -> None:
        """Test that chol_imputed metadata column is dropped during cleaning."""
        processor = DataProcessor(sample_data_csv_path)
        processor.load_data()
        processor.clean_data()
        assert "chol_imputed" not in processor.df.columns

    def test_clean_data_keeps_ca_and_thal(self, sample_data_csv_path: str) -> None:
        """ca and thal are real clinical features and should be kept."""
        processor = DataProcessor(sample_data_csv_path)
        processor.load_data()
        processor.clean_data()
        assert "ca" in processor.df.columns
        assert "thal" in processor.df.columns

    def test_clean_data_normalizes_columns(self, sample_data_csv_path: str) -> None:
        """Test that column names are normalized to lowercase."""
        processor = DataProcessor(sample_data_csv_path)
        processor.load_data()
        processor.clean_data()
        assert all(col == col.lower() for col in processor.df.columns)

    def test_clean_data_handles_missing_values(
        self, sample_data_with_missing_values: pd.DataFrame, tmp_path
    ) -> None:
        """Test handling of missing values."""
        csv_path = str(tmp_path / "missing.csv")
        sample_data_with_missing_values.to_csv(csv_path, index=False)
        processor = DataProcessor(csv_path)
        processor.load_data()
        processor.clean_data()
        assert processor.df.isnull().sum().sum() == 0

    def test_get_features_and_target(self, sample_data_csv_path: str) -> None:
        """Test extracting features and target."""
        processor = DataProcessor(sample_data_csv_path)
        processor.load_data()
        processor.clean_data()

        X, y = processor.get_features_and_target()

        assert len(X) == 100
        assert len(y) == 100
        assert "target" not in X.columns
        assert X.shape[1] == 13  # all 13 features including ca and thal

    def test_split_data(self, sample_data_csv_path: str) -> None:
        """Test data splitting."""
        processor = DataProcessor(sample_data_csv_path)
        processor.load_data()
        processor.clean_data()

        X_train, X_test, y_train, y_test = processor.split_data(
            test_size=0.2, random_state=42
        )

        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_split_data_stratified(self, sample_data_csv_path: str) -> None:
        """Test that split maintains class balance."""
        processor = DataProcessor(sample_data_csv_path)
        processor.load_data()
        processor.clean_data()

        X_train, X_test, y_train, y_test = processor.split_data(
            test_size=0.2, random_state=42
        )

        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.1

    def test_summary(self, sample_data_csv_path: str) -> None:
        """Test dataset summary generation."""
        processor = DataProcessor(sample_data_csv_path)
        processor.load_data()
        processor.clean_data()

        summary = processor.summary()

        assert isinstance(summary, pd.DataFrame)
        assert "dtype" in summary.columns
        assert "missing" in summary.columns
        assert "unique" in summary.columns

    def test_correlation_matrix(self, sample_data_csv_path: str) -> None:
        """Test correlation matrix computation."""
        processor = DataProcessor(sample_data_csv_path)
        processor.load_data()
        processor.clean_data()

        corr_matrix = processor.correlation_matrix()

        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape[0] == corr_matrix.shape[1]
        # 13 features + target = 14 columns
        assert len(corr_matrix) == 14

    def test_load_real_heart_dataset(self) -> None:
        """Test loading the prepared heart dataset."""
        from pathlib import Path
        data_path = str(Path(__file__).resolve().parent.parent / "data" / "heart_combined.csv")
        processor = DataProcessor(data_path)
        df = processor.load_data()

        assert not df.empty
        assert "target" in df.columns
        assert df.shape[0] >= 300

    def test_default_path_uses_combined(self) -> None:
        """Test that default path points to heart_combined.csv."""
        from src.data_processing import _DEFAULT_DATA_PATH
        assert "heart_combined.csv" in _DEFAULT_DATA_PATH
