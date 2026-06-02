"""Tests for data_processing module."""

from __future__ import annotations

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
        """Test loading data from CSV file."""
        processor = DataProcessor(sample_data_csv_path)
        df = processor.load_data()
        
        assert not df.empty
        assert len(df) == 100
        assert 'target' in df.columns

    def test_load_data_file_not_found(self) -> None:
        """Test loading data from non-existent file."""
        processor = DataProcessor("non_existent_file.csv")
        
        with pytest.raises(FileNotFoundError):
            processor.load_data()

    def test_clean_data(self, sample_data_csv_path: str) -> None:
        """Test data cleaning."""
        processor = DataProcessor(sample_data_csv_path)
        processor.load_data()
        cleaned_df = processor.clean_data()
        
        # Check column names are lowercase
        assert all(col.islower() for col in cleaned_df.columns)
        
        # Check no null values
        assert not cleaned_df.isnull().any().any()

    def test_clean_data_with_missing_values(self, sample_data_with_missing_values: pd.DataFrame) -> None:
        """Test cleaning data with missing values."""
        processor = DataProcessor("dummy_path")
        processor.df = sample_data_with_missing_values
        
        cleaned_df = processor.clean_data()
        
        # All missing values should be filled
        assert not cleaned_df.isnull().any().any()

    def test_get_features_and_target(self, sample_data_csv_path: str) -> None:
        """Test extracting features and target."""
        processor = DataProcessor(sample_data_csv_path)
        processor.load_data()
        processor.clean_data()
        
        X, y = processor.get_features_and_target()
        
        assert len(X) == 100
        assert len(y) == 100
        assert 'target' not in X.columns
        assert X.shape[1] == 13  # 13 features

    def test_split_data(self, sample_data_csv_path: str) -> None:
        """Test data splitting."""
        processor = DataProcessor(sample_data_csv_path)
        processor.load_data()
        processor.clean_data()
        
        X_train, X_test, y_train, y_test = processor.split_data(test_size=0.2, random_state=42)
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
        
        # Check stratification (roughly equal distribution in train and test)
        train_positive_ratio = y_train.sum() / len(y_train)
        test_positive_ratio = y_test.sum() / len(y_test)
        assert abs(train_positive_ratio - test_positive_ratio) < 0.2

    def test_split_data_custom_test_size(self, sample_data_csv_path: str) -> None:
        """Test splitting with custom test size."""
        processor = DataProcessor(sample_data_csv_path)
        processor.load_data()
        processor.clean_data()
        
        X_train, X_test, y_train, y_test = processor.split_data(test_size=0.3, random_state=42)
        
        assert len(X_train) == 70
        assert len(X_test) == 30

    def test_summary(self, sample_data_csv_path: str) -> None:
        """Test data summary generation."""
        processor = DataProcessor(sample_data_csv_path)
        processor.load_data()
        
        summary = processor.summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert 'dtype' in summary.index.names or len(summary.columns) > 0
        assert len(summary) == 14  # 14 columns in dataset

    def test_correlation_matrix(self, sample_data_csv_path: str) -> None:
        """Test correlation matrix computation."""
        processor = DataProcessor(sample_data_csv_path)
        processor.load_data()
        processor.clean_data()
        
        corr_matrix = processor.correlation_matrix()
        
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape[0] == corr_matrix.shape[1]  # Square matrix
        assert len(corr_matrix) == 14  # All columns

    def test_load_real_heart_dataset(self) -> None:
        """Test loading the actual heart dataset from data/heart.csv."""
        processor = DataProcessor('data/heart.csv')
        df = processor.load_data()

        assert not df.empty
        assert 'target' in df.columns
        assert df.shape[0] >= 1000
        assert df.shape[1] == 14

    def test_split_real_heart_dataset(self) -> None:
        """Test splitting the real heart dataset."""
        processor = DataProcessor('data/heart.csv')
        processor.load_data()
        processor.clean_data()
        X_train, X_test, y_train, y_test = processor.split_data(test_size=0.2, random_state=42)

        assert len(X_train) + len(X_test) == len(processor.df)
        assert 'target' not in X_train.columns
        assert len(y_train) > 0
        assert len(y_test) > 0
