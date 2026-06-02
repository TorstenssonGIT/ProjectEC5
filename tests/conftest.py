"""Pytest configuration and fixtures for heart disease prediction tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from joblib import dump


@pytest.fixture
def sample_heart_data() -> pd.DataFrame:
    """Create sample heart disease dataset for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'age': np.random.randint(29, 78, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),
        'trestbps': np.random.randint(90, 200, n_samples),
        'chol': np.random.randint(100, 400, n_samples),
        'fbs': np.random.randint(0, 2, n_samples),
        'restecg': np.random.randint(0, 3, n_samples),
        'thalach': np.random.randint(60, 200, n_samples),
        'exang': np.random.randint(0, 2, n_samples),
        'oldpeak': np.random.uniform(0, 6, n_samples),
        'slope': np.random.randint(0, 3, n_samples),
        'ca': np.random.randint(0, 5, n_samples),
        'thal': np.random.choice([3, 6, 7], n_samples),
        'target': np.random.randint(0, 2, n_samples),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_data_csv_path(sample_heart_data: pd.DataFrame) -> str:
    """Create temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_heart_data.to_csv(f.name, index=False)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def sample_data_with_missing_values() -> pd.DataFrame:
    """Create sample dataset with missing values."""
    np.random.seed(42)
    n_samples = 50
    
    data = {
        'age': np.random.randint(29, 78, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),
        'trestbps': np.random.randint(90, 200, n_samples),
        'chol': np.random.randint(100, 400, n_samples),
        'fbs': np.random.randint(0, 2, n_samples),
        'restecg': np.random.randint(0, 3, n_samples),
        'thalach': np.random.randint(60, 200, n_samples),
        'exang': np.random.randint(0, 2, n_samples),
        'oldpeak': np.random.uniform(0, 6, n_samples),
        'slope': np.random.randint(0, 3, n_samples),
        'ca': np.random.randint(0, 5, n_samples),
        'thal': np.random.choice([3, 6, 7], n_samples),
        'target': np.random.randint(0, 2, n_samples),
    }
    
    df = pd.DataFrame(data)
    # Introduce some missing values
    df.loc[5:10, 'chol'] = np.nan
    df.loc[15:20, 'thalach'] = np.nan
    
    return df


@pytest.fixture
def trained_mock_model():
    """Create and return a simple trained mock model."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    
    # Create a simple pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=100, random_state=42))
    ])
    
    # Train with sample data
    np.random.seed(42)
    X = np.random.randn(100, 13)
    y = np.random.randint(0, 2, 100)
    pipeline.fit(X, y)
    
    return pipeline


@pytest.fixture
def model_file_path(trained_mock_model) -> str:
    """Create temporary model file."""
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
        dump(trained_mock_model, f.name)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_model_output() -> str:
    """Create path for temporary model output."""
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
        temp_path = f.name
    
    # Remove the file so it can be created by tests
    Path(temp_path).unlink()
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)
