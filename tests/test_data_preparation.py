"""Tests for data_preparation module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_preparation import (
    AGE_CHOLESTEROL_MAP,
    combine_datasets,
    get_age_based_cholesterol,
    load_kaggle,
    load_original,
)


# --- Fixtures ---

@pytest.fixture
def sample_original_csv(tmp_path: Path) -> Path:
    """Create a temporary original-style CSV (UCI format, with ca and thal)."""
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "age":      np.random.randint(29, 78, n),
        "sex":      np.random.randint(0, 2, n),
        "cp":       np.random.randint(0, 4, n),
        "trestbps": np.random.randint(90, 200, n),
        "chol":     np.random.randint(150, 400, n),
        "fbs":      np.random.randint(0, 2, n),
        "restecg":  np.random.randint(0, 3, n),
        "thalach":  np.random.randint(60, 200, n),
        "exang":    np.random.randint(0, 2, n),
        "oldpeak":  np.random.uniform(0, 6, n),
        "slope":    np.random.randint(0, 3, n),
        "ca":       np.random.randint(0, 5, n),
        "thal":     np.random.choice([3, 6, 7], n),
        "target":   np.random.randint(0, 2, n),
    })
    path = tmp_path / "heart.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_kaggle_csv(tmp_path: Path) -> Path:
    """Create a temporary Kaggle-style CSV with text labels and some zero values."""
    np.random.seed(42)
    n = 40
    df = pd.DataFrame({
        "Age":            np.random.randint(29, 78, n),
        "Sex":            np.random.choice(["M", "F"], n),
        "ChestPainType":  np.random.choice(["ATA", "NAP", "ASY", "TA"], n),
        "RestingBP":      np.random.randint(90, 200, n),
        "Cholesterol":    np.random.randint(150, 400, n),
        "FastingBS":      np.random.randint(0, 2, n),
        "RestingECG":     np.random.choice(["Normal", "ST", "LVH"], n),
        "MaxHR":          np.random.randint(60, 200, n),
        "ExerciseAngina": np.random.choice(["Y", "N"], n),
        "Oldpeak":        np.random.uniform(0, 6, n),
        "ST_Slope":       np.random.choice(["Up", "Flat", "Down"], n),
        "HeartDisease":   np.random.randint(0, 2, n),
    })
    # Introduce zero cholesterol and RestingBP values
    df.loc[0, "Cholesterol"] = 0
    df.loc[1, "Cholesterol"] = 0
    df.loc[2, "RestingBP"] = 0
    path = tmp_path / "heart_kaggle.csv"
    df.to_csv(path, index=False)
    return path


# --- Tests for get_age_based_cholesterol ---

class TestGetAgeBasedCholesterol:
    """Tests for the age-based cholesterol lookup function."""

    def test_returns_correct_value_for_each_age_band(self) -> None:
        for low, high, expected in AGE_CHOLESTEROL_MAP:
            mid = (low + high) // 2
            assert get_age_based_cholesterol(mid) == float(expected)

    def test_boundary_ages(self) -> None:
        for low, high, expected in AGE_CHOLESTEROL_MAP:
            assert get_age_based_cholesterol(low) == float(expected)
            assert get_age_based_cholesterol(high) == float(expected)

    def test_fallback_for_out_of_range_age(self) -> None:
        assert get_age_based_cholesterol(15) == 200.0
        assert get_age_based_cholesterol(95) == 200.0

    def test_returns_float(self) -> None:
        assert isinstance(get_age_based_cholesterol(45), float)


# --- Tests for load_original ---

class TestLoadOriginal:
    """Tests for loading the original UCI dataset."""

    def test_loads_correctly(self, sample_original_csv: Path, monkeypatch) -> None:
        import src.data_preparation as dp
        monkeypatch.setattr(dp, "ORIGINAL_PATH", sample_original_csv)
        df = load_original()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50
        assert "target" in df.columns

    def test_adds_chol_imputed_flag(self, sample_original_csv: Path, monkeypatch) -> None:
        import src.data_preparation as dp
        monkeypatch.setattr(dp, "ORIGINAL_PATH", sample_original_csv)
        df = load_original()
        assert "chol_imputed" in df.columns
        assert (df["chol_imputed"] == 0).all()

    def test_column_names_are_lowercase(self, sample_original_csv: Path, monkeypatch) -> None:
        import src.data_preparation as dp
        monkeypatch.setattr(dp, "ORIGINAL_PATH", sample_original_csv)
        df = load_original()
        assert all(col == col.lower() for col in df.columns)

    def test_ca_and_thal_dropped(self, sample_original_csv: Path, monkeypatch) -> None:
        """ca and thal should be dropped from the original dataset."""
        import src.data_preparation as dp
        monkeypatch.setattr(dp, "ORIGINAL_PATH", sample_original_csv)
        df = load_original()
        assert "ca" not in df.columns
        assert "thal" not in df.columns


# --- Tests for load_kaggle ---

class TestLoadKaggle:
    """Tests for loading and converting the Kaggle dataset."""

    def test_loads_and_converts(self, sample_kaggle_csv: Path, monkeypatch) -> None:
        import src.data_preparation as dp
        monkeypatch.setattr(dp, "KAGGLE_PATH", sample_kaggle_csv)
        df = load_kaggle()
        assert isinstance(df, pd.DataFrame)
        assert "age" in df.columns
        assert "target" in df.columns

    def test_text_labels_converted_to_numeric(self, sample_kaggle_csv: Path, monkeypatch) -> None:
        import src.data_preparation as dp
        monkeypatch.setattr(dp, "KAGGLE_PATH", sample_kaggle_csv)
        df = load_kaggle()
        for col in ["sex", "cp", "restecg", "exang", "slope"]:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} should be numeric"

    def test_zero_cholesterol_replaced(self, sample_kaggle_csv: Path, monkeypatch) -> None:
        import src.data_preparation as dp
        monkeypatch.setattr(dp, "KAGGLE_PATH", sample_kaggle_csv)
        df = load_kaggle()
        assert (df["chol"] > 0).all()

    def test_chol_imputed_flag_set(self, sample_kaggle_csv: Path, monkeypatch) -> None:
        import src.data_preparation as dp
        monkeypatch.setattr(dp, "KAGGLE_PATH", sample_kaggle_csv)
        df = load_kaggle()
        assert df["chol_imputed"].sum() == 2

    def test_zero_resting_bp_replaced(self, sample_kaggle_csv: Path, monkeypatch) -> None:
        import src.data_preparation as dp
        monkeypatch.setattr(dp, "KAGGLE_PATH", sample_kaggle_csv)
        df = load_kaggle()
        assert (df["trestbps"] > 0).all()

    def test_no_ca_or_thal_columns(self, sample_kaggle_csv: Path, monkeypatch) -> None:
        """Kaggle dataset should not have ca or thal after loading."""
        import src.data_preparation as dp
        monkeypatch.setattr(dp, "KAGGLE_PATH", sample_kaggle_csv)
        df = load_kaggle()
        assert "ca" not in df.columns
        assert "thal" not in df.columns


# --- Tests for combine_datasets ---

class TestCombineDatasets:
    """Tests for the full dataset combination pipeline."""

    def test_combines_and_saves(
        self, sample_original_csv: Path, sample_kaggle_csv: Path,
        tmp_path: Path, monkeypatch
    ) -> None:
        import src.data_preparation as dp
        monkeypatch.setattr(dp, "ORIGINAL_PATH", sample_original_csv)
        monkeypatch.setattr(dp, "KAGGLE_PATH", sample_kaggle_csv)
        combined_path = tmp_path / "heart_combined.csv"
        monkeypatch.setattr(dp, "COMBINED_PATH", combined_path)

        result = combine_datasets()

        assert isinstance(result, pd.DataFrame)
        assert combined_path.exists()
        assert len(result) > 0

    def test_no_duplicate_rows(
        self, sample_original_csv: Path, sample_kaggle_csv: Path,
        tmp_path: Path, monkeypatch
    ) -> None:
        import src.data_preparation as dp
        monkeypatch.setattr(dp, "ORIGINAL_PATH", sample_original_csv)
        monkeypatch.setattr(dp, "KAGGLE_PATH", sample_kaggle_csv)
        monkeypatch.setattr(dp, "COMBINED_PATH", tmp_path / "heart_combined.csv")

        result = combine_datasets()
        feature_cols = [c for c in result.columns if c != "chol_imputed"]
        assert result.duplicated(subset=feature_cols).sum() == 0

    def test_combined_has_11_features_plus_target_plus_flag(
        self, sample_original_csv: Path, sample_kaggle_csv: Path,
        tmp_path: Path, monkeypatch
    ) -> None:
        """Combined dataset should have 11 features + target + chol_imputed = 13 columns."""
        import src.data_preparation as dp
        monkeypatch.setattr(dp, "ORIGINAL_PATH", sample_original_csv)
        monkeypatch.setattr(dp, "KAGGLE_PATH", sample_kaggle_csv)
        monkeypatch.setattr(dp, "COMBINED_PATH", tmp_path / "heart_combined.csv")

        result = combine_datasets()
        assert "chol_imputed" in result.columns
        assert "target" in result.columns
        assert "ca" not in result.columns
        assert "thal" not in result.columns
        assert len(result.columns) == 13  # 11 features + target + chol_imputed

    def test_no_zero_cholesterol_in_combined(
        self, sample_original_csv: Path, sample_kaggle_csv: Path,
        tmp_path: Path, monkeypatch
    ) -> None:
        import src.data_preparation as dp
        monkeypatch.setattr(dp, "ORIGINAL_PATH", sample_original_csv)
        monkeypatch.setattr(dp, "KAGGLE_PATH", sample_kaggle_csv)
        monkeypatch.setattr(dp, "COMBINED_PATH", tmp_path / "heart_combined.csv")

        result = combine_datasets()
        assert (result["chol"] > 0).all()
