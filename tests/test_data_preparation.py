"""Tests for data_preparation module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_preparation import (
    UCI_COLUMNS,
    prepare_dataset,
)


# --- Fixtures ---

@pytest.fixture
def sample_cleveland_data(tmp_path: Path) -> Path:
    """Create a temporary UCI-format data file with ca and thal included."""
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
        "ca":       np.random.randint(0, 4, n),
        "thal":     np.random.choice([3, 6, 7], n),
        "target":   np.random.randint(0, 5, n),  # multi-class like real UCI
    })
    path = tmp_path / "processed.cleveland.data"
    df.to_csv(path, index=False, header=False)
    return path


@pytest.fixture
def sample_cleveland_with_missing(tmp_path: Path) -> Path:
    """Create a temporary UCI file with some missing values (like real data)."""
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
        "ca":       np.random.randint(0, 4, n),
        "thal":     np.random.choice([3, 6, 7], n),
        "target":   np.random.randint(0, 5, n),
    })
    # Introduce missing values like real UCI data uses "?"
    df = df.astype(object)
    df.loc[0, "ca"] = "?"
    df.loc[1, "thal"] = "?"
    path = tmp_path / "processed.cleveland.data"
    df.to_csv(path, index=False, header=False)
    return path


# --- Tests for prepare_dataset ---

class TestPrepareDataset:
    """Tests for the main data preparation function."""

    def test_loads_and_saves(
        self, sample_cleveland_data: Path, tmp_path: Path, monkeypatch
    ) -> None:
        """Should load data, prepare it and save to combined path."""
        import src.data_preparation as dp
        monkeypatch.setattr(dp, "CLEVELAND_PATH", sample_cleveland_data)
        combined_path = tmp_path / "heart_combined.csv"
        monkeypatch.setattr(dp, "COMBINED_PATH", combined_path)

        result = prepare_dataset()

        assert isinstance(result, pd.DataFrame)
        assert combined_path.exists()
        assert len(result) == 50

    def test_target_converted_to_binary(
        self, sample_cleveland_data: Path, tmp_path: Path, monkeypatch
    ) -> None:
        """Target should be binary (0 or 1) after preparation."""
        import src.data_preparation as dp
        monkeypatch.setattr(dp, "CLEVELAND_PATH", sample_cleveland_data)
        monkeypatch.setattr(dp, "COMBINED_PATH", tmp_path / "heart_combined.csv")

        result = prepare_dataset()

        assert set(result["target"].unique()).issubset({0, 1})

    def test_has_all_13_features(
        self, sample_cleveland_data: Path, tmp_path: Path, monkeypatch
    ) -> None:
        """Dataset should have all 13 features including ca and thal."""
        import src.data_preparation as dp
        monkeypatch.setattr(dp, "CLEVELAND_PATH", sample_cleveland_data)
        monkeypatch.setattr(dp, "COMBINED_PATH", tmp_path / "heart_combined.csv")

        result = prepare_dataset()

        assert "ca" in result.columns
        assert "thal" in result.columns
        assert "target" in result.columns
        assert len(result.columns) == 14  # 13 features + target

    def test_no_missing_values_after_preparation(
        self, sample_cleveland_with_missing: Path, tmp_path: Path, monkeypatch
    ) -> None:
        """Missing values should be imputed after preparation."""
        import src.data_preparation as dp
        monkeypatch.setattr(dp, "CLEVELAND_PATH", sample_cleveland_with_missing)
        monkeypatch.setattr(dp, "COMBINED_PATH", tmp_path / "heart_combined.csv")

        result = prepare_dataset()

        assert result.isnull().sum().sum() == 0

    def test_raises_if_file_not_found(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Should raise FileNotFoundError if source file is missing."""
        import src.data_preparation as dp
        monkeypatch.setattr(dp, "CLEVELAND_PATH", tmp_path / "nonexistent.data")
        monkeypatch.setattr(dp, "COMBINED_PATH", tmp_path / "heart_combined.csv")

        with pytest.raises(FileNotFoundError):
            prepare_dataset()

    def test_column_names_match_uci(
        self, sample_cleveland_data: Path, tmp_path: Path, monkeypatch
    ) -> None:
        """Column names should match UCI_COLUMNS."""
        import src.data_preparation as dp
        monkeypatch.setattr(dp, "CLEVELAND_PATH", sample_cleveland_data)
        monkeypatch.setattr(dp, "COMBINED_PATH", tmp_path / "heart_combined.csv")

        result = prepare_dataset()

        expected = [c for c in UCI_COLUMNS]
        assert result.columns.tolist() == expected

    def test_saved_csv_matches_returned_dataframe(
        self, sample_cleveland_data: Path, tmp_path: Path, monkeypatch
    ) -> None:
        """Saved CSV should match the returned dataframe."""
        import src.data_preparation as dp
        monkeypatch.setattr(dp, "CLEVELAND_PATH", sample_cleveland_data)
        combined_path = tmp_path / "heart_combined.csv"
        monkeypatch.setattr(dp, "COMBINED_PATH", combined_path)

        result = prepare_dataset()
        saved = pd.read_csv(combined_path)

        assert len(result) == len(saved)
        assert result.columns.tolist() == saved.columns.tolist()
