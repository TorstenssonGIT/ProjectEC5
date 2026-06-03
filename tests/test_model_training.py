"""Tests for model_training module."""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

from src.model_training import ModelTrainer, ModelResult


class TestModelTrainer:
    """Test cases for ModelTrainer class."""

    def test_initialization(self) -> None:
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(random_state=42)
        assert trainer.random_state == 42
        assert trainer.results == {}

    def test_build_pipelines(self) -> None:
        """Test building model pipelines."""
        trainer = ModelTrainer()
        pipelines = trainer.build_pipelines()

        assert isinstance(pipelines, dict)
        assert 'Logistic Regression' in pipelines
        assert 'Random Forest' in pipelines
        assert 'Decision Tree' in pipelines
        assert len(pipelines) == 3

        for name, pipeline in pipelines.items():
            assert hasattr(pipeline, 'fit')
            assert hasattr(pipeline, 'predict')

    def test_decision_tree_in_pipelines(self) -> None:
        """Test that Decision Tree pipeline is correctly configured."""
        trainer = ModelTrainer(random_state=42)
        pipelines = trainer.build_pipelines()

        dt_pipeline = pipelines['Decision Tree']
        clf = dt_pipeline.named_steps['model']

        assert isinstance(clf, DecisionTreeClassifier)
        assert clf.max_depth == 10
        assert clf.random_state == 42

    def test_train_models(self) -> None:
        """Test training models."""
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(80, 13))
        y_train = pd.Series(np.random.randint(0, 2, 80))

        trainer = ModelTrainer(random_state=42)
        results = trainer.train_models(X_train, y_train)

        assert len(results) == 3
        assert 'Logistic Regression' in results
        assert 'Random Forest' in results
        assert 'Decision Tree' in results

        for name, result in results.items():
            assert isinstance(result, ModelResult)
            assert result.name == name
            assert hasattr(result.pipeline, 'predict')

    def test_evaluate_models(self) -> None:
        """Test model evaluation."""
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(80, 13))
        y_train = pd.Series(np.random.randint(0, 2, 80))
        X_test = pd.DataFrame(np.random.randn(20, 13))
        y_test = pd.Series(np.random.randint(0, 2, 20))

        trainer = ModelTrainer(random_state=42)
        trainer.train_models(X_train, y_train)
        results = trainer.evaluate(X_test, y_test)

        for result in results.values():
            assert result.accuracy > 0
            assert 0 <= result.accuracy <= 1
            assert 0 <= result.f1 <= 1
            assert 0 <= result.precision <= 1
            assert 0 <= result.recall <= 1
            assert 0 <= result.roc_auc <= 1
            assert result.confusion_matrix is not None
            assert result.classification_report is not None

    def test_decision_tree_evaluation(self) -> None:
        """Test that Decision Tree produces valid evaluation metrics."""
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(80, 13))
        y_train = pd.Series(np.random.randint(0, 2, 80))
        X_test = pd.DataFrame(np.random.randn(20, 13))
        y_test = pd.Series(np.random.randint(0, 2, 20))

        trainer = ModelTrainer(random_state=42)
        trainer.train_models(X_train, y_train)
        results = trainer.evaluate(X_test, y_test)

        dt_result = results['Decision Tree']
        assert 0 <= dt_result.accuracy <= 1
        assert 0 <= dt_result.roc_auc <= 1
        assert dt_result.confusion_matrix.shape == (2, 2)

    def test_compare_models(self) -> None:
        """Test model comparison."""
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(80, 13))
        y_train = pd.Series(np.random.randint(0, 2, 80))
        X_test = pd.DataFrame(np.random.randn(20, 13))
        y_test = pd.Series(np.random.randint(0, 2, 20))

        trainer = ModelTrainer(random_state=42)
        trainer.train_models(X_train, y_train)
        trainer.evaluate(X_test, y_test)

        comparison = trainer.compare()

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 3
        assert 'model' in comparison.columns
        assert 'accuracy' in comparison.columns
        assert 'f1_score' in comparison.columns
        assert 'precision' in comparison.columns
        assert 'recall' in comparison.columns
        assert 'roc_auc' in comparison.columns
        assert 'Decision Tree' in comparison['model'].values
        assert comparison['accuracy'].is_monotonic_decreasing

    def test_save_best_model(self, temp_model_output: str) -> None:
        """Test saving the best model."""
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(80, 13))
        y_train = pd.Series(np.random.randint(0, 2, 80))
        X_test = pd.DataFrame(np.random.randn(20, 13))
        y_test = pd.Series(np.random.randint(0, 2, 20))

        trainer = ModelTrainer(random_state=42)
        trainer.train_models(X_train, y_train)
        trainer.evaluate(X_test, y_test)

        best_name = trainer.save_best_model(temp_model_output)

        assert isinstance(best_name, str)
        assert best_name in ['Logistic Regression', 'Random Forest', 'Decision Tree']
        assert os.path.isfile(temp_model_output)

    def test_save_best_model_no_training(self, temp_model_output: str) -> None:
        """Test saving best model when no models are trained."""
        trainer = ModelTrainer()

        with pytest.raises(ValueError, match="No models have been trained"):
            trainer.save_best_model(temp_model_output)

    def test_model_result_dataclass(self) -> None:
        """Test ModelResult dataclass."""
        pipeline = LogisticRegression()
        result = ModelResult(
            name="Test Model",
            pipeline=pipeline,
            accuracy=0.95,
            f1=0.92,
            precision=0.90,
            recall=0.94,
            roc_auc=0.98
        )

        assert result.name == "Test Model"
        assert result.accuracy == 0.95
        assert result.f1 == 0.92
        assert result.precision == 0.90
        assert result.recall == 0.94
        assert result.roc_auc == 0.98

    def test_random_state_reproducibility(self) -> None:
        """Test that same random state produces reproducible results."""
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(80, 13))
        y_train = pd.Series(np.random.randint(0, 2, 80))
        X_test = pd.DataFrame(np.random.randn(20, 13))
        y_test = pd.Series(np.random.randint(0, 2, 20))

        trainer1 = ModelTrainer(random_state=42)
        trainer1.train_models(X_train, y_train)
        trainer1.evaluate(X_test, y_test)
        acc1 = trainer1.results['Logistic Regression'].accuracy

        trainer2 = ModelTrainer(random_state=42)
        trainer2.train_models(X_train, y_train)
        trainer2.evaluate(X_test, y_test)
        acc2 = trainer2.results['Logistic Regression'].accuracy

        assert acc1 == acc2
