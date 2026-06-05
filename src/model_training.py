from __future__ import annotations

from dataclasses import dataclass, field
from joblib import dump
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    accuracy: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    roc_auc: float = 0.0
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.empty(0))
    classification_report: Dict[str, Any] = field(default_factory=dict)


class ModelTrainer:
    """Train, tune and evaluate classification pipelines."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.results: Dict[str, ModelResult] = {}

    def build_pipelines(self) -> Dict[str, Pipeline]:
        """Build model pipelines for training."""
        return {
            "Logistic Regression": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000, random_state=self.random_state
                        ),
                    ),
                ]
            ),
            "Random Forest": Pipeline(
                [
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=200,
                            random_state=self.random_state,
                            n_jobs=-1,
                        ),
                    )
                ]
            ),
            "Decision Tree": Pipeline(
                [
                    (
                        "model",
                        DecisionTreeClassifier(
                            max_depth=10,
                            random_state=self.random_state,
                        ),
                    )
                ]
            ),
            "XGBoost": Pipeline(
                [
                    (
                        "model",
                        XGBClassifier(
                            n_estimators=200,
                            random_state=self.random_state,
                            eval_metric="logloss",
                            verbosity=0,
                        ),
                    )
                ]
            ),
        }

    def get_param_grids(self) -> Dict[str, Dict]:
        """Return hyperparameter grids for GridSearchCV tuning."""
        return {
            "Logistic Regression": {
                "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "model__solver": ["lbfgs", "liblinear"],
            },
            "Random Forest": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [None, 5, 10, 15],
                "model__min_samples_split": [2, 5, 10],
                "model__max_features": ["sqrt", "log2"],
            },
            "Decision Tree": {
                "model__max_depth": [3, 5, 7, 10, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__criterion": ["gini", "entropy"],
            },
            "XGBoost": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.1, 0.2],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
            },
        }

    def train_models(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Dict[str, ModelResult]:
        """Train all pipelines with default parameters."""
        pipelines = self.build_pipelines()
        for name, pipeline in pipelines.items():
            pipeline.fit(X_train, y_train)
            self.results[name] = ModelResult(name=name, pipeline=pipeline)
        return self.results

    def tune_models(
        self, X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5
    ) -> Dict[str, ModelResult]:
        """
        Tune all models using GridSearchCV with cross-validation.
        Replaces default pipelines with best found parameters.

        Args:
            X_train: Training features
            y_train: Training target
            cv: Number of cross-validation folds (default 5)
        """
        pipelines = self.build_pipelines()
        param_grids = self.get_param_grids()

        for name, pipeline in pipelines.items():
            print(f"Tuning {name}...")
            grid_search = GridSearchCV(
                pipeline,
                param_grids[name],
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
                verbose=0,
            )
            grid_search.fit(X_train, y_train)

            print(f"  Best params : {grid_search.best_params_}")
            print(f"  Best CV AUC : {grid_search.best_score_:.4f}")

            self.results[name] = ModelResult(
                name=name, pipeline=grid_search.best_estimator_
            )

        return self.results

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, ModelResult]:
        """Compute evaluation metrics for every trained model."""
        for result in self.results.values():
            predictions = result.pipeline.predict(X_test)
            if hasattr(result.pipeline, "predict_proba"):
                probabilities = result.pipeline.predict_proba(X_test)[:, 1]
            else:
                probabilities = result.pipeline.decision_function(X_test)

            result.accuracy = accuracy_score(y_test, predictions)
            result.f1 = f1_score(y_test, predictions)
            result.precision = precision_score(y_test, predictions)
            result.recall = recall_score(y_test, predictions)
            result.roc_auc = roc_auc_score(y_test, probabilities)
            result.confusion_matrix = confusion_matrix(y_test, predictions)
            result.classification_report = classification_report(
                y_test, predictions, output_dict=True
            )
        return self.results

    def compare(self) -> pd.DataFrame:
        """Return a comparison table of trained model metrics."""
        rows = []
        for result in self.results.values():
            rows.append(
                {
                    "model": result.name,
                    "accuracy": result.accuracy,
                    "f1_score": result.f1,
                    "precision": result.precision,
                    "recall": result.recall,
                    "roc_auc": result.roc_auc,
                }
            )
        return pd.DataFrame(rows).sort_values(by="accuracy", ascending=False)

    def save_best_model(self, output_path: str) -> str:
        """Save the best performing model to disk."""
        if not self.results:
            raise ValueError("No models have been trained yet.")

        best_name = max(
            self.results, key=lambda name: self.results[name].accuracy
        )
        dump(self.results[best_name].pipeline, output_path)
        return best_name
