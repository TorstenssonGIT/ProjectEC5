from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from joblib import dump
from src.terminal_app import HeartApp
from src.data_processing import DataProcessor
from src.model_training import ModelTrainer

# Resolve the project root for consistent path handling
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
TRAINING_RESULTS_PATH = MODELS_DIR / "training_results.json"
APP_FILE = PROJECT_ROOT / "app.py"


def ensure_models_dir() -> None:
    """Creates the models directory if it doesn't already exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def save_training_results(trainer: ModelTrainer) -> None:
    """Serializes the evaluation metrics for all trained models into a JSON file."""
    ensure_models_dir()
    results = trainer.compare().set_index("model").T.to_dict()
    TRAINING_RESULTS_PATH.write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


def save_model_pipelines(trainer: ModelTrainer) -> None:
    """Saves individual Scikit-learn pipelines to disk using joblib."""
    ensure_models_dir()
    for name, result in trainer.results.items():
        file_name = name.lower().replace(" ", "_") + ".pkl"
        output_file = MODELS_DIR / file_name
        dump(result.pipeline, output_file)


def train_and_save(data_path: str, output_path: str) -> None:
    """
    Executes the standard training workflow:
    1. Load/Clean data -> 2. Split data -> 3. Train models -> 
    4. Evaluate performance -> 5. Persist models and metrics.
    """
    processor = DataProcessor(data_path)
    processor.load_data()
    processor.clean_data()
    X_train, X_test, y_train, y_test = processor.split_data()

    trainer = ModelTrainer()
    trainer.train_models(X_train, y_train)
    trainer.evaluate(X_test, y_test)

    print("Model comparison:\n")
    print(trainer.compare().to_string(index=False))

    # Persist the artifacts
    save_model_pipelines(trainer)
    save_training_results(trainer)

    best_name = trainer.save_best_model(output_path)
    print(f"Saved best model ({best_name}) to {output_path}")
    print(f"Saved individual model pipelines to {MODELS_DIR}")
    print(f"Saved training results to {TRAINING_RESULTS_PATH}")


def train_and_save_full(data_path: str, output_path: str) -> None:
    """Trains the models using the entire dataset (no split) for production usage."""
    processor = DataProcessor(data_path)
    processor.load_data()
    processor.clean_data()
    X, y = processor.get_features_and_target()

    trainer = ModelTrainer()
    trainer.train_models(X, y)

    print("Training final model on the full dataset...")
    best_name = trainer.save_best_model(output_path)
    print(f"Saved full-data model ({best_name}) to {output_path}")


def run_app(model_path: str) -> None:
    """Launches the interactive terminal-based prediction application."""
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    app = HeartApp(model_path)
    app.run()


def run_streamlit(port: int) -> None:
    """Starts the Streamlit web application as a background process."""
    if not APP_FILE.exists():
        raise FileNotFoundError(f"Streamlit app file not found: {APP_FILE}")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_FILE),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    print(f"Starting Streamlit app on port {port}...")
    subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
    print(f"Streamlit server launched. Open http://localhost:{port}")


def configure_arg_parser() -> argparse.ArgumentParser:
    """Configures the Command Line Interface (CLI) arguments."""
    parser = argparse.ArgumentParser(
        description="Heart Disease ML workflow: train, evaluate, and run the prediction app"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train models and save the best-performing model",
    )
    parser.add_argument(
        "--train-full",
        action="store_true",
        help="Train a final model on the full dataset and save it",
    )
    parser.add_argument(
        "--app",
        action="store_true",
        help="Run the terminal prediction application",
    )
    parser.add_argument(
        "--streamlit",
        action="store_true",
        help="Launch the Streamlit application",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for the Streamlit application",
    )
    parser.add_argument(
        "--model-path",
        default="models/heart_model.joblib",
        help="Path to save or load the trained model",
    )
    parser.add_argument(
        "--data-path",
        default="data/heart.csv",
        help="Path to the heart disease dataset CSV file",
    )
    return parser


def main() -> None:
    """
    Entry point for the CLI. 
    Dispatches the workflow based on user flags.
    """
    parser = configure_arg_parser()
    args = parser.parse_args()

    if args.train:
        train_and_save(args.data_path, args.model_path)

    if args.train_full:
        train_and_save_full(args.data_path, args.model_path)

    if args.app:
        run_app(args.model_path)

    if args.streamlit:
        run_streamlit(args.port)

    if not args.train and not args.train_full and not args.app and not args.streamlit:
        parser.print_help()


if __name__ == "__main__":
    main()
