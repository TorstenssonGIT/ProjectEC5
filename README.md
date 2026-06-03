# Heart Disease Prediction Project

A complete machine learning pipeline for the UCI Heart Disease dataset — covering data processing, exploratory data analysis, model training and evaluation, a terminal app, and a Streamlit web interface.

## Project Structure

```
ProjectEC3/
├── data/
│   └── heart.csv                  # UCI Heart Disease dataset
├── models/                        # Saved model files (generated after training)
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── training_results.json
├── notebooks/
│   └── analysis.ipynb             # Full EDA, training, results & Streamlit demo
├── src/
│   ├── data_processing.py         # DataProcessor class
│   ├── model_training.py          # ModelTrainer class
│   ├── terminal_app.py            # Terminal prediction app (HeartApp)
│   ├── app.py                     # Streamlit web app (run via src/main.py)
│   ├── utils.py                   # Logging helpers
│   └── main.py                    # CLI entry point
├── tests/
│   ├── conftest.py
│   ├── test_data_processing.py
│   ├── test_model_training.py
│   ├── test_app.py
│   └── test_main.py
├── heart.zip                      # Original dataset archive
├── main.py                        # Root-level shim → calls src/main.py
├── requirements.txt
├── report.md
├── TEST_DOCUMENTATION.md
└── README.md
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

### Train models and save to `models/`

```bash
python src/main.py --train
```

This trains both Logistic Regression and Random Forest, prints a comparison table, and saves:
- `models/logistic_regression.pkl`
- `models/random_forest.pkl`
- `models/training_results.json`

### Train a final model on the full dataset

```bash
python src/main.py --train-full --model-path models/heart_model_full.joblib
```

### Run the terminal prediction app

```bash
python src/main.py --app
```

Prompts you to enter patient values and returns a risk prediction using the saved Random Forest model.

### Run the Streamlit web app

```bash
python src/main.py --streamlit
```

Opens the browser at `http://localhost:8501` with four pages: Home, Prediction, Model Performance, and About.

> **Note:** Models must be trained before launching the Streamlit app.

### Run everything from the notebook

Open `notebooks/analysis.ipynb` in Jupyter and run all cells top-to-bottom. The notebook covers EDA, training, evaluation, and launches the Streamlit app inline with an embedded screenshot of its UI.

```bash
jupyter notebook notebooks/analysis.ipynb
```

## Model Results

Both models are evaluated on a 20% held-out test split (stratified, `random_state=42`).

| Model               | Accuracy | F1    | Precision | Recall | ROC AUC |
|---------------------|----------|-------|-----------|--------|---------|
| Logistic Regression | 0.869    | 0.877 | 0.861     | 0.893  | 0.935   |
| Random Forest       | 0.885    | 0.893 | 0.878     | 0.909  | 0.955   |
| Decision Tree       | 0.754    | 0.771 | 0.750     | 0.793  | 0.754   |

Random Forest was selected as the primary model based on highest accuracy and ROC AUC. Decision Tree is included to demonstrate the value of ensembling — Random Forest's accuracy is notably higher despite being built from the same base learner.

> Exact values may vary slightly between runs despite the fixed seed, depending on scikit-learn version.

## Testing

The project includes **44 tests** with **92% code coverage**.

### Run all tests

```bash
pytest tests/ -v
```

### Generate HTML coverage report

```bash
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html in a browser
```

### Coverage by module

| Module                   | Coverage |
|--------------------------|----------|
| `src/app.py`             | 100%     |
| `src/model_training.py`  | 98%      |
| `src/data_processing.py` | 89%      |
| `src/main.py`            | 79%      |

See [TEST_DOCUMENTATION.md](TEST_DOCUMENTATION.md) for full details.

## CI/CD (GitHub Actions)

Add the following file as `.github/workflows/tests.yml` to run tests automatically on every push:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest tests/ --cov=src --cov-fail-under=85
```

## Notes

- `notebooks/analysis.ipynb` documents all major steps, results, and the ethical reflection.
- The terminal app (`src/terminal_app.py`) uses the trained Random Forest model for interactive predictions.
- `heart.zip` contains the original dataset archive for reference.
- This application is for **educational purposes only** and is not a medical diagnostic tool.
