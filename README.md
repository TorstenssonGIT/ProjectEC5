# Heart Disease Prediction Project

A complete machine learning pipeline for the UCI Heart Disease dataset — covering data processing, exploratory data analysis, model training and evaluation, a terminal app, and a Streamlit web interface.

## Project Structure

```
ProjectEC3/
├── .github/
│   └── workflows/
│       └── tests.yml              # CI/CD — runs tests on every push and PR
├── .streamlit/
│   └── config.toml                # Streamlit dark theme configuration
├── data/
│   └── heart.csv                  # UCI Heart Disease dataset
├── models/                        # Saved model files (generated after training)
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── decision_tree.pkl
│   └── training_results.json
├── notebooks/
│   └── analysis.ipynb             # Full EDA, training, results, terminal app & Streamlit demo
├── src/
│   ├── data_processing.py         # DataProcessor class
│   ├── model_training.py          # ModelTrainer class (Logistic Regression, Random Forest, Decision Tree)
│   ├── terminal_app.py            # Terminal prediction app (HeartApp)
│   ├── utils.py                   # Logging helpers
│   └── main.py                    # CLI entry point
├── tests/
│   ├── conftest.py
│   ├── test_data_processing.py
│   ├── test_model_training.py
│   ├── test_app.py
│   └── test_main.py
├── app.py                         # Streamlit web app
├── heart.zip                      # Original dataset archive
├── requirements.txt
├── report.md
├── TEST_DOCUMENTATION.md
└── README.md
```

## Setup

### First-time setup

```bash
# From the project root (ProjectEC3/)

# Create venv with Python 3.11 (recommended — matches CI)
py -3.11 -m venv .venv

# Activate
# Windows:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> If you need to recreate the venv (e.g. wrong Python version): `rm -rf .venv` then repeat above.

## Running the Project

### Train all three models and save to `models/`

```bash
python src/main.py --train
```

Trains Logistic Regression, Random Forest, and Decision Tree, prints a comparison table, and saves:
- `models/logistic_regression.pkl`
- `models/random_forest.pkl`
- `models/decision_tree.pkl`
- `models/training_results.json`

### Train a final model on the full dataset

```bash
python src/main.py --train-full --model-path models/heart_model_full.joblib
```

### Run the terminal prediction app

```bash
python src/main.py --app
```

Launches `HeartApp` which prompts you to enter values for all 13 clinical features one by one, then returns a risk prediction and confidence score. You can assess multiple patients in one session.

### Run the Streamlit web app

```bash
python src/main.py --streamlit
```

Opens the browser at `http://localhost:8501` with four pages: Home, Prediction, Model Performance, and About. The dark theme is applied automatically from `.streamlit/config.toml`.

> **Note:** Models must be trained before launching the Streamlit app.

### Run everything from the notebook

Open `notebooks/analysis.ipynb` and run all cells top-to-bottom. The notebook covers EDA, training, evaluation, terminal app demonstration, and launches the Streamlit app inline.

```bash
jupyter notebook notebooks/analysis.ipynb
```

## Model Results

All three models are evaluated on a 20% held-out test split (stratified, `random_state=42`).

| Model               | Accuracy | F1    | Precision | Recall | ROC AUC |
|---------------------|----------|-------|-----------|--------|---------|
| Logistic Regression | 0.869    | 0.877 | 0.861     | 0.893  | 0.935   |
| Random Forest       | 0.885    | 0.893 | 0.878     | 0.909  | 0.955   |
| Decision Tree       | 0.754    | 0.771 | 0.750     | 0.793  | 0.754   |

Random Forest was selected as the primary model based on highest accuracy and ROC AUC. Decision Tree is included to demonstrate the value of ensembling — Random Forest's accuracy is notably higher despite being built from the same base learner. In a medical context **recall** is the most critical metric since a missed disease case is more costly than a false alarm.

> Exact values may vary slightly between runs despite the fixed seed, depending on scikit-learn version.

## Testing

The project includes **47 tests** with **84% code coverage**.

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
| `src/terminal_app.py`    | 100%     |
| `src/model_training.py`  | 98%      |
| `src/data_processing.py` | 89%      |
| `src/main.py`            | 75%      |
| `src/utils.py`           | 0%       |

See [TEST_DOCUMENTATION.md](TEST_DOCUMENTATION.md) for full details.

## CI/CD (GitHub Actions)

The workflow at `.github/workflows/tests.yml` runs automatically on every push and pull request. It:

- Sets up Python 3.11
- Installs all dependencies
- Runs the full test suite with coverage
- Enforces a minimum of 84% total coverage
- Uploads the HTML coverage report as a build artifact

To run the same checks locally:

```bash
pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=84
```

## Notes

- `notebooks/analysis.ipynb` is the single entry point — it documents all steps including EDA, training, evaluation, terminal app, Streamlit, ethical reflection, and CI/CD.
- The terminal app (`src/terminal_app.py`) uses the trained Random Forest model for interactive predictions.
- The Streamlit app (`app.py`) uses a dark theme configured in `.streamlit/config.toml`.
- `heart.zip` contains the original dataset archive for reference.
- This application is for **educational purposes only** and is not a medical diagnostic tool.
