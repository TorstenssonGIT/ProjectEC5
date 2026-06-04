# Heart Disease Prediction Project (EC4)

A complete machine learning pipeline for heart disease prediction — covering data processing, exploratory data analysis, model training and evaluation, a terminal app, and a Streamlit web interface.

ProjectEC4 extends ProjectEC3 by combining two datasets to increase training data from 1025 to 1220 rows, and includes a dataset quality investigation that informed key feature engineering decisions.

## Project Structure

```
ProjectEC4/
├── .github/
│   └── workflows/
│       └── tests.yml              # CI/CD — runs tests on every push and PR
├── .streamlit/
│   └── config.toml                # Streamlit dark theme configuration
├── data/
│   ├── heart.csv                  # Original UCI Heart Disease dataset (1025 rows)
│   ├── heart_kaggle.csv           # Kaggle Heart Failure Prediction dataset (918 rows)
│   └── heart_combined.csv         # Combined dataset (1220 rows, generated)
├── models/                        # Saved model files (generated after training)
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── decision_tree.pkl
│   └── training_results.json
├── notebooks/
│   └── analysis.ipynb             # Full EDA, training, results, terminal app & Streamlit demo
├── scripts/
│   └── investigate_datasets.py    # Dataset quality investigation script
├── src/
│   ├── data_preparation.py        # Combines heart.csv + heart_kaggle.csv
│   ├── data_processing.py         # DataProcessor class
│   ├── model_training.py          # ModelTrainer class (3 models)
│   ├── terminal_app.py            # Terminal prediction app (HeartApp)
│   ├── utils.py                   # Logging helpers
│   └── main.py                    # CLI entry point
├── tests/
│   ├── conftest.py
│   ├── test_data_preparation.py
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

## Dataset Strategy

### Why two datasets?
ProjectEC4 combines the original UCI Heart Disease dataset with the Kaggle Heart Failure Prediction dataset (fedesoriano, 2021) to increase training data from 1025 to 1220 unique rows after deduplication.

### Dataset combination methodology (`src/data_preparation.py`)
The Kaggle dataset uses text labels for categorical variables which are converted to numeric values matching UCI encoding. Zero cholesterol values (172 rows) are replaced with age-appropriate clinical averages rather than a single global median. One zero RestingBP value is replaced with the dataset median.

### Why ca and thal were dropped
During investigation (`scripts/investigate_datasets.py`) it was found that `ca` (number of major vessels) and `thal` (thalassemia type) are the **two most important features** in the original dataset (~28% combined feature importance), but are absent from the Kaggle dataset.

Initial attempts to impute these with median values caused accuracy to drop from ~88% to ~64% — confirming that 918 rows all receiving the same median value created misleading signals for the models. Since these are clinical measurements that cannot be reliably estimated, they are dropped from both datasets before combining.

| Approach | Random Forest Accuracy |
|----------|----------------------|
| Original UCI only (1025 rows, 13 features) | ~88% |
| Combined with imputed ca/thal (1220 rows, 13 features) | ~64% |
| Combined without ca/thal (1220 rows, 11 features) | TBD after retraining |

The `chol_imputed` flag column is included in `heart_combined.csv` for transparency and auditing but is automatically dropped before model training.

## Setup

### First-time setup

```bash
# From the project root (ProjectEC4/)

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

> If you need to recreate the venv: `rm -rf .venv` then repeat above.

## Running the Project

### Step 1 — Prepare the combined dataset

```bash
python src/data_preparation.py
```

Combines `heart.csv` and `heart_kaggle.csv` into `heart_combined.csv`. Safe to re-run — always regenerates from source files. Original files are never modified.

### Step 2 — Train all three models

```bash
python src/main.py --train
```

Trains Logistic Regression, Random Forest, and Decision Tree on `heart_combined.csv`, prints a comparison table, and saves model files to `models/`.

### Train a final model on the full dataset

```bash
python src/main.py --train-full --model-path models/heart_model_full.joblib
```

### Run the terminal prediction app

```bash
python src/main.py --app
```

Launches `HeartApp` which prompts you to enter values for all 11 clinical features one by one, then returns a risk prediction and confidence score.

### Run the Streamlit web app

```bash
python src/main.py --streamlit
```

Opens the browser at `http://localhost:8501`. The dark theme is applied automatically from `.streamlit/config.toml`.

> **Note:** Models must be trained before launching the Streamlit app.

### Run everything from the notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

The notebook runs `data_preparation.py` automatically in Section 2, then covers EDA, training, evaluation, terminal app demonstration, and launches Streamlit inline.

## Model Results

All three models evaluated on a 20% held-out test split of `heart_combined.csv` (stratified, `random_state=42`), 11 features (ca and thal dropped).

| Model               | Accuracy | F1    | Precision | Recall | ROC AUC |
|---------------------|----------|-------|-----------|--------|---------|
| Logistic Regression | 0.705    | 0.719 | 0.742     | 0.697  | 0.821   |
| Random Forest       | 0.672    | 0.677 | 0.724     | 0.636  | 0.809   |
| Decision Tree       | 0.639    | 0.621 | 0.720     | 0.545  | 0.648   |

### Investigation Findings

During development a systematic investigation was conducted comparing different dataset combinations. Key findings:

| Dataset | Rows | Features | Random Forest Accuracy |
|---------|------|----------|----------------------|
| Original UCI (with duplicates) | 1025 | 13 | 1.000 — overfitting |
| Original UCI (deduplicated) | 302 | 13 | ~0.885 |
| Combined with imputed ca/thal | 1220 | 13 | ~0.643 |
| Combined without ca/thal | 1220 | 11 | 0.672 |

**Conclusions:**
- The original `heart.csv` contained 723 duplicate rows — cleaned to 302 unique rows
- `ca` and `thal` are the two most important features (~28% combined importance) but are absent from the Kaggle dataset
- Imputing them with median values caused accuracy to drop to ~64% — confirming imputation was unreliable
- Dropping them and using the combined 1220-row dataset recovers to ~67-70%
- The accuracy gap vs the original 13-feature dataset reflects the genuine information loss from dropping two clinically significant features

> Investigation script: `scripts/investigate_datasets.py`

## Testing

The project includes **66 tests** with **84% code coverage**.

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

| Module                      | Coverage |
|-----------------------------|----------|
| `src/data_preparation.py`   | 99%      |
| `src/terminal_app.py`       | 100%     |
| `src/model_training.py`     | 98%      |
| `src/data_processing.py`    | 89%      |
| `src/main.py`               | 75%      |
| `src/utils.py`              | 0%       |

## Code Quality

The project follows PEP8 standards. To check:

```bash
pip install flake8
flake8 src/ --max-line-length=100
```

## CI/CD (GitHub Actions)

The workflow at `.github/workflows/tests.yml` runs automatically on every push and pull request. It:

- Sets up Python 3.11
- Installs all dependencies
- Runs `data_preparation.py` to generate `heart_combined.csv`
- Runs the full test suite with coverage
- Enforces a minimum of 84% total coverage
- Uploads the HTML coverage report as a build artifact

To run the same checks locally:

```bash
pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=84
```

## Ethical Considerations

This project uses the UCI Heart Disease dataset which has known demographic bias — the majority of patients are middle-aged men. The models may perform worse for underrepresented groups such as women and elderly patients.

The decision to drop `ca` and `thal` features is documented transparently. These are clinically significant features but cannot be reliably estimated for the Kaggle patients without actual measurement.

The predictions are for **educational purposes only** and must not be used as a substitute for medical diagnosis. Always consult a healthcare professional.

A full ethical reflection is included in `report.md`.

## Notes

- Code style checked with `flake8 src/ --max-line-length=100` — no issues.
- `notebooks/analysis.ipynb` is the single entry point — run all cells top-to-bottom.
- The terminal app uses 11 features (ca and thal excluded).
- The Streamlit app uses a dark theme configured in `.streamlit/config.toml`.
- `heart.zip` contains the original dataset archive for reference.
- This application is for **educational purposes only** and is not a medical diagnostic tool.
