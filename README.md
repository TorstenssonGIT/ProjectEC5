# Heart Disease Prediction Project (EC5.1)

A complete machine learning pipeline for heart disease prediction — covering data processing, exploratory data analysis, model training and evaluation, a terminal app, and a Streamlit web interface.

**ProjectEC5.1** is a patch release that fixes the following issues found in EC5:

| Issue | Fix |
|-------|-----|
| Section numbering broken | Sections now run 1–15 sequentially |
| `pr_checks.yml` missing | Added — flake8, bandit, mypy on every PR |
| Test count regression (44 → 66) | Restored to 66 tests matching EC4 |
| Node.js 20 deprecation | Both workflows upgraded to Node.js 24 |
| Summary table incomplete | Updated to reflect all EC5 features |
| Ethical reflection outdated | Updated to mention XGBoost and SHAP |

ProjectEC5 extended ProjectEC4 by adding XGBoost as a fourth model and SHAP explainability. Built on the Cleveland dataset (303 rows, 13 features).

## Project Structure

```
ProjectEC5/
├── .github/
│   └── workflows/
│       ├── tests.yml              # CI/CD — runs on every push
│       └── pr_checks.yml          # PR gates — flake8, bandit, mypy (added EC5.1)
├── .streamlit/
│   └── config.toml                # Streamlit dark theme configuration
├── data/
│   ├── heart.csv                  # Original UCI Cleveland dataset (legacy)
│   ├── processed.cleveland.data   # UCI Cleveland (303 rows, 13 features) — primary
│   ├── processed.hungarian.data   # UCI Hungarian (294 rows) — research only
│   ├── processed.switzerland.data # UCI Switzerland (123 rows) — research only
│   ├── processed.va.data          # UCI VA Long Beach (200 rows) — research only
│   └── heart_combined.csv         # Prepared dataset (303 rows, generated)
├── models/                        # Saved model files (generated after training)
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── decision_tree.pkl
│   └── training_results.json
├── notebooks/
│   └── analysis.ipynb             # Full EDA, training, XGBoost, SHAP, Streamlit demo
├── scripts/
│   └── investigate_datasets.py    # Dataset quality investigation script
├── src/
│   ├── data_preparation.py        # Prepares Cleveland dataset → heart_combined.csv
│   ├── data_processing.py         # DataProcessor class
│   ├── model_training.py          # ModelTrainer class (LR, RF, DT + GridSearchCV tuning)
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

### Why Cleveland only?
After a systematic investigation (`scripts/investigate_datasets.py`), the Cleveland dataset was chosen as the primary source. It is the only UCI hospital dataset with reliable measurements for all 13 features including `ca` and `thal` — the two most important predictive features (~28% combined importance).

| Dataset | ca missing | thal missing | Rows |
|---------|-----------|-------------|------|
| Cleveland | 4 | 2 | 303 |
| Hungarian | 291 | 266 | 294 |
| Switzerland | 118 | 52 | 123 |
| VA Long Beach | 198 | 166 | 200 |

Imputing `ca` and `thal` for 90%+ of rows in three datasets degraded accuracy from ~88% to ~64%. Cleveland only with all 13 features gives the best results.

| Approach | Random Forest Accuracy |
|----------|----------------------|
| Original UCI (with duplicates) | 1.000 — overfitting |
| Original UCI (deduplicated) | ~0.885 |
| Combined with imputed ca/thal | ~0.643 |
| Combined without ca/thal | 0.672 |
| Cleveland only, 13 features | **0.902 ← selected** |

## Setup

```bash
# From the project root (ProjectEC5/)
py -3.11 -m venv .venv

# Windows:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Running the Project

### Step 1 — Prepare the dataset
```bash
py -m src.data_preparation
```

### Step 2 — Train all models
```bash
py -m src.main --train
```

### Step 3 — Terminal prediction app
```bash
py -m src.main --app
```

### Step 4 — Streamlit web app
```bash
py -m src.main --streamlit
```

### Run everything from the notebook
```bash
jupyter notebook notebooks/analysis.ipynb
```

## Model Results — Evolution from EC3 to EC5

### EC3 — baseline (1025 rows, with duplicates)
| Model | Accuracy | ROC AUC | Note |
|-------|----------|---------|------|
| Random Forest | 0.885 | 0.955 | ⚠️ Inflated — 723 duplicate rows |
| Logistic Regression | 0.869 | 0.935 | ⚠️ Inflated |
| Decision Tree | 0.754 | 0.754 | ⚠️ Inflated |

### EC4 — goal: add Kaggle dataset as new data source → finding: more data did not help

| Approach | Rows | Features | RF Accuracy | Conclusion |
|----------|------|----------|------------|------------|
| EC3 original (with duplicates) | 1025 | 13 | 1.000 | ❌ Overfitting — data leakage |
| EC3 deduplicated | 302 | 13 | ~0.885 | Honest baseline |
| UCI 4 hospitals — drop ca + thal | 918 | 11 | ~0.672 | ❌ Feature loss too costly |
| Kaggle combined — impute ca + thal | 1220 | 13 | ~0.643 | ❌ Imputing 90%+ rows unreliable |
| **Cleveland only — all 13 features** | **303** | **13** | **0.902** | ✅ Selected |

`ca` and `thal` are the two most important features (~28% combined importance). The Kaggle dataset
does not include them — adding it requires dropping or imputing these features. Both options degrade
accuracy more than the extra rows help. **Data quality beats data quantity.**

### EC4 — after deduplication + hyperparameter tuning (303 rows, 13 features)
| Model | Default Acc | Tuned Acc | Default AUC | Tuned AUC |
|-------|------------|-----------|-------------|-----------|
| Random Forest | 0.902 | 0.902 | 0.955 | 0.958 |
| Logistic Regression | 0.869 | 0.853 | 0.951 | 0.958 |
| Decision Tree | 0.787 | **0.869** | 0.808 | **0.871** |

Decision Tree was the biggest winner from tuning (+8.2% accuracy).

### EC5 — with XGBoost (303 rows, 13 features, actual results)
| Model | Accuracy | F1 | Precision | Recall | ROC AUC |
|-------|----------|----|-----------|--------|---------|
| Random Forest | 0.9016 | 0.9000 | 0.8438 | **0.9643** | 0.9545 |
| Logistic Regression | 0.8689 | 0.8667 | 0.8125 | 0.9286 | 0.9513 |
| XGBoost | 0.8689 | 0.8667 | 0.8125 | 0.9286 | 0.9102 |
| Decision Tree | 0.7869 | 0.7937 | 0.7143 | 0.8929 | 0.8084 |
| Decision Tree (tuned) | **0.869** | — | — | — | **0.871** |

Random Forest is the best model. In a medical context **Recall** is the most critical metric — Random Forest leads at 96.4%. XGBoost matches Logistic Regression baseline without tuning.

### Evolution Summary — Conclusions

| Version | Goal | Key finding | Conclusion |
|---------|------|-------------|------------|
| **EC3** | Build baseline pipeline | Results looked strong (RF 0.885) | ⚠️ Later found to be inflated by 723 duplicate rows |
| **EC4** | Add Kaggle as new data source | More data degraded accuracy from 0.885 → 0.643 | Data quality beats data quantity — Cleveland only wins |
| **EC4** | Understand why more data hurt | `ca` + `thal` missing from Kaggle (~28% importance) | Dropping/imputing key features costs more than extra rows gain |
| **EC4** | Tune models on best dataset | Decision Tree biggest winner (+8.2%) | GridSearchCV with cv=5 recovers weak models |
| **EC5** | Add XGBoost on validated dataset | Matches LR baseline without tuning | Confirms dataset quality — strong models need good data first |
| **EC5** | Explain predictions (address ethics) | SHAP confirms `thal`, `ca`, `cp` are top features | Findings from EC4 investigation validated by explainability |

**The evolving system:** EC3 built the foundation. EC4 proved that engineering rigour matters more
than adding data — the investigation, deduplication and dataset justification turned a flawed 0.885
into an honest 0.902. EC5 built on that solid foundation to add a competitive 4th model and
explainability that validates every previous decision.

### Hyperparameter Tuning Results

| Model | Default Acc | Tuned Acc | Default ROC AUC | Tuned ROC AUC |
|-------|-------------|-----------|-----------------|---------------|
| Logistic Regression | 0.869 | 0.853 | 0.951 | 0.958 |
| Random Forest | 0.902 | 0.902 | 0.955 | 0.958 |
| Decision Tree | 0.787 | 0.869 | 0.808 | 0.871 |

Decision Tree improved most significantly (+8.2% accuracy, +6.3% ROC AUC) through tuning.

## SHAP — Explainability

ProjectEC5 added SHAP (SHapley Additive exPlanations) to explain model predictions:

| Feature | SHAP finding |
|---------|-------------|
| **thal** | Most influential — low values strongly predict disease |
| **ca** | Fewer major vessels = higher risk |
| **cp** | Chest pain type = strong disease indicator |
| **thalach** | Low max heart rate = higher predicted risk |
| **exang** | Exercise-induced angina = pushes toward disease |

See Section 10 of the notebook for beeswarm and waterfall plots.

## Testing

The project includes **66 tests** with **84% code coverage**.

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

### Coverage by module

| Module | Coverage |
|--------|----------|
| `src/terminal_app.py` | 100% |
| `src/model_training.py` | 98% |
| `src/data_preparation.py` | 99% |
| `src/data_processing.py` | 89% |
| `src/main.py` | 75% |
| `src/utils.py` | 0% |

## CI/CD (GitHub Actions)

ProjectEC5.1 uses two separate workflows:

### `tests.yml` — every push
- Python 3.11, Node.js 24
- Installs dependencies + prepares dataset
- Runs full 66-test suite
- Enforces minimum 84% coverage
- Uploads HTML coverage report

### `pr_checks.yml` — every pull request
Everything in `tests.yml` plus:
- **flake8** — PEP8 code style (`--max-line-length=100`)
- **bandit** — security scan (medium + high severity)
- **mypy** — type checking (`--ignore-missing-imports`)

```bash
# Run all checks locally
pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=84
flake8 src/ --max-line-length=100
bandit -r src/ -ll
mypy src/ --ignore-missing-imports
```

## Version History

| Tag | Description |
|-----|-------------|
| v5.1 | EC5.1 — section numbering fixed, pr_checks.yml, Node.js 24, 66 tests |
| v5.0 | EC5 — XGBoost, SHAP explainability |
| v4.0 | EC4 — hyperparameter tuning, combined dataset investigation |

## Ethical Considerations

- UCI Heart Disease dataset has known demographic bias — majority are middle-aged men
- SHAP explainability makes predictions auditable for clinical review
- `ca` and `thal` imputation decisions are documented transparently
- **Educational purposes only** — not a medical diagnostic tool

## Notes
- `notebooks/analysis.ipynb` is the single entry point — run all cells top-to-bottom
- Sections 1–15 run sequentially with no numbering gaps
- Node.js 24 used in CI/CD ahead of GitHub's Node.js 20 deprecation (16 June 2026)
- This application is for **educational purposes only** and is not a medical diagnostic tool
