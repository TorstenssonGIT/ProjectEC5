# Heart Disease Prediction Project

This project demonstrates a complete AI workflow for the UCI Heart Disease dataset. It includes data processing, exploratory data analysis, model training, evaluation, and a terminal-based prediction application.

## Structure
- `data/heart.csv` - dataset for modelling
- `src/data_processing.py` - data loading, cleaning and splitting
- `src/model_training.py` - model pipelines and evaluation
- `src/terminal_app.py` - terminal application for interactive predictions
- `src/main.py` - training and run entry point
- `notebooks/analysis.ipynb` - documented notebook with EDA, training and results
- `requirements.txt` - Python dependencies
- `.gitignore` - ignored files and folders

## Setup
1. Create virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```

## Run training and save model
```powershell
python src/main.py --train
```

## Train final model on the full dataset
```powershell
python src/main.py --train-full --model-path models/heart_model_full.joblib
```

## Run interactive application
```powershell
python src/main.py --app
```

## Run Streamlit web application
```powershell
python src/main.py --streamlit
```

## Testing

The project includes a comprehensive test suite with **44 tests** and **92% code coverage**:

### Run all tests
```powershell
pytest tests/ -v
```

### Generate coverage report
```powershell
pytest tests/ --cov=src --cov-report=html
```

### Test structure
- `tests/test_data_processing.py` - Tests for data loading, cleaning, and splitting
- `tests/test_model_training.py` - Tests for model training and evaluation
- `tests/test_app.py` - Tests for the prediction application
- `tests/test_main.py` - Tests for CLI and workflow functions
- `tests/conftest.py` - Shared test fixtures

See [TEST_DOCUMENTATION.md](TEST_DOCUMENTATION.md) for detailed test documentation.

## Notes
- The notebook documents all major steps, results and the ethical reflection.
- The terminal app (`src/terminal_app.py`) uses the trained Random Forest model for predictions.
