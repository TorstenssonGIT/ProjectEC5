# Heart Disease Prediction Application - Test Suite

## Overview

This document describes the comprehensive test suite for the Heart Disease Prediction application. The test suite provides **92% code coverage** with **44 passing tests** across all modules.

## Test Statistics

- **Total Tests**: 44
- **Passed**: 44 (100%)
- **Failed**: 0
- **Code Coverage**: 92%
- **Execution Time**: ~3.7 seconds

### Coverage by Module

| Module | Coverage | Lines | Missing |
|--------|----------|-------|---------|
| `src/app.py` | 100% | 45 | 0 |
| `src/data_processing.py` | 89% | 38 | 4 |
| `src/main.py` | 79% | 42 | 9 |
| `src/model_training.py` | 98% | 60 | 1 |

## Test Organization

Tests are organized into four main test modules, each corresponding to a source module:

### 1. Test Data Processing (`tests/test_data_processing.py`)

Tests for the `DataProcessor` class - responsible for data loading, cleaning, and splitting.

**Test Cases (10 tests)**:

- `test_initialization` - Verify DataProcessor initialization
- `test_load_data` - Load CSV data successfully
- `test_load_data_file_not_found` - Handle missing file errors
- `test_clean_data` - Verify column normalization and missing value handling
- `test_clean_data_with_missing_values` - Test handling of datasets with NaN values
- `test_get_features_and_target` - Extract feature matrix and target vector
- `test_split_data` - Split data into train/test sets with stratification
- `test_split_data_custom_test_size` - Test custom train/test ratio
- `test_summary` - Generate dataset statistics
- `test_correlation_matrix` - Compute feature correlation

**Coverage**: 89%

### 2. Test Model Training (`tests/test_model_training.py`)

Tests for the `ModelTrainer` class - handles model training, evaluation, and comparison.

**Test Cases (9 tests)**:

- `test_initialization` - Initialize trainer with random state
- `test_build_pipelines` - Create Logistic Regression and Random Forest pipelines
- `test_train_models` - Train both models on training data
- `test_evaluate_models` - Compute evaluation metrics (accuracy, F1, precision, recall, ROC-AUC)
- `test_compare_models` - Generate comparison dataframe sorted by accuracy
- `test_save_best_model` - Save highest-performing model to disk
- `test_save_best_model_no_training` - Handle error when no models are trained
- `test_model_result_dataclass` - Verify ModelResult dataclass structure
- `test_random_state_reproducibility` - Ensure reproducible results with same seed

**Coverage**: 98%

### 3. Test Application (`tests/test_app.py`)

Tests for the `HeartApp` class - terminal application for heart disease predictions.

**Test Cases (12 tests)**:

- `test_feature_info_creation` - Create FeatureInfo dataclass instances
- `test_initialization` - Initialize HeartApp with model
- `test_invalid_model_path` - Handle missing model file
- `test_predict_valid_input` - Make prediction on valid patient data
- `test_predict_multiple_samples` - Predict on multiple patients
- `test_features_list` - Verify all 13 features are present
- `test_prompt_for_inputs_valid` - Prompt user for patient values
- `test_prompt_for_inputs_invalid_then_valid` - Handle invalid input with retry
- `test_run_single_prediction` - Run app with single prediction
- `test_run_multiple_predictions` - Run app with multiple predictions
- `test_run_exit_on_first_no` - Exit app on user request
- `test_predict_edge_cases` - Test with minimum and maximum values

**Coverage**: 100%

### 4. Test Main Module (`tests/test_main.py`)

Tests for the command-line interface and workflow functions.

**Test Cases (13 tests)**:

#### TestArgumentParser (7 tests)
- `test_parser_creation` - Create argument parser
- `test_parser_train_flag` - Parse `--train` flag
- `test_parser_app_flag` - Parse `--app` flag
- `test_parser_both_flags` - Parse both flags simultaneously
- `test_parser_custom_paths` - Parse custom data/model paths
- `test_parser_default_paths` - Verify default path values
- `test_parser_all_arguments` - Parse all arguments together

#### TestRunApp (2 tests)
- `test_run_app_model_not_found` - Handle missing model file
- `test_run_app_valid_model` - Run app with valid model

#### TestTrainAndSave (2 tests)
- `test_train_and_save` - Complete training workflow
- `test_train_and_save_prints_comparison` - Verify comparison output

#### TestIntegration (2 tests)
- `test_parser_help` - Verify help text generation
- `test_invalid_arguments` - Handle invalid arguments

**Coverage**: 79%

## Test Fixtures

The `tests/conftest.py` file provides reusable fixtures for testing:

### Data Fixtures

- `sample_heart_data()` - Generate 100 sample heart disease records
- `sample_data_csv_path()` - Create temporary CSV with sample data
- `sample_data_with_missing_values()` - Dataset with NaN values

### Model Fixtures

- `trained_mock_model()` - Pre-trained mock model for testing
- `model_file_path()` - Temporary model file path
- `temp_model_output()` - Output path for saving models

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test File
```bash
pytest tests/test_data_processing.py -v
```

### Run Specific Test Class
```bash
pytest tests/test_app.py::TestHeartApp -v
```

### Run Specific Test
```bash
pytest tests/test_model_training.py::TestModelTrainer::test_evaluate_models -v
```

### Generate Coverage Report (Terminal)
```bash
pytest tests/ --cov=src --cov-report=term-missing
```

### Generate Coverage Report (HTML)
```bash
pytest tests/ --cov=src --cov-report=html
```
Then open `htmlcov/index.html` in a browser to view detailed coverage.

## Test Features

### Mocking
Tests use `unittest.mock` to isolate components:
- Mock user input with `patch('builtins.input')`
- Mock `HeartApp` instance in `run_app` tests
- Mock `DataProcessor` and `ModelTrainer` in workflow tests

### Error Handling
Tests verify error cases:
- File not found exceptions
- Invalid input validation
- Missing model errors

### Edge Cases
Tests include boundary conditions:
- Minimum and maximum feature values
- Empty datasets
- Missing/NaN values
- Invalid arguments

### Reproducibility
- All random seeds are fixed (42) for consistent results
- Tests verify random state produces reproducible model accuracy

## Continuous Integration

To integrate these tests into CI/CD:

```bash
# Run tests with JUnit XML output
pytest tests/ --junitxml=results.xml

# Run tests with coverage and report
pytest tests/ --cov=src --cov-report=xml --cov-report=term

# Run tests with HTML report and fail on low coverage
pytest tests/ --cov=src --cov-report=html --cov-fail-under=85
```

## Dependencies

Test-related packages in `requirements.txt`:
```
pytest>=7.4
pytest-cov>=4.1
```

## Coverage Details

### Uncovered Lines

**src/main.py** (9 missing lines):
- Lines 64-74: Main workflow when both `--train` and `--app` are used (integration path)
- Line 78: Entry point `if __name__ == "__main__"` guard

**src/data_processing.py** (4 missing lines):
- Lines 23, 39, 61, 75: Edge cases in auto-loading and data retrieval

**src/model_training.py** (1 missing line):
- Line 90: Fallback decision function (rarely called path)

## Best Practices Used

1. **Fixture Reuse** - Fixtures reduce code duplication
2. **Parametrization Ready** - Tests can be easily parametrized for multiple scenarios
3. **Clear Test Names** - Descriptive test names indicate what is being tested
4. **Isolation** - Each test is independent and can run in any order
5. **Mocking** - External dependencies are mocked to isolate units
6. **Assertions** - Clear, specific assertions that fail with informative messages

## Future Improvements

Potential areas for additional testing:
1. Integration tests with real models and data
2. Performance benchmarks for model training
3. Load testing for concurrent predictions
4. Test edge cases with very large datasets
5. Mutation testing for test quality assessment

## Conclusion

The test suite provides comprehensive coverage of the heart disease prediction application with 44 tests achieving 92% code coverage. All tests pass successfully in ~3.7 seconds, providing confidence in the application's correctness and reliability.
