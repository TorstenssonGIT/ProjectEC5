"""Tests for app module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from src.app import HeartApp, FeatureInfo


class TestFeatureInfo:
    """Test cases for FeatureInfo dataclass."""

    def test_feature_info_creation(self) -> None:
        """Test creating FeatureInfo instances."""
        feature = FeatureInfo(
            name="age",
            description="Age in years",
            data_type="int"
        )
        
        assert feature.name == "age"
        assert feature.description == "Age in years"
        assert feature.data_type == "int"


class TestHeartApp:
    """Test cases for HeartApp class."""

    def test_initialization(self, model_file_path: str) -> None:
        """Test HeartApp initialization."""
        app = HeartApp(model_file_path)
        
        assert app.model_path == model_file_path
        assert app.model is not None
        assert len(app.features) == 13
        
        # Check first feature
        assert app.features[0].name == "age"
        assert app.features[0].data_type == "int"

    def test_invalid_model_path(self) -> None:
        """Test initialization with invalid model path."""
        with pytest.raises(FileNotFoundError):
            HeartApp("non_existent_model.joblib")

    def test_predict_valid_input(self, model_file_path: str) -> None:
        """Test prediction with valid input."""
        app = HeartApp(model_file_path)
        
        # Create valid input values (13 features)
        values = [55, 1, 0, 120, 200, 0, 1, 140, 0, 1.0, 1, 0, 3]
        
        prediction, probability = app.predict(values)
        
        assert isinstance(prediction, int)
        assert prediction in [0, 1]
        assert isinstance(probability, float)
        assert 0.0 <= probability <= 1.0

    def test_predict_multiple_samples(self, model_file_path: str) -> None:
        """Test predictions with multiple different inputs."""
        app = HeartApp(model_file_path)
        
        samples = [
            [55, 1, 0, 120, 200, 0, 1, 140, 0, 1.0, 1, 0, 3],
            [45, 0, 1, 130, 250, 1, 0, 160, 1, 2.0, 2, 1, 6],
            [65, 1, 2, 150, 300, 0, 2, 100, 0, 0.5, 1, 2, 7],
        ]
        
        predictions = []
        probabilities = []
        
        for sample in samples:
            pred, prob = app.predict(sample)
            predictions.append(pred)
            probabilities.append(prob)
        
        assert len(predictions) == 3
        assert len(probabilities) == 3
        assert all(p in [0, 1] for p in predictions)
        assert all(0.0 <= p <= 1.0 for p in probabilities)

    def test_features_list(self, model_file_path: str) -> None:
        """Test that all expected features are present."""
        app = HeartApp(model_file_path)
        
        feature_names = [f.name for f in app.features]
        expected_features = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        
        assert feature_names == expected_features

    @patch('builtins.input')
    def test_prompt_for_inputs_valid(self, mock_input, model_file_path: str) -> None:
        """Test prompting for inputs with valid values."""
        app = HeartApp(model_file_path)
        
        # Mock input values for all 13 features
        input_values = [
            '55',   # age
            '1',    # sex
            '0',    # cp
            '120',  # trestbps
            '200',  # chol
            '0',    # fbs
            '1',    # restecg
            '140',  # thalach
            '0',    # exang
            '1.0',  # oldpeak
            '1',    # slope
            '0',    # ca
            '3'     # thal
        ]
        
        mock_input.side_effect = input_values
        
        values = app.prompt_for_inputs()
        
        assert len(values) == 13
        assert values[0] == 55  # age as int
        assert values[9] == 1.0  # oldpeak as float

    @patch('builtins.input')
    def test_prompt_for_inputs_invalid_then_valid(self, mock_input, model_file_path: str) -> None:
        """Test prompting with invalid input followed by valid input."""
        app = HeartApp(model_file_path)
        
        # First attempt: invalid input, then valid
        input_values = [
            'invalid',  # invalid for age
            '55',       # valid for age
        ] + ['0'] * 12  # valid for remaining features
        
        mock_input.side_effect = input_values
        
        values = app.prompt_for_inputs()
        
        assert len(values) == 13
        assert values[0] == 55

    @patch('builtins.input')
    @patch('builtins.print')
    def test_run_single_prediction(self, mock_print, mock_input, model_file_path: str) -> None:
        """Test running the app with a single prediction."""
        app = HeartApp(model_file_path)
        
        # Mock inputs: 13 feature values + "n" to exit
        input_values = ['55', '1', '0', '120', '200', '0', '1', '140', '0', '1.0', '1', '0', '3', 'n']
        mock_input.side_effect = input_values
        
        app.run()
        
        # Check that prediction was made
        assert mock_print.called
        # Check for prediction output
        print_calls = [str(call) for call in mock_print.call_args_list]
        prediction_made = any('Prediction:' in str(call) for call in print_calls)
        assert prediction_made

    @patch('builtins.input')
    @patch('builtins.print')
    def test_run_multiple_predictions(self, mock_print, mock_input, model_file_path: str) -> None:
        """Test running the app with multiple predictions."""
        app = HeartApp(model_file_path)
        
        # Mock inputs: two predictions, then exit
        input_values = [
            # First prediction
            '55', '1', '0', '120', '200', '0', '1', '140', '0', '1.0', '1', '0', '3',
            'y',  # Continue
            # Second prediction
            '65', '0', '1', '130', '250', '1', '0', '160', '1', '2.0', '2', '1', '6',
            'n'   # Exit
        ]
        mock_input.side_effect = input_values
        
        app.run()
        
        # Check that output was generated
        assert mock_print.called

    @patch('builtins.input')
    @patch('builtins.print')
    def test_run_exit_on_first_no(self, mock_print, mock_input, model_file_path: str) -> None:
        """Test exiting app when user says no to continue."""
        app = HeartApp(model_file_path)
        
        input_values = [
            # First prediction
            '55', '1', '0', '120', '200', '0', '1', '140', '0', '1.0', '1', '0', '3',
            'n'   # Exit
        ]
        mock_input.side_effect = input_values
        
        app.run()
        
        # Check for exit message
        print_calls = [str(call) for call in mock_print.call_args_list]
        exit_message_found = any('Exiting' in str(call) for call in print_calls)
        assert exit_message_found

    def test_predict_edge_cases(self, model_file_path: str) -> None:
        """Test prediction with edge case values."""
        app = HeartApp(model_file_path)
        
        # Edge case: minimum values
        min_values = [29, 0, 0, 70, 100, 0, 0, 60, 0, 0.0, 0, 0, 3]
        pred_min, prob_min = app.predict(min_values)
        assert pred_min in [0, 1]
        assert 0.0 <= prob_min <= 1.0
        
        # Edge case: maximum values
        max_values = [77, 1, 3, 200, 400, 1, 2, 200, 1, 6.0, 2, 4, 7]
        pred_max, prob_max = app.predict(max_values)
        assert pred_max in [0, 1]
        assert 0.0 <= prob_max <= 1.0
