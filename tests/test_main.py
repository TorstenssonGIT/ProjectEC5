"""Tests for main module."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch
import pytest

from src.main import configure_arg_parser, train_and_save, train_and_save_full, run_app


class TestArgumentParser:
    """Test cases for argument parser configuration."""

    def test_parser_creation(self) -> None:
        """Test parser is created successfully."""
        parser = configure_arg_parser()
        assert parser is not None

    def test_parser_train_flag(self) -> None:
        """Test --train argument."""
        parser = configure_arg_parser()
        args = parser.parse_args(['--train'])
        
        assert args.train is True
        assert args.app is False

    def test_parser_app_flag(self) -> None:
        """Test --app argument."""
        parser = configure_arg_parser()
        args = parser.parse_args(['--app'])
        
        assert args.app is True
        assert args.train is False

    def test_parser_both_flags(self) -> None:
        """Test both --train and --app arguments."""
        parser = configure_arg_parser()
        args = parser.parse_args(['--train', '--app'])
        
        assert args.train is True
        assert args.app is True

    def test_parser_custom_paths(self) -> None:
        """Test custom data and model paths."""
        parser = configure_arg_parser()
        args = parser.parse_args([
            '--data-path', 'custom_data.csv',
            '--model-path', 'custom_model.joblib'
        ])
        
        assert args.data_path == 'custom_data.csv'
        assert args.model_path == 'custom_model.joblib'

    def test_parser_default_paths(self) -> None:
        """Test default paths."""
        parser = configure_arg_parser()
        args = parser.parse_args([])
        
        assert args.data_path == 'data/heart.csv'
        assert args.model_path == 'models/heart_model.joblib'
        assert args.train_full is False

    def test_parser_all_arguments(self) -> None:
        """Test all arguments together."""
        parser = configure_arg_parser()
        args = parser.parse_args([
            '--train',
            '--app',
            '--train-full',
            '--data-path', 'test_data.csv',
            '--model-path', 'test_model.joblib'
        ])
        
        assert args.train is True
        assert args.app is True
        assert args.train_full is True
        assert args.data_path == 'test_data.csv'
        assert args.model_path == 'test_model.joblib'

    @patch('src.main.ModelTrainer')
    @patch('src.main.DataProcessor')
    @patch('builtins.print')
    def test_train_full_and_save(self, mock_print, mock_data_processor_class, mock_trainer_class, sample_data_csv_path: str, temp_model_output: str) -> None:
        """Test train_full_and_save function."""
        mock_processor = MagicMock()
        mock_data_processor_class.return_value = mock_processor
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        mock_processor.get_features_and_target.return_value = (MagicMock(), MagicMock())
        mock_trainer.train_models.return_value = {'Random Forest': MagicMock()}
        mock_trainer.save_best_model.return_value = 'Random Forest'

        train_and_save_full(sample_data_csv_path, temp_model_output)

        mock_processor.load_data.assert_called_once()
        mock_processor.clean_data.assert_called_once()
        mock_processor.get_features_and_target.assert_called_once()
        mock_trainer.train_models.assert_called_once()
        mock_trainer.save_best_model.assert_called_once_with(temp_model_output)


class TestRunApp:
    """Test cases for run_app function."""

    def test_run_app_model_not_found(self) -> None:
        """Test run_app with non-existent model file."""
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            run_app("non_existent_model.joblib")

    @patch('src.main.HeartApp')
    def test_run_app_valid_model(self, mock_heart_app, model_file_path: str) -> None:
        """Test run_app with valid model file."""
        mock_instance = MagicMock()
        mock_heart_app.return_value = mock_instance
        
        run_app(model_file_path)
        
        # Check HeartApp was instantiated with correct path
        mock_heart_app.assert_called_once_with(model_file_path)
        # Check run method was called
        mock_instance.run.assert_called_once()


class TestTrainAndSave:
    """Test cases for train_and_save function."""

    @patch('src.main.ModelTrainer')
    @patch('src.main.DataProcessor')
    @patch('builtins.print')
    def test_train_and_save(self, mock_print, mock_data_processor_class, mock_trainer_class, sample_data_csv_path: str, temp_model_output: str) -> None:
        """Test train_and_save function."""
        # Setup mocks
        mock_processor = MagicMock()
        mock_data_processor_class.return_value = mock_processor
        
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        
        # Mock trainer methods
        mock_trainer.compare.return_value = MagicMock()
        mock_trainer.save_best_model.return_value = "Random Forest"
        
        # Mock split data return
        import pandas as pd
        import numpy as np
        X_train = pd.DataFrame(np.random.randn(80, 13))
        X_test = pd.DataFrame(np.random.randn(20, 13))
        y_train = pd.Series(np.random.randint(0, 2, 80))
        y_test = pd.Series(np.random.randint(0, 2, 20))
        
        mock_processor.split_data.return_value = (X_train, X_test, y_train, y_test)
        
        # Call the function
        train_and_save(sample_data_csv_path, temp_model_output)
        
        # Verify calls
        mock_processor.load_data.assert_called_once()
        mock_processor.clean_data.assert_called_once()
        mock_processor.split_data.assert_called_once()
        mock_trainer.train_models.assert_called_once()
        mock_trainer.evaluate.assert_called_once()
        mock_trainer.compare.assert_called_once()
        mock_trainer.save_best_model.assert_called_once_with(temp_model_output)

    @patch('src.main.ModelTrainer')
    @patch('src.main.DataProcessor')
    @patch('builtins.print')
    def test_train_and_save_prints_comparison(self, mock_print, mock_data_processor_class, mock_trainer_class, sample_data_csv_path: str, temp_model_output: str) -> None:
        """Test that train_and_save prints model comparison."""
        # Setup mocks
        mock_processor = MagicMock()
        mock_data_processor_class.return_value = mock_processor
        
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        
        # Create mock comparison dataframe
        import pandas as pd
        comparison_df = pd.DataFrame({
            'model': ['Random Forest', 'Logistic Regression'],
            'accuracy': [0.95, 0.88]
        })
        
        mock_trainer.compare.return_value = comparison_df
        mock_trainer.save_best_model.return_value = "Random Forest"
        
        # Mock split data
        import numpy as np
        X_train = pd.DataFrame(np.random.randn(80, 13))
        X_test = pd.DataFrame(np.random.randn(20, 13))
        y_train = pd.Series(np.random.randint(0, 2, 80))
        y_test = pd.Series(np.random.randint(0, 2, 20))
        
        mock_processor.split_data.return_value = (X_train, X_test, y_train, y_test)
        
        # Call the function
        train_and_save(sample_data_csv_path, temp_model_output)
        
        # Check that print was called with comparison message
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any('Model comparison' in str(call) for call in print_calls)
        assert any('Saved best model' in str(call) for call in print_calls)


class TestIntegration:
    """Integration tests for main module."""

    def test_parser_help(self) -> None:
        """Test parser help generation."""
        parser = configure_arg_parser()
        
        # Should not raise exception
        help_text = parser.format_help()
        assert 'Heart Disease' in help_text
        assert '--train' in help_text
        assert '--app' in help_text
        assert '--model-path' in help_text
        assert '--data-path' in help_text

    def test_invalid_arguments(self) -> None:
        """Test parser with invalid arguments."""
        parser = configure_arg_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(['--invalid-argument'])
