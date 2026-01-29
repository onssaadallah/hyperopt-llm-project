import pytest
from unittest.mock import MagicMock, patch, mock_open
import yaml
from src.tuning.mlflow_logger import MLflowLogger

@pytest.fixture
def mock_config_file():
    config = {"tracking_uri": "file:///tmp/mlruns"}
    with patch("builtins.open", mock_open(read_data=yaml.dump(config))):
        yield

@patch("src.tuning.mlflow_logger.mlflow")
def test_logger_initialization(mock_mlflow, mock_config_file):
    logger = MLflowLogger(config_path="dummy_path.yaml")
    
    # Check if tracking URI was set
    mock_mlflow.set_tracking_uri.assert_called_with("file:///tmp/mlruns")

@patch("src.tuning.mlflow_logger.mlflow")
def test_logger_methods(mock_mlflow, mock_config_file):
    logger = MLflowLogger(config_path="dummy_path.yaml")
    
    logger.set_experiment("test_exp")
    mock_mlflow.set_experiment.assert_called_with("test_exp")
    
    params = {"param1": 1}
    logger.log_params(params)
    mock_mlflow.log_params.assert_called_with(params)
    
    metrics = {"metric1": 0.5}
    logger.log_metrics(metrics)
    mock_mlflow.log_metrics.assert_called_with(metrics)
