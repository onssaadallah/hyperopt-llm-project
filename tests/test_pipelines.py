import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.pipelines.forecasting_pipeline import ForecastingPipeline

@pytest.fixture
def mock_data():
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    df = pd.DataFrame({
        "target1": np.random.rand(100),
        "target2": np.random.rand(100),
        "feature1": np.random.rand(100)
    }, index=dates)
    return df

@patch("src.pipelines.forecasting_pipeline.mlflow")
@patch("src.pipelines.forecasting_pipeline.Trainer")
@patch("src.pipelines.forecasting_pipeline.DataPreprocessing")
@patch("src.pipelines.forecasting_pipeline.BiLSTMForecast")
def test_pipeline_run(mock_bilstm, mock_dataprep, mock_trainer, mock_mlflow, mock_data):
    # Setup mocks
    mock_dataprep_instance = mock_dataprep.return_value
    mock_dataprep_instance.normalize.return_value = (None, None) # dummy return
    # Mock create_windows return values (X, y)
    mock_dataprep_instance.create_windows.return_value = (np.zeros((10, 5, 2)), np.zeros((10, 2)))
    # Mock to_dataloader
    mock_dataprep_instance.to_dataloader.return_value = MagicMock()
    
    mock_trainer_instance = mock_trainer.return_value
    # mock train to return loss histories
    mock_trainer_instance.train.return_value = ([0.1], [0.1]) 
    # mock evaluate to return rmse, smape, preds, true
    mock_trainer_instance.evaluate.return_value = (0.5, 0.1, np.zeros((10, 2)), np.zeros((10, 2)))

    pipeline = ForecastingPipeline(
        clean_train=mock_data,
        clean_test=mock_data,
        target_cols=["target1", "target2"],
        experiment_name="test_exp",
        mlflow_uri=None, # Mocked anyway
        save_dir="." # Dont want deep usage
    )
    
    # Mock the internal save_results to avoid FS writes or just let it write to tmp if we used tmp_path.
    # For now we rely on the fact that we mocked mlflow so it won't actually log artifacts to real server
    # But save_results writes to disk. Let's patch save_results too to be safe/clean.
    
    with patch.object(pipeline, "_save_results") as mock_save:
        params = {
            "lag": 5, 
            "hidden_size": 32, 
            "num_layers": 1, 
            "dropout": 0.0, 
            "batch_size": 32, 
            "optimizer": "adam",
            "lr": 0.001,
            "epochs": 1
        }
        
        results = pipeline.run("Bi-LSTM", params)
        
        assert results["rmse"] == 0.5
        assert results["smape"] == 0.1
        mock_trainer.assert_called()
        mock_save.assert_called()
