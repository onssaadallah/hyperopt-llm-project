import pytest
from unittest.mock import MagicMock, patch
from src.tuning.bo_tuning import BayesianOptimization

@patch("src.tuning.bo_tuning.ForecastingPipeline")
def test_bo_tuning_run(mock_pipeline_cls):
    # Mock pipeline instance
    mock_pipeline = mock_pipeline_cls.return_value
    mock_pipeline.run.return_value = {"rmse": 0.1, "smape": 0.05}
    
    clean_train = MagicMock()
    clean_test = MagicMock()
    target_cols = ["t1"]
    
    bo = BayesianOptimization(clean_train, clean_test, target_cols, "Bi-LSTM", "test_exp")
    
    # We want to run just 1 trial to verify flow
    # Since optuna is real, it will call objective.
    # objective calls pipeline.run.
    
    time_taken = bo.run_bo(n_trials=1)
    
    assert time_taken >= 0
    # Check if pipeline was called
    assert mock_pipeline.run.call_count >= 1
