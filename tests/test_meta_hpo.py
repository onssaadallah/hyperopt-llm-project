import pytest
from unittest.mock import MagicMock, patch, mock_open
import pandas as pd
import numpy as np
import json
import sys

# Mock langchain modules if not available, to allow import of the source code
# We need to mock them BEFORE importing the source modules
sys.modules["langchain_community.llms"] = MagicMock()
sys.modules["langchain.prompts"] = MagicMock()
sys.modules["langchain.chains"] = MagicMock()

# Import the modules under test
# Note: we might need to mock sys.path inside the test if dependencies are messy, 
# but ignoring that for now as we run from root.
from src.meta_HPO.LLM_Load import load_llm, _ollama_installed, _ollama_model_exists
from src.meta_HPO.meta_knowledge_building import MetaKnowledgeBuilder
from src.tuning.meta_llm_tuning import LLMTuning

# --- LLM_Load Tests ---

@patch("src.meta_HPO.LLM_Load.shutil.which")
def test_ollama_installed(mock_which):
    mock_which.return_value = "/usr/bin/ollama"
    assert _ollama_installed() is True

    mock_which.return_value = None
    assert _ollama_installed() is False

@patch("src.meta_HPO.LLM_Load.subprocess.check_output")
def test_ollama_model_exists(mock_check_output):
    mock_check_output.return_value = "qwen2:3b\nllama2:latest"
    assert _ollama_model_exists("qwen2:3b") is True
    assert _ollama_model_exists("mistral") is False

@patch("src.meta_HPO.LLM_Load._ollama_installed", return_value=True)
@patch("src.meta_HPO.LLM_Load._ollama_model_exists", return_value=True)
def test_load_llm_ollama(mock_exists, mock_installed):
    # Since we mocked sys.modules["langchain_community.llms"] at the top,
    # importing it here gives us the same MagicMock object.
    from langchain_community.llms import Ollama
    
    llm = load_llm(backend="ollama", model_name="qwen2:3b")
    
    # Check if Ollama class was initialized
    Ollama.assert_called()
    assert llm == Ollama.return_value

# --- MetaKnowledgeBuilder Tests ---

@patch("src.meta_HPO.meta_knowledge_building.mlflow")
@patch("src.meta_HPO.meta_knowledge_building.Path")
def test_extract_univariate_meta_features(mock_path, mock_mlflow):
    # Mock Path to avoid creating directories
    mock_path.return_value.mkdir.return_value = None
    
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    builder = MetaKnowledgeBuilder(pd.DataFrame(), "test_exp", [])
    
    # We might need to handle imports of statsmodels if they fail
    # Assuming they are installed in the environment
    try:
        features = builder.extract_univariate_meta_features(s, freq=2, acf_lags=(1, 2))
        assert features["count"] == 10
        assert features["mean"] == 5.5
    except ImportError:
        pytest.skip("Statsmodels or Scipy not installed")

@patch("src.meta_HPO.meta_knowledge_building.mlflow")
@patch("src.meta_HPO.meta_knowledge_building.Path")
def test_aggregate_and_regime(mock_path, mock_mlflow):
    mock_path.return_value.mkdir.return_value = None
    
    meta_features_data = {
        "col1": {
            "acf_lag_1": 0.95, "adf_pvalue": 0.01, "trend_strength_decomp": 0.8,
            "seasonal_strength_decomp": 0.01, "coef_of_variation": 0.1, 
            "skewness": 0.5, "outlier_ratio": 0.0
        },
        "col2": {
            "acf_lag_1": 0.92, "adf_pvalue": 0.03, "trend_strength_decomp": 0.7,
            "seasonal_strength_decomp": 0.02, "coef_of_variation": 0.2, 
            "skewness": 0.4, "outlier_ratio": 0.01
        }
    }
    builder = MetaKnowledgeBuilder(pd.DataFrame(), "test_exp", [])
    agg = builder.aggregate_dataset_meta(meta_features_data)
    regime = builder.infer_dataset_regime(agg)
    
    assert regime["temporal_dependence"] == "strong" 
    assert regime["stationarity"] == "mostly stationary"

# --- LLMTuning Tests ---

@patch("src.tuning.meta_llm_tuning.load_llm")
@patch("src.tuning.meta_llm_tuning.ForecastingPipeline")
@patch("src.tuning.meta_llm_tuning.LLMChain")
def test_llmtuning_initialization(mock_llmchain, mock_pipeline, mock_load_llm):
    tuner = LLMTuning(
        model_name="Bi-LSTM", 
        experiment_name="test_exp", 
        clean_train=pd.DataFrame(), 
        clean_test=pd.DataFrame(),
        target_cols=["t1"], 
        mlflow_uri="file:///tmp/mlruns", # Added missing arg
        llm_name="dummy", 
        model_description="desc"
    )
    
    mock_load_llm.assert_called_once()
    mock_pipeline.assert_called_once()
    mock_llmchain.assert_called_once()
    assert tuner.experiment_name == "test_exp"

@patch("src.tuning.meta_llm_tuning.load_llm")
@patch("src.tuning.meta_llm_tuning.ForecastingPipeline")
@patch("src.tuning.meta_llm_tuning.LLMChain")
def test_extract_params_and_reasoning(mock_llmchain, mock_pipeline, mock_load_llm):
    tuner = LLMTuning(
        model_name="Bi-LSTM", 
        experiment_name="test_exp", 
        clean_train=pd.DataFrame(), 
        clean_test=pd.DataFrame(),
        target_cols=["t1"], 
        mlflow_uri="file:///tmp/mlruns", # Added missing arg
        llm_name="dummy", 
        model_description="desc"
    )
    
    valid_json_text = """
    Here is the suggestion:
    ```json
    {
        "suggested_params": {"lag": 24},
        "meta_feature_reasoning": "Reasoning here",
        "non_repetition_check": "ok",
        "expected_improvement": "yes"
    }
    ```
    """
    result = tuner.extract_params_and_reasoning(valid_json_text)
    assert result["suggested_params"]["lag"] == 24
    assert result["meta_feature_reasoning"] == "Reasoning here"
