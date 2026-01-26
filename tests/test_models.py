import torch
import pytest
from src.models.model import BiLSTMForecast

def test_bilstm_forecast_shape():
    batch_size = 32
    input_size = 10
    hidden_size = 64
    num_layers = 2
    dropout = 0.1
    horizon = 5
    seq_len = 24
    
    model = BiLSTMForecast(input_size, hidden_size, num_layers, dropout, horizon)
    
    # Create dummy input [batch, seq_len, input_size]
    x = torch.randn(batch_size, seq_len, input_size)
    
    output = model(x)
    
    # Expected output shape: [batch, horizon]
    assert output.shape == (batch_size, horizon)

def test_bilstm_forecast_single_layer():
    # Test with 1 layer (dropout should be 0 handled by model logic)
    model = BiLSTMForecast(
        input_size=10, 
        hidden_size=32, 
        num_layers=1, 
        dropout=0.5, # Should be ignored/handled
        horizon=1
    )
    x = torch.randn(16, 12, 10)
    output = model(x)
    assert output.shape == (16, 1)
