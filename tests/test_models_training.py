import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock
from src.models.training import Trainer

class MockModel(nn.Module):
    def __init__(self, out_features=1):
        super().__init__()
        self.linear = nn.Linear(10, out_features)
        
    def forward(self, x):
        return self.linear(x)

def test_trainer_train_step():
    # Use 2 targets to avoid singleton dimension squeezing issues/warnings
    model = MockModel(out_features=2)
    
    data_prep = MagicMock()
    trainer = Trainer(
        model=model,
        data_prep=data_prep,
        target_col=["t1", "t2"],
        optimizer_name="adam",
        lr=0.01,
        device="cpu"
    )
    
    # Loader yields (xb, yb)
    # yb should be [Batch, 2] so squeeze(1) keeps it [Batch, 2]
    X = torch.randn(4, 10)
    y = torch.randn(4, 2)
    loader = [(X, y)]
    
    loss_hist, rmse_hist = trainer.train(loader, epochs=1)
    
    assert len(loss_hist) == 1
    assert len(rmse_hist) == 1
    assert loss_hist[0] > 0

def test_trainer_evaluate():
    model = MockModel(out_features=2)
    data_prep = MagicMock()
    # mock inverse_target to just return input
    data_prep.inverse_target.side_effect = lambda x, y: x
    
    trainer = Trainer(
        model=model,
        data_prep=data_prep,
        target_col=["t1", "t2"],
        device="cpu"
    )
    
    X = torch.randn(4, 10)
    y = torch.randn(4, 2)
    loader = [(X, y)]
    
    rmse, smape, preds, true = trainer.evaluate(loader)
    
    # Check if numpy scalar or float
    assert isinstance(rmse, (float, np.floating))
    assert isinstance(smape, (float, np.floating))
    
    # Check shapes
    # preds and true are concatenated results from batches
    # Here 1 batch of 4 samples -> 4 samples
    assert len(preds) == 4
    assert len(true) == 4
    
    # Double check values logic roughly
    assert rmse >= 0
    assert smape >= 0
