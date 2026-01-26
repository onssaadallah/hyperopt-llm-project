#!/usr/bin/env python
# coding: utf-8

"""
 Define the Candidate Deep forecasting model
"""

import torch 
import torch.nn as nn


# ==============================
#  Candidate Models
# ==============================
class BiLSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, horizon):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        # hidden_size*2 because bidirectional
        self.fc = nn.Linear(hidden_size * 2, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]            # (batch, hidden*2)
        out = self.fc(out)             # (batch, horizon)
        return out