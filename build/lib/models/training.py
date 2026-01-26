
import pandas as pd 
import numpy as np
import torch 
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import time
import random
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        data_prep,
        target_col,
        optimizer_name: str = "adam",
        lr: float = 1e-3,
        device: str = None,
    ):
        """
        Generic trainer for forecasting models.

        Args:
            model: torch model
            data_prep: DataPreprocessing object (for inverse scaling)
            target_col: list of target column names
            optimizer_name: adam | adamw | sgd | rmsprop | nadam
            lr: learning rate
            device: cuda | cpu
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.data_prep = data_prep
        self.target_col = target_col

        self.criterion = nn.MSELoss()
        self.optimizer = self._build_optimizer(optimizer_name, lr)

    # ------------------------------------------------------------------
    # Optimizer factory
    # ------------------------------------------------------------------
    def _build_optimizer(self, name, lr):
        name = name.lower()
        optimizers = {
            "adam": optim.Adam,
            "adamw": optim.AdamW,
            #"sgd": lambda p,lr: optim.SGD(p, lr=lr, momentum=0.9),
            "rmsprop": optim.RMSprop,
            "adagrad": optim.Adagrad,
            "adadelta": optim.Adadelta,
            "adamax": optim.Adamax,
            "nadam": optim.NAdam
        
        }

        if name not in optimizers:
            raise ValueError(f"Unknown optimizer: {name}")

        return optimizers[name](self.model.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, train_loader, epochs: int):
        self.model.train()

        loss_history = []
        rmse_history = []

        for epoch in range(epochs):
            batch_losses = []
            preds_all, y_all = [], []

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device).squeeze(1)

                self.optimizer.zero_grad()
                preds = self.model(xb)

                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()

                batch_losses.append(loss.item())
                preds_all.append(preds.detach().cpu().numpy())
                y_all.append(yb.detach().cpu().numpy())

            # Epoch metrics
            preds_all = np.concatenate(preds_all)
            y_all = np.concatenate(y_all)

            epoch_loss = np.mean(batch_losses)
            epoch_rmse = np.sqrt(np.mean((preds_all - y_all) ** 2))

            loss_history.append(epoch_loss)
            rmse_history.append(epoch_rmse)

            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Loss: {epoch_loss:.6f} | RMSE: {epoch_rmse:.6f}"
            )

        return loss_history, rmse_history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self,test_loader):
        self.model.eval()

        preds_scaled = []
        y_scaled = []

        for xb, yb in test_loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device).squeeze(1)

            preds = self.model(xb)

            preds_scaled.append(preds.cpu().numpy())
            y_scaled.append(yb.cpu().numpy())

        preds_scaled = np.concatenate(preds_scaled)
        y_scaled = np.concatenate(y_scaled)

        # Inverse scaling
        preds_true = self.data_prep.inverse_target(
            preds_scaled, self.target_col
        )
        y_true = self.data_prep.inverse_target(
            y_scaled, self.target_col
        )

        rmse = np.sqrt(np.mean((preds_true - y_true) ** 2))
        smape = np.mean(
            2
            * np.abs(preds_true - y_true)
            / (np.abs(preds_true) + np.abs(y_true) + 1e-8)
        ) * 100

        print(f"Test RMSE: {rmse:.4f} | SMAPE: {smape:.2f}%")

        return rmse, smape, preds_true, y_true

    

   


   
   






