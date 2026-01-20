import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset,DataLoader,TensorDataset
import torch 

# ==============================
# Data Preprocessing
# ==============================
class DataPreprocessing:
    def __init__(self, df: pd.DataFrame,scaler:MinMaxScaler):
        self.df = df.copy()
        self.scaler = scaler
        
        self.feature_names = list(df.columns)

    def normalize(self) -> pd.DataFrame:
        arr = self.scaler.fit_transform(self.df.values)
        return pd.DataFrame(arr, index=self.df.index, columns=self.df.columns)

    def inverse_target(self,arr: np.ndarray,target_cols) -> np.ndarray:
      """
      Inverse transform multiple target columns.

      arr : shape (N, horizon) or (N, len(target_cols))
      target_cols : list of strings of target variable names
      """
      arr = np.array(arr)

      # Ensure 2D
      if arr.ndim == 1:
          arr = arr.reshape(-1, 1)

      # If arr has 1 column but multiple targets requested → ERROR CHECK
      if arr.shape[1] != len(target_cols):
          raise ValueError(
              f"Shape mismatch: arr has {arr.shape[1]} columns, "
              f"but target_cols has {len(target_cols)} targets. "
              f"Expected arr.shape = (N, {len(target_cols)})."
          )

      inv_arr = np.zeros((arr.shape[0], len(target_cols)))

      for i, col in enumerate(target_cols):
          target_idx = self.feature_names.index(col)

          # build dummy vector
          dummy = np.zeros((arr.shape[0], len(self.feature_names)))
          dummy[:, target_idx] = arr[:, i]

          # inverse scale
          inv = self.scaler.inverse_transform(dummy)
          inv_arr[:, i] = inv[:, target_idx]

      return inv_arr


    def create_windows(self, df, target_cols, lag, horizon=1):
        data = df.values
        target_idx = [df.columns.get_loc(col) for col in target_cols]

        X = []
        Y = []

        for i in range(len(df) - lag - horizon + 1):
            X.append(data[i:i+lag, :])                    # shape (lag, features)
            Y.append(data[i+lag:i+lag+horizon, target_idx])  # shape (horizon, targets)

        return np.array(X), np.array(Y)



    def to_dataloader(self, X, Y, batch_size=64):
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        data_loader = DataLoader(
            TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float()),
            batch_size=batch_size, shuffle=False
        )

        return data_loader