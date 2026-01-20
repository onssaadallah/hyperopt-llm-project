import sys
import os
import time
import json
import random
import numpy as np
import pandas as pd
import torch
import mlflow
from sklearn.preprocessing import MinMaxScaler
sys.path.insert(0, r"C:/Users/user.IBRAHIM-IK-SZHE/Meta_LLM_HPO/hyperopt-llm-project/src")
from data.data_preprocessing import DataPreprocessing
from models.model import BiLSTMForecast
from models.training import Trainer
import os
import time
import json
import random
import numpy as np
import pandas as pd
import torch
import mlflow
from sklearn.preprocessing import MinMaxScaler
from data.data_preprocessing import DataPreprocessing
from models.training import Trainer
from models.model import BiLSTMForecast


# -------- MLflow path --------
MLFLOW_PATH = "file:///C:/Users/user.IBRAHIM-IK-SZHE/Meta_LLM_HPO/hyperopt-llm-project/mlruns"

# ---- Results folder path ----
SAVE_DIR = "C:/Users/user.IBRAHIM-IK-SZHE/Meta_LLM_HPO/hyperopt-llm-project/results/data_results"

class ForecastingPipeline:
    def __init__(
        self,
        clean_train: pd.DataFrame,
        clean_test: pd.DataFrame,
        target_cols: list,
        experiment_name: str,
        mlflow_uri=MLFLOW_PATH,
        seed: int = 42,
        device: str = None,
        save_dir = SAVE_DIR,
    ):
        self.clean_train = clean_train
        self.clean_test = clean_test
        self.target_cols = target_cols
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = save_dir
        self._set_seed()
        self._setup_mlflow(mlflow_uri, experiment_name)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _set_seed(self):
      random.seed(self.seed)
      np.random.seed(self.seed)
      torch.manual_seed(self.seed)
      torch.cuda.manual_seed_all(self.seed)

    
   

    def _setup_mlflow(self, uri, experiment_name):
        if uri:
            mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def _prepare_data(self, lag, batch_size):
        train_scaler = MinMaxScaler()
        test_scaler = MinMaxScaler()

        train_proc = DataPreprocessing(self.clean_train, train_scaler)
        test_proc = DataPreprocessing(self.clean_test, test_scaler)

        train_norm = train_proc.normalize()
        test_norm = test_proc.normalize()

        X_train, y_train = train_proc.create_windows(train_norm, self.target_cols, lag)
        X_test, y_test = test_proc.create_windows(test_norm, self.target_cols, lag)

        train_loader = train_proc.to_dataloader(X_train, y_train, batch_size)
        test_loader = test_proc.to_dataloader(X_test, y_test, batch_size)

        return X_train, train_loader, test_proc, test_loader

    # ------------------------------------------------------------------
    # Model factory
    # ------------------------------------------------------------------
    def _build_model(self, model_name, X_train, params):
        horizon = len(self.target_cols)
        if model_name == "Bi-LSTM":
            return BiLSTMForecast(
                input_size=X_train.shape[2],
                hidden_size=params["hidden_size"],
                num_layers=params["num_layers"],
                dropout=params["dropout"],
                horizon=horizon,
            )

        raise ValueError(f"Unknown model: {model_name}")

    # ------------------------------------------------------------------
    # Run experiment
    # ------------------------------------------------------------------
    def run(self, model_name: str, params: dict, run_suffix: str = None):
        run_name = f"{model_name}_{run_suffix}" if run_suffix else model_name

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("seed", self.seed)

            # -----------------------------
            # Data
            # -----------------------------
            X_train, train_loader, test_proc, test_loader = self._prepare_data(
                lag=params["lag"],
                batch_size=params["batch_size"],
            )
           

            mlflow.log_param("input_dim", X_train.shape[2])
            mlflow.log_param("horizon", len(self.target_cols))

            # -----------------------------
            # Model & trainer
            # -----------------------------
            model = self._build_model(model_name, X_train, params).to(self.device)

            trainer = Trainer(
                model=model,
                data_prep=test_proc,
                target_col=self.target_cols,
                optimizer_name=params["optimizer"],
                lr=params["lr"],
                device=self.device,
            )

            # -----------------------------
            # Train
            # -----------------------------
            start = time.time()
            loss_hist, rmse_hist = trainer.train(
                train_loader, epochs=params["epochs"]
            )
            train_time = time.time() - start

            # -----------------------------
            # Evaluate
            # -----------------------------
            rmse, smape, preds, true = trainer.evaluate(test_loader)

            # -----------------------------
            # Log metrics
            # -----------------------------
            mlflow.log_metric("rmse_test", rmse)
            mlflow.log_metric("smape_test", smape)
            mlflow.log_metric("train_time", train_time)

            for i, v in enumerate(loss_hist):
                mlflow.log_metric("train_loss", v, step=i)

            for i, v in enumerate(rmse_hist):
                mlflow.log_metric("val_rmse", v, step=i)

            self._save_results(model_name, preds, true, rmse, smape, train_time,loss_hist,rmse_hist)
            mlflow.log_artifacts(self.save_dir, artifact_path="results")
            return {
                "rmse": rmse,
                "smape": smape,
                "train_time": train_time,
                "loss_history": loss_hist,
                "rmse_history": rmse_hist,
                "preds": preds,
                "true": true,
                "run_name": run_name,
            }

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    def _save_results(self,model_name,preds,true,rmse,smape,train_time,loss_history,rmse_history):
        os.makedirs(self.save_dir, exist_ok=True)

        # Ensure 2D arrays
        true = np.atleast_2d(true)
        preds = np.atleast_2d(preds)

        if true.shape[0] != preds.shape[0]:
            raise ValueError("true and preds must have the same number of samples")

        # Predictions dataframe
        df = pd.DataFrame(
            np.hstack([true, preds]),
            columns=(
                [f"true_{i}" for i in range(true.shape[1])]
                + [f"pred_{i}" for i in range(preds.shape[1])]
            )
            
        )
        df.index = self.clean_test.index[-len(df):]  # last N rows
        df.index.name = "datetime"
        df.to_csv(
            os.path.join(self.save_dir, f"{model_name.lower()}_predictions.csv")
        )
 
        # Training history
        history_df = pd.DataFrame({
            "loss": [float(l) for l in loss_history],
            "rmse": [float(r) for r in rmse_history],
        })

        history_df.to_csv(
            os.path.join(self.save_dir, f"{model_name.lower()}_history.csv"),
            index=False,
        )

        # Metrics
        metrics = {
            "rmse": float(rmse),
            "smape": float(smape),
            "train_time": float(train_time),
        }

        with open(
            os.path.join(self.save_dir, f"{model_name.lower()}_metrics.json"),
            "w",
        ) as f:
            json.dump(metrics, f, indent=4)

