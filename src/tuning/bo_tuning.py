
import pandas as pd 
import numpy as np 
import optuna 
import mlflow 
import time 
import sys 
from pathlib import Path 
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.data_preprocessing import  DataPreprocessing
from models.model import BiLSTMForecast
from pipelines.forecasting_pipeline import ForecastingPipeline

class BayesianOptimization:
    def __init__(
        self,
        clean_train,
        clean_test,
        target_cols,
        mlflow_uri,
        model_name="Bi-LSTM",
        experiment_name="Bayesian_Optimization",):
        self.clean_train = clean_train
        self.clean_test = clean_test
        self.target_cols = target_cols
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.mlflow_uri = mlflow_uri

        self.pipeline = ForecastingPipeline(
            clean_train=clean_train,
            clean_test=clean_test,
            target_cols=target_cols,
            experiment_name=experiment_name,
            mlflow_uri=mlflow_uri,
        )

        self.study = optuna.create_study(
                direction="minimize",
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
            )

    def objective(self, trial):

        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.0, 0.5) if num_layers > 1 else 0.0

        params = {
            "lag": trial.suggest_int("lag", 12, 72),
            "horizon": 4,
            "hidden_size": trial.suggest_int("hidden_size", 16, 128),
            "num_layers": num_layers,
            "dropout": dropout,
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "epochs": trial.suggest_int("epochs", 5, 20),
            "optimizer": trial.suggest_categorical(
                "optimizer",
                ["adam", "adamw", "rmsprop", "adagrad", "adamax"]
            ),
        }

        try:
            results = self.pipeline.run(
                self.model_name,
                params,
                run_suffix=f"{self.experiment_name}_trial_{trial.number}"
            )

            rmse = float(results["rmse"])
            smape = float(results["smape"])

            if not np.isfinite(rmse):
                raise ValueError("RMSE is NaN or inf")

            trial.set_user_attr("rmse", rmse)
            trial.set_user_attr("smape", smape)

            trial.report(rmse, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()

            return rmse

        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return float("inf")

    def run_bo(self, n_trials):

        start_time = time.time()
        self.study.optimize(self.objective, n_trials=n_trials)
        end_time = time.time()

        bo_optimization_time = end_time - start_time
        print(f"Optimization took {bo_optimization_time:.2f} seconds")

        return bo_optimization_time
