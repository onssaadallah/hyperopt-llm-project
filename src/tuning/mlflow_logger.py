import mlflow
import os
import yaml

class MLflowLogger:
    """MLflow logger for experiments."""
    
    def __init__(self, config_path="src/tuning/mlflow_config.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.tracking_uri = config["tracking_uri"]
        mlflow.set_tracking_uri(self.tracking_uri)

    def set_experiment(self, name):
        mlflow.set_experiment(name)

    def log_params(self, params):
        mlflow.log_params(params)

    def log_metrics(self, metrics):
        mlflow.log_metrics(metrics)
