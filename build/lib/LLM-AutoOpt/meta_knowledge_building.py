import mlflow
from pathlib import Path
import numpy as np
import pandas as pd
import json 

MLFLOW_PATH = "file:///C:/Users/user.IBRAHIM-IK-SZHE/Meta_LLM_HPO/hyperopt-llm-project/mlruns"
mlflow.set_tracking_uri(MLFLOW_PATH)

class MetaKnowledgeBuilder:
    def __init__(self,clean_train,experience_name):
        self.clean_train  = clean_train 
        self.experience_name = experience_name

    def extract_params(self,run_row):
        return {
        col.replace("params.", ""): run_row[col]
        for col in run_row.index
        if col.startswith("params.")
    }

    def create_meta_trials(self,top_k,initial_trials_path="C:/Users/user.IBRAHIM-IK-SZHE/Meta_LLM_HPO/hyperopt-llm-project/data/meta_data/run_data.json"):
        best_trials = []
        RMSE_METRIC = "metrics.rmse_test"
        experiment = mlflow.get_experiment_by_name(self.experience_name)
        if experiment is None:
            raise ValueError(f"Experiment '{self.experience_name}' not found")
        experiment_id = experiment.experiment_id
        runs_df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        order_by=[f"{RMSE_METRIC} ASC"],
        output_format="pandas")
        best_runs = runs_df.head(top_k)
        for i ,row in best_runs.iterrows():
            best_trials.append({
                "run_id": "trial_" + str(i+1),
                "test_rmse": row[RMSE_METRIC],
                "params": self.extract_params(row)
            })
        # save data as json 
        with open(initial_trials_path,"w") as f:
            json.dump(best_trials, f, indent=2)  


    def meta_features_builder(self) :
        pass     