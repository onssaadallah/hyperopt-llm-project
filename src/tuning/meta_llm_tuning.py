import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.data_cleaning import CleanData
from data.data_preprocessing import DataPreprocessing
from models.model import BiLSTMForecast
from models.training import Trainer
from meta_HPO.LLM_Load import load_llm
from pipelines.forecasting_pipeline import ForecastingPipeline
from meta_HPO.llm_prompt import prompt_template
from langchain.chains import LLMChain



import re
import json


"""LLM-guided tuning using LangChain and MLflow."""

class LLMTuning:
    def __init__(self,model_name,experiment_name,clean_train,clean_test,target_cols,mlflow_uri,llm_name,model_description,temperature=0.15):
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.clean_train = clean_train
        self.clean_test = clean_test
        self.target_cols = target_cols
        self.mlflow_uri = mlflow_uri
        self.model_description = model_description
        self.llm_name = llm_name
        self.temperature = temperature

        # Load the LLM via Ollama
        self.llm_model = load_llm(
            backend="ollama",
            model_name=self.llm_name,
            temperature=self.temperature,
        )

        self.prompt_template = prompt_template

        # Initialize LangChain LLMChain
        self.suggest_chain = LLMChain(
            llm=self.llm_model,
            prompt=self.prompt_template,
        )

        

        self.pipeline = ForecastingPipeline(
            clean_train=self.clean_train,
            clean_test=self.clean_test,
            target_cols=self.target_cols,
            experiment_name=self.experiment_name,
            mlflow_uri=self.mlflow_uri
        )

    # ==================================================
    # 1. Extract meta-initial trials and meta-knowledge
    # ==================================================
    @staticmethod
    def compute_target_rmse(trials, epsilon: float = 0.0, default: float = 1.0) -> float:
        """Target RMSE is slightly lower than best historical RMSE."""
        rmses = [t["test_rmse"] for t in trials if "test_rmse" in t and t["test_rmse"] is not None]
        if not rmses:
            return default
        return float(min(rmses) - epsilon)
    def extract_metadata(self, meta_trials_data, meta_features_data, epsilon=0.08):

        with open(meta_trials_data, "r", encoding="utf-8") as f:
            best_trials = json.load(f)

        historical_trials = [t["params"] for t in best_trials]

        with open(meta_features_data, "r", encoding="utf-8") as f:
            meta_features = json.load(f)

        raw_features_summary = json.dumps(meta_features["raw_meta_features"], indent=2)
        summary_meta_features = json.dumps(meta_features["dataset_regime"], indent=2)

        # Shots of train data
        dataset_sample = json.dumps(
            self.clean_train.tail(30).to_dict(orient="records"),
            indent=2,
        )

        # Compute target RMSE
        target_rmse = self.compute_target_rmse(
            best_trials, epsilon=epsilon
        )

        # Best RMSE extraction (FIXED: was min(params) before)
        best_rmse = min(t["test_rmse"] for t in best_trials)

        return (
            historical_trials,
            dataset_sample,
            raw_features_summary,
            summary_meta_features,
            target_rmse,
            best_rmse,
        )

    # ==========================================================================
    # 2. Tools
    # ==========================================================================
    def extract_params_and_reasoning(self, text: str) -> dict:
        """
        Flexible extractor for suggested_params and reasoning from LLM output.
        Works whether JSON is inside ```json``` fences or plain text.
        """
        # ---------------------------
        # 1. Extract JSON-like substring
        # ---------------------------
        json_match = re.search(r"```json(.*?)```", text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON object found in text")
            json_str = json_match.group(0)

        # ---------------------------
        # 2. Load JSON safely
        # ---------------------------
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON detected: {e}")

        # ---------------------------
        # 3. Extract keys
        # ---------------------------
        if "suggested_params" not in data:
            raise KeyError("'suggested_params' not found in JSON")

        reasoning = data.get("meta_feature_reasoning")

        return {
            "suggested_params": data["suggested_params"],
            "meta_feature_reasoning": reasoning,
        }

    def record_trials(self,params,rmse,reasoning,llm_name,base_dir=None):
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent.parent.parent / "data" / "LLM_trials_memory"
        record = {
            "params": params,
            "rmse": rmse,
            "reasoning": reasoning,
        }

        # 🔒 Windows-safe filename
        safe_llm_name = llm_name.replace(":", "_").replace("/", "_")
        filename = f"trials_history_{safe_llm_name}.json"

        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        file_path = base_dir / filename

        # Load existing history
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        data.append(record)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"Trial recorded → {file_path}")




    def hpo_optimization(
        self,
        n_trials,
        historical_trials,
        dataset_sample,
        raw_features_summary,
        target_rmse,
        best_rmse,
    ):
        for i in range(n_trials):
            print(f"\n===== HPO Trial {i+1} =====")

            # Ask LLM for new configuration
            config_json = self.suggest_chain.run(
                model_description=self.model_description,
                num_features=4,
                dataset_meta_summary=dataset_sample,
                raw_features_summary=raw_features_summary,
                dataset_sample=dataset_sample,
                historical_trials=json.dumps(historical_trials, indent=2),
                current_best_rmse=best_rmse,
                target_rmse=target_rmse,
            )

            print("Suggested JSON:", config_json)
            parsing = self.extract_params_and_reasoning(config_json)
            print("Reasoning:", parsing["meta_feature_reasoning"])

            # Run model
            results = self.pipeline.run("Bi-LSTM",parsing["suggested_params"],run_suffix=f"{self.experiment_name}_trial_{i}")
            rmse = results["rmse"]

            #  Update best
            if rmse < best_rmse:
                best_rmse = rmse
                historical_trials.append(parsing["suggested_params"])
                self.record_trials(
                    parsing["suggested_params"],
                    best_rmse,
                    parsing["meta_feature_reasoning"],
                    self.llm_name,
                )
                print("New best RMSE!")

            #  Early stopping
            if best_rmse <= target_rmse:
                print("Target RMSE reached!")
                break



    def run_llm_autoopt(self,n_trials,meta_trials_data,meta_features_data):
        historical_trials,dataset_sample,raw_features_summary,summary_meta_features,target_rmse,best_rmse = self.extract_metadata(meta_trials_data, meta_features_data)
        #Run LLM AutoOpt
        self.hpo_optimization(n_trials,historical_trials,dataset_sample,raw_features_summary,target_rmse,best_rmse)
          