from typing import Tuple, List, Dict
import os
import sys
import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import mlflow

# -------------------------------------------------
# Project root
# -------------------------------------------------

sys.path.insert(
    0, r"C:\Users\user.IBRAHIM-IK-SZHE\hyperopt-llm-project\src"
)



class ResultsPlotter:
    def __init__(
        self,
        test_df: pd.DataFrame,
        rmse_threshold: float | None = None,
        mlflow_uri: str | None = None,
    ):
        self.test_df = test_df
        self.rmse_threshold = rmse_threshold
        self.mlflow_uri = mlflow_uri
        mlflow.set_tracking_uri(self.mlflow_uri)

    # ===============================================================
    # List experiments
    # ===============================================================
    def get_experiments(self) -> List[str]:
        experiments = mlflow.search_experiments()
        for exp in experiments:
            print(f"[{exp.experiment_id}] '{exp.name}'")
        return [exp.name for exp in experiments]

    # ===============================================================
    # List runs
    # ===============================================================
    def list_experiment_runs(self, exp_name: str) -> pd.DataFrame:
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp is None:
            raise ValueError(f"Experiment '{exp_name}' not found")

        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        if runs.empty:
            print(f"⚠️ No runs found for experiment '{exp_name}'")
        return runs

    # ===============================================================
    # Load MLflow runs below RMSE
    # ===============================================================
    def load_mlflow_runs_below_rmse(self,runs: pd.DataFrame,
                                    results_subdir: str = "results",) -> Tuple[List[Dict], List[str], Dict, Dict]:

        experiment_results = []
        experiment_labels = []
        rmse_histories = {}
        final_metrics_all = {}

        for _, run in runs.iterrows():

            # -----------------------------
            # Correct metric key
            # -----------------------------
            test_rmse = run.get("metrics.rmse_test", None)
            if self.rmse_threshold is not None:
                if test_rmse is None or test_rmse > self.rmse_threshold:
                    continue

            artifact_uri = run.artifact_uri.replace("file:///", "")
            results_dir = os.path.join(artifact_uri, results_subdir)

            model_name = run.get("params.model_name", "").lower()

            preds_csv = os.path.join(results_dir, f"{model_name}_predictions.csv")
            hist_csv  = os.path.join(results_dir, f"{model_name}_history.csv")
            metr_json = os.path.join(results_dir, f"{model_name}_metrics.json")

            if not os.path.exists(preds_csv):
                continue

            # -----------------------------
            # Load predictions
            # -----------------------------
            df_preds = pd.read_csv(preds_csv, index_col=0)

            y_true_cols = [c for c in df_preds.columns if c.startswith("true_")]
            y_pred_cols = [c for c in df_preds.columns if c.startswith("pred_")]

            experiment_results.append({
                "true": df_preds[y_true_cols].values,
                "preds": df_preds[y_pred_cols].values,
            })

            run_name = run.get("tags.mlflow.runName", run.run_id)
            experiment_labels.append(run_name)

            # -----------------------------
            # Load RMSE history
            # -----------------------------
            if os.path.exists(hist_csv):
                df_hist = pd.read_csv(hist_csv)
                rmse_histories[run_name] = {
                    "train_loss": df_hist["loss"].tolist(),
                    "val_rmse": df_hist["rmse"].tolist(),
                }

            # -----------------------------
            # Load final metrics
            # -----------------------------
            if os.path.exists(metr_json):
                with open(metr_json, "r") as f:
                    final_metrics_all[run_name] = json.load(f)

        return experiment_results, experiment_labels, rmse_histories, final_metrics_all

    # ===============================================================
    # Plot forecasts
    # ===============================================================
    def plot_multiple_experiments_forecast(
        self,
        experiment_results: list,
        model_names: list,
        test_df: pd.DataFrame,
        save_dir: str,
    ):
        if not experiment_results:
            raise ValueError("❌ No experiment results to plot.")

        os.makedirs(save_dir, exist_ok=True)

        if not isinstance(test_df.index, pd.DatetimeIndex):
            test_df.index = pd.to_datetime(test_df.index, errors="coerce")

        n_targets = experiment_results[0]["preds"].shape[1]
        n_points = experiment_results[0]["true"].shape[0]
        idx = test_df.index[-n_points:]

        palette = [
            "#1f77b4", "#2ca02c", "#e377c2",
            "#8c564b", "#9467bd", "#17becf"
        ]

        for i in range(n_targets):
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=idx,
                y=experiment_results[0]["true"][:, i],
                name="True",
                mode="lines",
                line=dict(color="orange", width=3),
            ))

            for j, (exp, name) in enumerate(zip(experiment_results, model_names)):
                fig.add_trace(go.Scatter(
                    x=idx,
                    y=exp["preds"][:, i],
                    mode="lines",
                    name=name,
                    line=dict(
                        color=palette[j % len(palette)],
                        width=2,
                        dash="dashdot",
                    ),
                ))

            fig.update_layout(
                title=f"Forecast comparison – Target {i}",
                xaxis_title="Time",
                yaxis_title="Value",
                template="plotly_white",
                width=1200,
                height=600,
            )

            fig.show()
            fig.write_image(os.path.join(save_dir, f"target_{i}_comparison.png"))

    # ===============================================================
    # Plot RMSE history
    # ===============================================================
    def plot_multiple_experiments_rmse_history(
        self,
        rmse_histories: dict,
        save_dir: str,
        title: str = "Validation RMSE comparison",
    ):
        if not rmse_histories:
            print("⚠️ No RMSE histories to plot.")
            return

        os.makedirs(save_dir, exist_ok=True)
        fig = go.Figure()

        for exp_name, hist in rmse_histories.items():
            rmse = hist["val_rmse"]
            fig.add_trace(go.Scatter(
                x=list(range(1, len(rmse) + 1)),
                y=rmse,
                mode="lines+markers",
                name=exp_name,
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Epoch",
            yaxis_title="RMSE",
            template="plotly_white",
            width=1100,
            height=550,
        )

        fig.show()
        fig.write_image(os.path.join(save_dir, "rmse_history_comparison.png"))
    


    def plot_filtered_rmse_history(self,rmse_histories: dict,save_dir: str,title: str = "Validation RMSE comparison",):
        if not rmse_histories:
            print("⚠️ No RMSE histories to plot.")
            return

        os.makedirs(save_dir, exist_ok=True)
        fig = go.Figure()

        for exp_name, hist in rmse_histories.items():
            rmse = hist["val_rmse"]
            fig.add_trace(go.Scatter(
                x=list(range(1, len(rmse) + 1)),
                y=rmse,
                mode="lines+markers",
                name=exp_name,
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Epoch",
            yaxis_title="RMSE",
            template="plotly_white",
            width=1100,
            height=550,
        )

        fig.show()
        fig.write_image(os.path.join(save_dir, "rmse_history_comparison.png"))


    def plot_filtered_results(self,save_dir: str,results_subdir: str = "results",):
        """
        Collect results from all MLflow experiments, filter by RMSE threshold,
        and plot forecasts + RMSE histories.

        Parameters
        ----------
        save_dir : str
            Directory where plots will be saved
        results_subdir : str
            Subdirectory inside MLflow artifacts where results are stored
        """

        all_results = []
        all_labels = []
        all_rmse_histories = {}
        all_final_metrics = {}

        experiments = self.get_experiments()

        for exp_name in experiments:
            trials_df = self.list_experiment_runs(exp_name)

            (
                exp_results,
                exp_labels,
                rmse_hist,
                final_metrics,
            ) = self.load_mlflow_runs_below_rmse(
                trials_df,
                results_subdir=results_subdir,
            )

            all_results.extend(exp_results)
            all_labels.extend(exp_labels)
            all_rmse_histories.update(rmse_hist)
            all_final_metrics.update(final_metrics)

        #print(f"✅ Collected {len(all_results)} valid runs")

        # -----------------------------
        # Plot forecasts
        # -----------------------------
        if all_results:
            self.plot_multiple_experiments_forecast(
                experiment_results=all_results,
                model_names=all_labels,
                test_df=self.test_df,
                save_dir=save_dir,
            )
        else:
            print("⚠️ No experiment results passed the RMSE threshold")

        # -----------------------------
        # Plot RMSE history
        # -----------------------------
        self.plot_multiple_experiments_rmse_history(
            rmse_histories=all_rmse_histories,
            save_dir=save_dir,
        )

        return {
            "results": all_results,
            "labels": all_labels,
            "rmse_histories": all_rmse_histories,
            "final_metrics": all_final_metrics,
        }