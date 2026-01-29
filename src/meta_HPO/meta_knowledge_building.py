import mlflow
from pathlib import Path
import numpy as np
import pandas as pd
import json
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MLFLOW_PATH = "file:///" + str(ROOT_DIR / "mlruns").replace("\\", "/")
mlflow.set_tracking_uri(MLFLOW_PATH)


class MetaKnowledgeBuilder:
    def __init__(self, clean_train: pd.DataFrame, experiment_name: str, target_cols: list):
        self.clean_train = clean_train
        self.experiment_name = experiment_name
        self.target_cols = target_cols

        self.meta_dir = ROOT_DIR / "data" / "meta_data"
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------
    # Meta-initial Trials
    # ----------------------------------------------
    @staticmethod
    def extract_params(run_row: pd.Series) -> dict:
        return {
            col.replace("params.", ""): run_row[col]
            for col in run_row.index
            if col.startswith("params.")
        }

    def create_meta_trials(
        self,
        top_k: int,
        initial_trials_path: str | None = None,
    ):
        if initial_trials_path is None:
            initial_trials_path = self.meta_dir / "run_data.json"
        else:
            initial_trials_path = Path(initial_trials_path)

        rmse_metric = "metrics.rmse_test"

        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            raise ValueError(f"❌ Experiment '{self.experiment_name}' not found in MLflow")

        runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"{rmse_metric} ASC"],
            output_format="pandas",
        )

        if runs_df.empty:
            raise RuntimeError("❌ No MLflow runs found for this experiment")

        best_runs = runs_df.head(top_k)

        best_trials = []
        for idx, row in best_runs.iterrows():
            best_trials.append(
                {
                    "run_id": f"trial_{len(best_trials) + 1}",
                    "test_rmse": float(row[rmse_metric]),
                    "params": self.extract_params(row),
                }
            )

        with open(initial_trials_path, "w") as f:
            json.dump(best_trials, f, indent=2)

        return best_trials

    # ----------------------------------------------
    # Meta-features (Univariate)
    # ----------------------------------------------
    def extract_univariate_meta_features(
        self,
        series: pd.Series,
        freq: int = 24,
        acf_lags=(1, 3, 6, 12, 24, 36, 42, 60),
    ) -> dict:

        s = series.dropna().values.astype(float)
        n = len(s)

        if n == 0:
            return {"count": 0}

        # Distribution
        mean = np.mean(s)
        std = np.std(s)
        cv = std / mean if mean != 0 else None
        q25, q50, q75 = np.percentile(s, [25, 50, 75])

        # ACF / PACF
        try:
            max_lag = max(acf_lags)
            acf_vals = acf(s, nlags=max_lag, fft=True)
            pacf_vals = pacf(s, nlags=max_lag)
        except Exception:
            acf_vals = [None] * (max_lag + 1)
            pacf_vals = [None] * (max_lag + 1)

        # Stationarity
        try:
            adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
        except Exception:
            adf_stat, adf_p = None, None

        # Trend (linear regression R²)
        try:
            t = np.arange(n).reshape(-1, 1)
            lr = LinearRegression().fit(t, s)
            trend_strength_lr = lr.score(t, s)
        except Exception:
            trend_strength_lr = None

        # Decomposition
        trend_strength = seasonal_strength = resid_var = None
        if n >= 2 * freq and np.var(s) > 0:
            try:
                decomp = seasonal_decompose(s, period=freq, model="additive")
                trend = decomp.trend.dropna()
                seasonal = decomp.seasonal.dropna()
                resid = decomp.resid.dropna()

                total_var = np.var(s)
                if total_var > 0:
                    trend_strength = float(np.var(trend) / total_var)
                    seasonal_strength = float(np.var(seasonal) / total_var)
                    resid_var = float(np.var(resid))
            except Exception:
                pass

        # Peaks / troughs
        diff = np.diff(s)
        peaks = int(np.sum((diff[:-1] > 0) & (diff[1:] < 0))) if len(diff) > 1 else None
        troughs = int(np.sum((diff[:-1] < 0) & (diff[1:] > 0))) if len(diff) > 1 else None

        # Outliers (IQR rule)
        outliers = int(
            np.sum((s < q25 - 1.5 * (q75 - q25)) | (s > q75 + 1.5 * (q75 - q25)))
        )

        return {
            "count": n,
            "mean": float(mean),
            "std": float(std),
            "coef_of_variation": float(cv) if cv is not None else None,
            "min": float(np.min(s)),
            "25%": float(q25),
            "50%_median": float(q50),
            "75%": float(q75),
            "max": float(np.max(s)),
            "range": float(np.max(s) - np.min(s)),
            "iqr": float(q75 - q25),
            "skewness": float(skew(s)),
            "kurtosis": float(kurtosis(s)),
            **{f"acf_lag_{l}": float(acf_vals[l]) if acf_vals[l] is not None else None for l in acf_lags},
            **{f"pacf_lag_{l}": float(pacf_vals[l]) if pacf_vals[l] is not None else None for l in acf_lags},
            "adf_stat": float(adf_stat) if adf_stat is not None else None,
            "adf_pvalue": float(adf_p) if adf_p is not None else None,
            "trend_strength_lr": float(trend_strength_lr) if trend_strength_lr is not None else None,
            "trend_strength_decomp": trend_strength,
            "seasonal_strength_decomp": seasonal_strength,
            "residual_variance": resid_var,
            "num_peaks": peaks,
            "num_troughs": troughs,
            "zero_ratio": float(np.mean(s == 0)),
            "outlier_ratio": float(outliers / n),
        }

    # ----------------------------------------------
    # Meta-features (Multivariate)
    # ----------------------------------------------
    def extract_multivariate_meta_features(
        self,
        df: pd.DataFrame,
        target_cols: list,
        freq: int = 24,
    ) -> dict:
        return {
            col: self.extract_univariate_meta_features(df[col], freq=freq)
            for col in target_cols
            if col in df.columns
        }

    def aggregate_dataset_meta(self,meta: dict) -> dict:
        df = pd.DataFrame(meta).T

        return {
            "temporal_dependence_mean_acf1": df["acf_lag_1"].mean(),
            "stationarity_ratio": np.mean(df["adf_pvalue"] < 0.05),
            "avg_trend_strength": df["trend_strength_decomp"].mean(),
            "avg_seasonality_strength": df["seasonal_strength_decomp"].mean(),
            "avg_volatility_cv": df["coef_of_variation"].mean(),
            "avg_nonlinearity_proxy": df["skewness"].abs().mean(),
            "avg_outlier_ratio": df["outlier_ratio"].mean()
        }


    def infer_dataset_regime(self, agg: dict) -> dict:
        return {
            "temporal_dependence": "strong" if agg["temporal_dependence_mean_acf1"] > 0.9 else "moderate",
            "stationarity": "mostly stationary" if agg["stationarity_ratio"] > 0.75 else "mixed",
            "trend": "strong" if agg["avg_trend_strength"] > 0.6 else "moderate",
            "seasonality": "weak" if agg["avg_seasonality_strength"] < 0.05 else "strong",
            "volatility": "high" if agg["avg_volatility_cv"] > 0.7 else "moderate",
            "noise_level": "low" if agg["avg_outlier_ratio"] < 0.02 else "moderate",
        }

    # ----------------------------------------------
    # Serialization helpers
    # ----------------------------------------------
    def to_json_safe(self, obj):
        if isinstance(obj, dict):
            return {k: self.to_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.to_json_safe(v) for v in obj]
        elif isinstance(obj, tuple):
            return [self.to_json_safe(v) for v in obj]
        elif isinstance(obj, np.generic):
            return obj.item()
        else:
            return obj

    # ----------------------------------------------
    # Main entry
    # ----------------------------------------------
    def meta_features_creation(self):
        # 1️⃣ Extract per-column meta-features
        meta_features_data = self.extract_multivariate_meta_features(
            self.clean_train, self.target_cols
        )

        # 2️⃣ Aggregate meta-features across dataset
        agg = self.aggregate_dataset_meta(meta_features_data)

        # 3️⃣ Infer dataset regime (trend, seasonality, stationarity, etc.)
        dataset_regime = self.infer_dataset_regime(agg)

        # 4️⃣ Build the final meta-features dictionary
        meta_features_dataset = {
            "dataset_regime": dataset_regime,
            "aggregated_meta": agg,
            "raw_meta_features": meta_features_data,  # optional
        }

        # 5️⃣ Convert all np.nan or NaN to None for safe JSON
        def replace_nan_with_none(obj):
            if isinstance(obj, dict):
                return {k: replace_nan_with_none(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_nan_with_none(v) for v in obj]
            elif isinstance(obj, float) and (np.isnan(obj)):
                return None
            elif isinstance(obj, np.generic):  # handles np.int32, np.float64, etc.
                return obj.item()
            else:
                return obj

        safe_meta = replace_nan_with_none(meta_features_dataset)

        # 6️⃣ Save to JSON safely
        out_path = self.meta_dir / "meta_features.json"
        with open(out_path, "w") as f:
            json.dump(safe_meta, f, indent=2)

        return safe_meta

    
    # ----------------------------------------------
    # Target RMSE Utility
    # ----------------------------------------------
    @staticmethod
    def compute_target_rmse(trials, epsilon: float = 0.0, default: float = 1.0) -> float:
        """Target RMSE is slightly lower than best historical RMSE."""
        rmses = [t["test_rmse"] for t in trials if "test_rmse" in t and t["test_rmse"] is not None]
        if not rmses:
            return default
        return float(min(rmses) - epsilon)