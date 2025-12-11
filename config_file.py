import os
import json

# -----------------------------------
# Helper function
# -----------------------------------
def create_file(path, content=""):
    """Create a file safely."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def create_project_structure(base_dir="hyperopt-llm-project"):
    # Create the root project directory
    os.makedirs(base_dir, exist_ok=True)

    # -----------------------------------
    # Folder structure (relative to base_dir)
    # -----------------------------------
    folders = [
        "data/raw_data",
        "data/clean_csv",
        "data/processed",

        "notebooks",

        "src",
        "src/data",
        "src/models",
        "src/tuning",
        "src/pipelines",
        "src/app",
        "src/config",

        "results/json_logs",
        "results/plots",

        "mlruns/mlrun_bo",
        "mlruns/mlrun_llm",

        "docker",
        ".github/workflows",
    ]

    # Create folders
    for folder in folders:
        os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

    # -----------------------------------
    # FILES
    # -----------------------------------

    # Data placeholders
    create_file(f"{base_dir}/data/raw_data/raw.csv", "")
    create_file(f"{base_dir}/data/clean_csv/clean.csv", "")
    create_file(f"{base_dir}/data/processed/preprocessed.csv", "")

    # Notebooks
    create_file(f"{base_dir}/notebooks/exploratory_analysis.ipynb", "")

    # SRC files
    create_file(f"{base_dir}/src/__init__.py", "")
    
    create_file(f"{base_dir}/src/data/data_cleaning.py",
f"""
class DataCleaning:
    \"\"\"Handles missing values, outliers, and raw data cleaning.\"\"\"
    def __init__(self):
        pass

    def load_raw_data(self, path):
        pass

    def clean(self, df):
        pass
""")

    create_file(f"{base_dir}/src/data/data_preprocessing.py",
f"""
class DataPreprocessing:
    \"\"\"Handles MinMax scaling, normalization, and transformations.\"\"\"
    def __init__(self):
        pass

    def preprocess(self, df):
        pass
""")

    create_file(f"{base_dir}/src/data/utils_data.py",
"# Utility functions for loading and saving data\n")

    # Models
    create_file(f"{base_dir}/src/models/base_model.py",
f"""
class BaseModel:
    \"\"\"Base model with fit and predict methods.\"\"\"
    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
""")

    create_file(f"{base_dir}/src/models/train.py", "\"\"\"ModelTrainer class\"\"\"")
    create_file(f"{base_dir}/src/models/evaluation.py", "\"\"\"ModelEvaluation class\"\"\"")
    create_file(f"{base_dir}/src/models/plot_results.py", "\"\"\"ResultsPlotter class\"\"\"")

    # Tuning modules
    create_file(f"{base_dir}/src/tuning/bo_tuning.py",
f"""
class BayesianOptimizer:
    \"\"\"Bayesian Optimization HPO engine.\"\"\"
    def __init__(self):
        pass

    def optimize(self, model, X, y):
        pass
""")

    create_file(f"{base_dir}/src/tuning/llm_tuning.py",
f"""
class LLMTuner:
    \"\"\"LLM-guided tuning using LangChain and MLflow.\"\"\"
    def __init__(self):
        pass

    def suggest_parameters(self, history=None):
        pass
""")

    create_file(f"{base_dir}/src/tuning/mlflow_logger.py",
f"""
class MLflowLogger:
    \"\"\"MLflow logging utility.\"\"\"
    def __init__(self):
        pass

    def log_params(self, params):
        pass

    def log_metrics(self, metrics):
        pass
""")

    create_file(f"{base_dir}/src/tuning/search_spaces.json", json.dumps({}, indent=4))

    # Pipelines
    create_file(f"{base_dir}/src/pipelines/full_pipeline.py",
f"""
class HPOPipeline:
    \"\"\"Full pipeline: data → model → tuning → evaluation.\"\"\"
    def run(self):
        pass
""")

    create_file(f"{base_dir}/src/pipelines/utils_pipeline.py", "")

    # Streamlit app
    create_file(f"{base_dir}/src/app/streamlit_app.py",
"\"\"\"Streamlit UI for running experiments and visualizing results.\"\"\"\n")

    # Config files
    create_file(f"{base_dir}/src/config/settings.yaml", "default_config: true\n")
    create_file(f"{base_dir}/src/config/credentials.yaml", "# Add your API keys here\n")
    create_file(f"{base_dir}/src/config/logger_config.yaml", "# Logging configuration\n")

    # Results
    create_file(f"{base_dir}/results/json_logs/bo_trials.json", "{}")
    create_file(f"{base_dir}/results/json_logs/llm_trials.json", "{}")
    create_file(f"{base_dir}/results/plots/performance.png", "")

    # Docker files
    create_file(f"{base_dir}/docker/Dockerfile", "# Dockerfile\n")
    create_file(f"{base_dir}/docker/docker-compose.yml", "# docker-compose config\n")

    # GitHub CI/CD
    create_file(f"{base_dir}/.github/workflows/ci-cd.yml", "# GitHub Actions pipeline\n")

    # Root-level files
    create_file(f"{base_dir}/.env", "")
    create_file(f"{base_dir}/README.md", "# Hyperparameter Optimization with LLMs\n")
    create_file(f"{base_dir}/LICENSE", "")

    print("✅ All project files created under:", base_dir)


if __name__ == "__main__":
    create_project_structure()
