"""Example ML training job.

Engine: polars. Iterate interactively in `model_notebooks/`, then
promote the stable bits to this module.

MLflow integration is optional but recommended for anything beyond
throwaway experiments — uncomment the `mlflow.*` calls below to get
experiment tracking and a model registry. Point MLflow at a tracking
server via the MLFLOW_TRACKING_URI env var (local sqlite, hosted,
Databricks, etc). See model_notebooks/README.md for details.
"""
from __future__ import annotations

# import mlflow              # pip install mlflow
# import mlflow.sklearn      # swap for mlflow.pytorch / .pyfunc / ...

MODEL_NAME = "example_model"


def train(features):
    """Fit a model on `features` and return it.

    Suggested MLflow flow:
        with mlflow.start_run():
            mlflow.log_param("engine", "polars")
            mlflow.log_metric("auc", auc)
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
            )
    """
    raise NotImplementedError


def load_latest(stage: str = "Production"):
    """Pull the latest registered version of MODEL_NAME from MLflow.

    Example:
        model = mlflow.pyfunc.load_model(
            f"models:/{MODEL_NAME}/{stage}"
        )
        return model

    Falls back to local disk if MLflow isn't configured.
    """
    raise NotImplementedError
