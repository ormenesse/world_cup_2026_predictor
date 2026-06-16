# Model notebooks

Scratch space for ML experimentation — feature exploration, model
selection, hyperparameter tuning, evaluation. Once a notebook flow
stabilizes, port it to `models/train_example.py` (engine: **polars**)
and call it from a `diamond/` ETL job so it runs on a schedule.

## Suggested workflow

1. Pull curated features from the `gold` layer.
2. Iterate on a model in `train_example.ipynb`.
3. **Track experiments with MLflow** (optional but recommended):

   ```bash
   pip install mlflow
   export MLFLOW_TRACKING_URI=sqlite:///mlflow.db   # local quickstart
   # or point at your team's tracking server:
   # export MLFLOW_TRACKING_URI=https://mlflow.your-org.example.com
   mlflow ui   # open http://localhost:5000
   ```

   In your notebook:

   ```python
   import mlflow, mlflow.sklearn
   with mlflow.start_run():
       mlflow.log_param("lr", 0.01)
       mlflow.log_metric("auc", 0.87)
       mlflow.sklearn.log_model(model, artifact_path="model",
                                 registered_model_name="example_model")
   ```

4. Promote the best run in the MLflow registry, then load it from
   `models/train_example.py::load_latest()` so the diamond-layer ETL
   job picks up new versions without code changes.

## Without MLflow

If you don't want the dep, pickle models to S3/GCS/disk under a
versioned path and load by hash. The `load_latest()` stub is the
right place to wire that in.
