from pathlib import Path
from typing import Dict, Any, Callable, Optional

import mlflow
from mlflow import sklearn as mlflow_sklearn


PROJECT_ROOT = Path(__file__).resolve().parent
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


def setup_mlflow(experiment_name: str = "hand_gesture_experiments") -> None:
    """
    Configure MLflow to use a local tracking directory and set the experiment.
    All MLflow configuration for this project should go through this function.
    """
    tracking_uri = f"file:{MLRUNS_DIR.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def start_run(run_name: str) -> None:
    """Start an MLflow run with a representative run name."""
    mlflow.start_run(run_name=run_name)


def end_run() -> None:
    """End the active MLflow run."""
    mlflow.end_run()


def log_dataset_info(path: str, n_rows: int) -> None:
    """Log basic dataset information as parameters."""
    mlflow.log_param("dataset_path", path)
    mlflow.log_param("dataset_n_rows", n_rows)


def log_model_params(params: Dict[str, Any]) -> None:
    """Log model hyper-parameters."""
    for key, value in params.items():
        mlflow.log_param(key, value)


def log_metrics(metrics: Dict[str, float]) -> None:
    """Log evaluation metrics."""
    for key, value in metrics.items():
        mlflow.log_metric(key, float(value))


def log_model(model, name: str) -> str:
    """
    Log a trained model artifact.

    Newer MLflow versions interpret the argument as a *model name*,
    so it must not contain slashes or special characters.
    Returns the logged model URI.
    """
    info = mlflow_sklearn.log_model(model, artifact_path=name)
    return info.model_uri


def run_experiment(
    *,
    experiment_name: str,
    run_name: str,
    dataset_path: str,
    dataset_n_rows: int,
    model_name: str,
    model,
    model_params: Dict[str, Any],
    metric_values: Dict[str, float],
    register_as: Optional[str] = None,
) -> Optional[str]:
    """
    High-level helper to wrap a full MLflow run.

    This does not train the model itself; the caller is responsible for
    fitting and evaluating the model, then passing the metrics here.
    """
    setup_mlflow(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        log_dataset_info(dataset_path, dataset_n_rows)
        log_model_params({"model_name": model_name, **model_params})
        log_metrics(metric_values)
        model_uri = log_model(model, f"{model_name}_model")

        if register_as is not None:
            mv = mlflow.register_model(model_uri=model_uri, name=register_as)
            return mv.version

    return None

