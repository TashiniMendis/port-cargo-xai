"""MLflow connection: DagsHub when DAGSHUB_OWNER + DAGSHUB_TOKEN are set, else a local store."""

import os

import mlflow

from .config import ROOT


def setup(experiment: str) -> str:
    owner, token = os.getenv("DAGSHUB_OWNER"), os.getenv("DAGSHUB_TOKEN")
    repo = os.getenv("DAGSHUB_REPO", "port-cargo-xai")
    if owner and token:
        uri = f"https://dagshub.com/{owner}/{repo}.mlflow"
        os.environ.setdefault("MLFLOW_TRACKING_USERNAME", owner)
        os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", token)
    else:
        uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return uri


def log_sklearn_model(model, name: str = "model") -> str:
    """Log a scikit-learn compatible model; returns its URI (MLflow 2 and 3 compatible)."""
    # cloudpickle: MLflow 3's default (skops) refuses third-party classes such as the EBM
    try:
        info = mlflow.sklearn.log_model(model, name=name, serialization_format="cloudpickle")
    except TypeError:  # MLflow 2.x
        info = mlflow.sklearn.log_model(
            model, artifact_path=name, serialization_format="cloudpickle"
        )
    return info.model_uri
