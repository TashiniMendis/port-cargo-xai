"""MLflow model registry.

Usage:
  python -m src.registry promote   register the model in run_info.json and point @champion at it
  python -m src.registry show      print the current champion
"""

import json
import sys

import mlflow
from mlflow import MlflowClient

from . import tracking
from .config import ROOT, load_params
from .gate import champion_metrics


def promote():
    name = load_params()["model"]["name"]
    tracking.setup("port-cargo-ebm")
    info = json.loads((ROOT / "run_info.json").read_text())
    version = mlflow.register_model(info["model_uri"], name)
    client = MlflowClient()
    client.set_registered_model_alias(name, "champion", version.version)
    print(f"{name} version {version.version} is now @champion (run {info['run_id']})")


def show():
    name = load_params()["model"]["name"]
    tracking.setup("port-cargo-ebm")
    champ = champion_metrics(name)
    print(json.dumps(champ, indent=2) if champ else f"{name} has no @champion yet")


if __name__ == "__main__":
    {"promote": promote, "show": show}[sys.argv[1] if len(sys.argv) > 1 else "show"]()
