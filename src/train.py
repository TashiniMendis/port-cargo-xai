"""Stage `train`: tune the EBM on the 2024 validation year, refit on train + validation.

Usage:  python -m src.train
Writes models/model.pkl, models/best_params.json, metrics/tuning.csv, run_info.json
"""

import itertools
import json
import pickle
import time

import mlflow
import numpy as np
import pandas as pd

from . import tracking
from .config import METRICS, MODELS, PROCESSED, ROOT, load_params
from .data import xy
from .metrics import regression_metrics
from .models import ebm


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED / f"{name}.csv", parse_dates=["date"])


def tune(params: dict, train: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    grid = params["grid"]
    keys = list(grid)
    X_tr, y_tr = xy(train)
    X_va, y_va = xy(val)
    rows = []
    for values in itertools.product(*(grid[k] for k in keys)):
        hp = dict(zip(keys, values, strict=True))
        t0 = time.perf_counter()
        model = ebm(params, params["random_state"], **hp).fit(X_tr, y_tr)
        m = regression_metrics(y_va, np.clip(model.predict(X_va), 0, None))
        rows.append(
            {**hp, **{f"val_{k}": v for k, v in m.items()}, "fit_s": time.perf_counter() - t0}
        )
        print(f"[tune] {hp}  val MAE={m['mae']:,.0f}  ({rows[-1]['fit_s']:.0f}s)")
    return pd.DataFrame(rows).sort_values("val_mae").reset_index(drop=True)


def main():
    params = load_params()["model"]
    train, val = load_split("train"), load_split("val")
    table = tune(params, train, val)
    METRICS.mkdir(exist_ok=True)
    table.round(4).to_csv(METRICS / "tuning.csv", index=False)
    # Read each column separately so integer hyperparameters stay integers
    best = {k: table.at[0, k].item() for k in params["grid"]}

    # Final model: best hyperparameters, refit on train + validation (2019-2024)
    full = pd.concat([train, val], ignore_index=True)
    X, y = xy(full)
    t0 = time.perf_counter()
    model = ebm(params, params["random_state"], **best).fit(X, y)
    fit_s = time.perf_counter() - t0

    MODELS.mkdir(exist_ok=True)
    with open(MODELS / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    (MODELS / "best_params.json").write_text(json.dumps(best, indent=2))

    uri = tracking.setup("port-cargo-ebm")
    with mlflow.start_run(run_name="ebm-train") as run:
        mlflow.log_params({**{k: v for k, v in params.items() if k != "grid"}, **best})
        mlflow.log_metrics({k: float(v) for k, v in table.iloc[0].items() if k.startswith("val_")})
        mlflow.log_metric("fit_seconds", fit_s)
        mlflow.log_artifact(str(METRICS / "tuning.csv"))
        model_uri = tracking.log_sklearn_model(model)
        run_id = run.info.run_id
    (ROOT / "run_info.json").write_text(
        json.dumps({"run_id": run_id, "model_uri": model_uri, "tracking_uri": uri}, indent=2)
    )
    print(f"best {best}  val MAE={table.iloc[0]['val_mae']:,.0f}  refit {fit_s:.0f}s  run {run_id}")


if __name__ == "__main__":
    main()
