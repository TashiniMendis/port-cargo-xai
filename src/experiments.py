"""Stage `compare`: EBM vs standard models, same data and split, each run logged to MLflow.

Usage:  python -m src.experiments   (needs models/model.pkl from the train stage)
Writes metrics/experiments.csv and plots/model_comparison.png
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from . import tracking
from .config import METRICS, PLOTS, load_params
from .data import xy
from .evaluate import load_model, predict
from .metrics import regression_metrics
from .models import baselines
from .train import load_split

EBM_NAME = "EBM (new algorithm)"


def main():
    params = load_params()["model"]
    train = pd.concat([load_split("train"), load_split("val")], ignore_index=True)
    test = load_split("test")
    X_tr, y_tr = xy(train)
    X_te, y_te = xy(test)

    tracking.setup("port-cargo-experiments")
    rows = []
    candidates = {**baselines(params["random_state"]), EBM_NAME: None}
    for name, model in candidates.items():
        t0 = time.perf_counter()
        if model is None:
            model = load_model()  # already fitted on train + validation by the train stage
            fit_s = float("nan")
        else:
            model.fit(X_tr, y_tr)
            fit_s = time.perf_counter() - t0
        m = regression_metrics(y_te, predict(model, X_te))
        rows.append({"model": name, **m, "fit_s": fit_s})
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            mlflow.log_metrics({f"test_{k}": v for k, v in m.items()})
        print(f"[compare] {name:22s} MAE={m['mae']:9,.0f}  R2={m['r2']:.3f}")

    table = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    METRICS.mkdir(exist_ok=True)
    table.round(4).to_csv(METRICS / "experiments.csv", index=False)

    PLOTS.mkdir(exist_ok=True)
    t = table.sort_values("mae", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = ["#0f766e" if n == EBM_NAME else "#9aa5b1" for n in t["model"]]
    ax.barh(t["model"], t["mae"], color=colors)
    for i, v in enumerate(t["mae"]):
        ax.text(v, i, f" {v:,.0f}", va="center", fontsize=8)
    ax.set_xlabel("Test MAE, tonnes/day (lower is better)")
    ax.set_title("EBM vs standard models, test period 2025-2026")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOTS / "model_comparison.png", dpi=120)
    plt.close(fig)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
