"""Stage `evaluate`: score the trained EBM on the untouched 2025-2026 test set.

Usage:  python -m src.evaluate
Writes metrics/eval_metrics.json, metrics/holdout_predictions.csv, plots/actual_vs_predicted.png,
plots/residuals.png
"""

import json
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from . import tracking
from .config import METRICS, MODELS, PLOTS, PORTS, ROOT, TARGET
from .data import xy
from .metrics import regression_metrics
from .train import load_split

BLUE, GREY = "#2563eb", "#9aa5b1"


def load_model():
    with open(MODELS / "model.pkl", "rb") as f:
        return pickle.load(f)


def predict(model, X) -> np.ndarray:
    return np.clip(model.predict(X), 0, None)  # tonnage cannot be negative


def per_port(test: pd.DataFrame, pred: np.ndarray) -> dict:
    out = {}
    for port in PORTS:
        m = (test["port"] == port).to_numpy()
        y = test.loc[m, TARGET].to_numpy()
        out[port] = {
            **regression_metrics(y, pred[m]),
            "bias": float((pred[m] - y).mean()),  # + = over-prediction
            "mean_actual": float(y.mean()),
            "rows": int(m.sum()),
        }
    return out


def plots(test, pred):
    PLOTS.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
    c = (test["port"] == "Colombo").to_numpy()
    s = (
        pd.DataFrame(
            {"Actual": test.loc[c, TARGET].to_numpy(), "EBM": pred[c]}, index=test.loc[c, "date"]
        )
        .rolling(7)
        .mean()
    )
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(s.index, s["Actual"] / 1e3, color="black", lw=1.5, label="Actual")
    ax.plot(s.index, s["EBM"] / 1e3, color=BLUE, lw=1.3, label="EBM prediction")
    ax.set_ylabel("'000 tonnes/day (7-day mean)")
    ax.set_title("Colombo: actual vs predicted daily import, test period 2025-2026")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "actual_vs_predicted.png")
    plt.close(fig)

    res = test[TARGET].to_numpy() - pred
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(res / 1e3, bins=70, color=BLUE)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Actual - predicted ('000 tonnes)")
    ax.set_ylabel("Port-days")
    ax.set_title("EBM residuals on the test set")
    fig.tight_layout()
    fig.savefig(PLOTS / "residuals.png")
    plt.close(fig)


def main():
    model = load_model()
    test, holdout = load_split("test"), load_split("holdout")
    X_te, y_te = xy(test)
    pred = predict(model, X_te)
    overall = regression_metrics(y_te, pred)
    ports = per_port(test, pred)
    flat = {f"test_{k}": round(v, 4) for k, v in overall.items()}
    flat.update({f"mae_{p.lower()}": round(v["mae"], 2) for p, v in ports.items()})
    flat["mean_prediction"] = round(float(pred.mean()), 2)

    METRICS.mkdir(exist_ok=True)
    (METRICS / "eval_metrics.json").write_text(json.dumps(flat, indent=2))
    (METRICS / "per_port.json").write_text(json.dumps(ports, indent=2))
    hold = holdout[["date", "port"]].copy()
    hold["actual"] = holdout[TARGET].to_numpy()
    hold["predicted"] = predict(model, xy(holdout)[0]).round(0)
    hold.to_csv(METRICS / "holdout_predictions.csv", index=False)
    plots(test, pred)

    run_info = json.loads((ROOT / "run_info.json").read_text())
    tracking.setup("port-cargo-ebm")
    with mlflow.start_run(run_id=run_info["run_id"]):
        mlflow.log_metrics({k: float(v) for k, v in flat.items()})
        for f in ["actual_vs_predicted.png", "residuals.png"]:
            mlflow.log_artifact(str(PLOTS / f), "plots")
    print(json.dumps(flat, indent=2))
    print(hold.to_string(index=False))


if __name__ == "__main__":
    main()
