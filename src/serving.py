"""Prediction + explanation used by both the API and the front end."""

import json
import pickle
from functools import cached_property

import numpy as np
import pandas as pd

from .config import FEATURE_LABELS, METRICS, MODELS, PROCESSED, ROOT, VESSEL_TYPES
from .data import make_row, recent_history


def term_label(name: str) -> str:
    return " x ".join(FEATURE_LABELS.get(p, p) for p in name.split(" & "))


class Predictor:
    def __init__(self):
        with open(MODELS / "model.pkl", "rb") as f:
            self.model = pickle.load(f)
        self.history = pd.read_csv(PROCESSED / "history.csv", parse_dates=["date"])
        info = ROOT / "run_info.json"
        self.version = json.loads(info.read_text())["run_id"][:8] if info.exists() else "dev"

    @cached_property
    def metrics(self) -> dict:
        out = {}
        for name in ["eval_metrics", "per_port", "fairness", "xai_summary"]:
            p = METRICS / f"{name}.json"
            out[name] = json.loads(p.read_text()) if p.exists() else {}
        return out

    def recorded(self, port: str, date) -> dict | None:
        h = self.history
        r = h[(h["port"] == port) & (h["date"] == pd.Timestamp(date))]
        if r.empty:
            return None
        r = r.iloc[0]
        return {"import": float(r["import"]), **{v: int(r[v]) for v in VESSEL_TYPES}}

    def predict(
        self, port: str, date, calls: dict, import_roll7=None, portcalls_roll7=None
    ) -> dict:
        hist = recent_history(self.history, port, date)
        autofilled = import_roll7 is None or portcalls_roll7 is None
        if import_roll7 is None:
            import_roll7 = hist["import_roll7"]
        if portcalls_roll7 is None:
            portcalls_roll7 = hist["portcalls_roll7"]
        row = make_row(port, date, calls, import_roll7, portcalls_roll7)

        # EBM is additive: prediction = intercept + sum of term contributions (exact, no sampling)
        contrib = self.model.eval_terms(row)[0]
        raw = float(self.model.intercept_ + contrib.sum())
        terms = sorted(
            (
                {"term": n, "label": term_label(n), "value": round(float(v), 1)}
                for n, v in zip(self.model.term_names_, contrib, strict=True)
            ),
            key=lambda t: -abs(t["value"]),
        )
        port_mae = self.metrics["per_port"].get(port, {}).get("mae")
        return {
            "port": port,
            "date": str(pd.Timestamp(date).date()),
            "prediction_tonnes": round(max(0.0, raw), 0),
            "raw_prediction": round(raw, 1),
            "intercept": round(float(self.model.intercept_), 1),
            "contributions": terms,
            "typical_error_tonnes": None if port_mae is None else round(port_mae, 0),
            "inputs": {
                k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                for k, v in row.iloc[0].items()
            },
            "recent_activity_autofilled": autofilled,
            "history_window": [hist["from"], hist["to"]],
            "model_version": self.version,
        }

    def global_importance(self) -> list:
        imp = self.model.term_importances()
        order = np.argsort(imp)[::-1]
        return [
            {
                "term": self.model.term_names_[i],
                "label": term_label(self.model.term_names_[i]),
                "importance": round(float(imp[i]), 1),
            }
            for i in order
        ]

    def shape_function(self, feature: str) -> dict:
        names = list(self.model.term_names_)
        d = self.model.explain_global().data(names.index(feature))
        return {
            k: (list(map(str, v)) if k == "names" and feature == "port" else np.asarray(v).tolist())
            for k, v in d.items()
            if k in ("names", "scores", "lower_bounds", "upper_bounds")
        }
