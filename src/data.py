"""Stage `prepare`: validate the raw data, engineer features, split chronologically.

Usage:  python -m src.data
Writes data/processed/{train,val,test,holdout,history}.csv and metrics/data_report.json
"""

import json

import numpy as np
import pandas as pd

from .config import FEATURES, METRICS, PORTS, PROCESSED, ROOT, TARGET, VESSEL_TYPES, load_params

REQUIRED_COLUMNS = ["date", "portname", *VESSEL_TYPES, "portcalls", "portcalls_cargo", TARGET]


class DataContractError(ValueError):
    """Raised when the raw file breaks an assumption the model relies on."""


def load_raw(path=None) -> pd.DataFrame:
    path = path or ROOT / load_params()["data"]["raw_path"]
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.sort_values(["portname", "date"]).reset_index(drop=True)


def validate(df: pd.DataFrame) -> dict:
    """Data contract: fail fast, before training, if the data is not what the model expects."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataContractError(f"missing columns: {missing}")
    if df[REQUIRED_COLUMNS].isna().any().any():
        raise DataContractError("null values in required columns")
    if (df[VESSEL_TYPES + [TARGET]] < 0).any().any():
        raise DataContractError("negative counts or tonnage")
    unknown = set(df["portname"]) - set(PORTS)
    if unknown:
        raise DataContractError(f"unknown ports: {sorted(unknown)}")
    if df.duplicated(["portname", "date"]).any():
        raise DataContractError("duplicate port-days")
    if not (df["portcalls"] == df[VESSEL_TYPES].sum(axis=1)).all():
        raise DataContractError("portcalls is not the sum of the vessel-type columns")

    span = df.groupby("portname")["date"].agg(lambda s: (s.max() - s.min()).days + 1)
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "ports": sorted(df["portname"].unique().tolist()),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "null_cells": int(df.isna().sum().sum()),
        "duplicate_port_days": 0,
        "missing_calendar_days": int((span - df.groupby("portname").size()).sum()),
        "constant_columns": [c for c in df.columns if df[c].nunique() == 1],
        "import_on_zero_call_days": int(((df["portcalls"] == 0) & (df[TARGET] > 0)).sum()),
        "zero_import_share": round(float((df[TARGET] == 0).mean()), 4),
    }


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Model features. Rolling features look only at PAST days (shift(1)), so no leakage.

    Dropped: country/ISO3 (constant), portid (duplicate of portname), portcalls_cargo and
    portcalls (sums of the vessel-type columns), every other import_*/export_* column
    (same-day cargo estimates = target leakage).
    """
    out = pd.DataFrame({"date": df["date"], "port": df["portname"]})
    for v in VESSEL_TYPES:
        out[v] = df[v].astype(int)
    out["portcalls_total"] = out[VESSEL_TYPES].sum(axis=1)
    out["day_of_week"] = out["date"].dt.day_of_week
    out["month"] = out["date"].dt.month
    out[TARGET] = df[TARGET].astype(float)
    g = out.groupby("port")
    out["import_roll7"] = g[TARGET].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    out["portcalls_roll7"] = g["portcalls_total"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    )
    # First day of each port has no history -> NaN (EBM and the baselines' imputers handle it)
    return out.sort_values(["date", "port"]).reset_index(drop=True)


def split(feat: pd.DataFrame, val_start: str, test_start: str):
    """Chronological train / validation / test, plus the last day held out for inference."""
    last_day = feat["date"].max()
    holdout = feat[feat["date"] == last_day]
    rest = feat[feat["date"] < last_day]
    train = rest[rest["date"] < val_start]
    val = rest[(rest["date"] >= val_start) & (rest["date"] < test_start)]
    test = rest[rest["date"] >= test_start]
    return train, val, test, holdout


def xy(frame: pd.DataFrame):
    return frame[FEATURES], frame[TARGET]


def make_row(port, date, calls: dict, import_roll7=None, portcalls_roll7=None) -> pd.DataFrame:
    """One input row in model format (used by the API and the front end)."""
    date = pd.Timestamp(date)
    row = {"port": port, **{v: int(calls.get(v, 0)) for v in VESSEL_TYPES}}
    row["portcalls_total"] = sum(row[v] for v in VESSEL_TYPES)
    row["day_of_week"] = date.day_of_week
    row["month"] = date.month
    row["import_roll7"] = np.nan if import_roll7 is None else float(import_roll7)
    row["portcalls_roll7"] = np.nan if portcalls_roll7 is None else float(portcalls_roll7)
    return pd.DataFrame([row])[FEATURES]


def recent_history(history: pd.DataFrame, port: str, date) -> dict:
    """Previous-7-day means for a port before `date`; falls back to the latest 7 days."""
    p = history[history["port"] == port].sort_values("date")
    past = p[p["date"] < pd.Timestamp(date)].tail(7)
    if past.empty:
        past = p.tail(7)
    return {
        "import_roll7": float(past[TARGET].mean()),
        "portcalls_roll7": float(past["portcalls_total"].mean()),
        "from": str(pd.Timestamp(past["date"].min()).date()),
        "to": str(pd.Timestamp(past["date"].max()).date()),
    }


def main():
    params = load_params()["data"]
    raw = load_raw()
    report = validate(raw)
    feat = engineer_features(raw)
    train, val, test, holdout = split(feat, params["val_start"], params["test_start"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(exist_ok=True)
    for name, frame in [("train", train), ("val", val), ("test", test), ("holdout", holdout)]:
        frame.to_csv(PROCESSED / f"{name}.csv", index=False)
    # Compact history the app uses to auto-fill the rolling features
    feat[["date", "port", "portcalls_total", TARGET] + VESSEL_TYPES].to_csv(
        PROCESSED / "history.csv", index=False
    )
    report["splits"] = {
        n: {"rows": len(f), "from": str(f["date"].min().date()), "to": str(f["date"].max().date())}
        for n, f in [("train", train), ("val", val), ("test", test), ("holdout", holdout)]
    }
    (METRICS / "data_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["splits"], indent=2))


if __name__ == "__main__":
    main()
