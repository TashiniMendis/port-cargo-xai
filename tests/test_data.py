import numpy as np
import pandas as pd
import pytest

from src.config import FEATURES, VESSEL_TYPES
from src.data import DataContractError, engineer_features, make_row, split, validate


def raw_frame(days=20):
    rng = np.random.default_rng(0)
    rows = []
    for port in ["Colombo", "Galle"]:
        for d in pd.date_range("2024-12-20", periods=days):
            calls = {v: int(rng.integers(0, 3)) for v in VESSEL_TYPES}
            rows.append(
                {
                    "date": d,
                    "portname": port,
                    **calls,
                    "portcalls": sum(calls.values()),
                    "portcalls_cargo": sum(calls[v] for v in VESSEL_TYPES[:4]),
                    "import": float(rng.integers(0, 50_000)),
                }
            )
    return pd.DataFrame(rows)


def test_valid_data_passes():
    report = validate(raw_frame())
    assert report["null_cells"] == 0
    assert report["missing_calendar_days"] == 0


@pytest.mark.parametrize(
    "breaker",
    [
        lambda d: d.drop(columns=["portcalls_tanker"]),
        lambda d: d.assign(portname=d["portname"].replace("Galle", "Jaffna")),
        lambda d: pd.concat([d, d.iloc[[0]]]),
        lambda d: d.assign(portcalls=d["portcalls"] + 1),
        lambda d: d.assign(**{"import": -d["import"]}),
    ],
)
def test_contract_violations_fail(breaker):
    with pytest.raises(DataContractError):
        validate(breaker(raw_frame()))


def test_rolling_features_use_only_past_days():
    feat = engineer_features(raw_frame())
    c = feat[feat["port"] == "Colombo"].reset_index(drop=True)
    # day 8's 7-day mean = mean of days 1..7, never including day 8 itself
    assert c.loc[7, "import_roll7"] == pytest.approx(c.loc[0:6, "import"].mean())
    assert np.isnan(c.loc[0, "import_roll7"])


def test_split_is_chronological_and_disjoint():
    feat = engineer_features(raw_frame(days=20))
    train, val, test, hold = split(feat, "2024-12-28", "2025-01-03")
    assert train["date"].max() < val["date"].min() <= val["date"].max() < test["date"].min()
    assert test["date"].max() < hold["date"].min()
    assert len(train) + len(val) + len(test) + len(hold) == len(feat)


def test_make_row_matches_feature_order():
    row = make_row("Colombo", "2026-07-25", {"portcalls_container": 3, "portcalls_tanker": 1})
    assert list(row.columns) == FEATURES
    assert row.loc[0, "portcalls_total"] == 4
    assert row.loc[0, "day_of_week"] == 5  # Saturday
