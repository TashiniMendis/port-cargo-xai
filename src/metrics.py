import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y, p) -> dict:
    """MAE/RMSE in tonnes per day; WAPE = total absolute error / total cargo.
    (MAPE is undefined on the 61% of port-days with zero imports.)"""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)),
        # undefined when the port had no cargo at all (Galle in 2025-2026)
        "wape": float(100 * np.abs(y - p).sum() / y.sum()) if y.sum() > 0 else None,
    }
