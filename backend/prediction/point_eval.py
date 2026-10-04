"""
FLUX Prediction — point-forecast evaluation
===========================================

How good is the single predicted price (pred_price = last_close * (1 + pred_return))?
Everything is measured on h-day RETURNS against the naive "no change" forecast, which for
daily-ish horizons is the bar to beat (a random walk's best point forecast is today's price).

    point_forecast_report(pred, actual, lo=None, hi=None, tol=0.01) -> dict
    point_forecast_has_skill(conformal_report) -> bool

skill = 1 - MAE(model) / MAE(no change). Above 0 means the point beats "no change"; at or
below 0 it doesn't, and the page should present the RANGE, not a single target price.
"""

from __future__ import annotations

import numpy as np


def point_forecast_report(pred, actual, lo=None, hi=None, tol: float = 0.01) -> dict:
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    ok = np.isfinite(pred) & np.isfinite(actual)
    pred, actual = pred[ok], actual[ok]
    n = int(len(pred))
    if not n:
        return {"n": 0}
    err = actual - pred
    mae, mae_naive = float(np.abs(err).mean()), float(np.abs(actual).mean())
    moved = actual != 0
    rep = {
        "n": n,
        "mae": mae,
        "mae_naive": mae_naive,
        "skill": float(1 - mae / mae_naive) if mae_naive > 0 else float("nan"),
        "rmse": float(np.sqrt((err**2).mean())),
        "median_abs_err": float(np.median(np.abs(err))),
        "within_tol": float((np.abs(err) <= tol).mean()),  # |actual - pred| <= tol
        "within_tol_naive": float((np.abs(actual) <= tol).mean()),
        "direction_hit": float((np.sign(pred[moved]) == np.sign(actual[moved])).mean())
        if moved.any()
        else float("nan"),
        "tol": tol,
    }
    if lo is not None and hi is not None:
        lo, hi = np.asarray(lo, float)[ok], np.asarray(hi, float)[ok]
        rep["band_coverage"] = float(((actual >= lo) & (actual <= hi)).mean())
        rep["band_mean_width"] = float((hi - lo).mean())
    return rep


def point_forecast_has_skill(conformal_report: dict, margin: float = 0.0) -> bool:
    """True only if the shipped regressor beat 'no change' out of fold (from model_meta.json)."""
    mae, naive = conformal_report.get("mae"), conformal_report.get("mae_predict_zero")
    if not mae or not naive:
        return False
    return (1 - mae / naive) > margin
