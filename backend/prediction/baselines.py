"""
FLUX Prediction — Foundation-Model Baselines (Chronos + StatsForecast, Phase 8–9)
================================================================================
The honesty contract says: if our engineered, meta-labeled XGBoost can't beat naive baselines
out-of-sample, it's overfit. Persistence and ARIMA(1,0,0) are checked in train.py. This adds the
hardest *free* foundation/classical baselines used the same leak-free way (each fits on a trailing
window only, forecasts `horizon` steps, and is scored on the SIGN of the move vs realized):

  • **Amazon Chronos** (`chronos_directional_baseline`)   — a pretrained TS foundation model, ZERO-SHOT.
  • **StatsForecast** (`statsforecast_directional_baseline`) — fast classical AutoARIMA / AutoETS /
    AutoTheta (Phase 9). The workflow lists these explicitly as honesty baselines.

We always report each baseline's accuracy AND the same-sample up-drift, so the real EDGE-OVER-DRIFT
is visible — comparing raw accuracy across different samples (or against an up-biased universe)
would be misleading. Edge-over-drift is the project's sanctioned cross-sample-comparable metric.

Honest framing (Phase 9, GATE-9): on raw 5-day direction the universe drifts up, so always-up and
even a cheap ARIMA look strong on RAW accuracy — that is the efficient-market ceiling,
not skill. The thing FLUX-X has that NONE of these sign-forecasters do is calibrated **AUC ranking
skill**, **selective precision** (meta-labeling), and a deployable **cross-sectional Sharpe**
(GATE-7) — you cannot build the cross-sectional book out of a single ARIMA point forecast. So GATE-9
is judged on edge-over-drift + AUC + Sharpe, not on drift-dominated raw accuracy.

Degrades gracefully: a missing package returns status='unavailable' (never crashes the gate).

Public API:
    await chronos_directional_baseline(symbols, horizon, model, max_points)       -> dict
    await statsforecast_directional_baseline(symbols, horizon, max_points, window) -> dict
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

log = logging.getLogger("flux.prediction.baselines")

_DEFAULT_MODEL = "amazon/chronos-bolt-tiny"  # ~9M params, CPU-friendly, fast
_CONTEXT = 256  # trailing days fed to the model
_pipe_cache: dict = {}


def _load_pipeline(model: str):
    """Load a Chronos pipeline (cached). Returns None if the package is unavailable."""
    if model in _pipe_cache:
        return _pipe_cache[model]
    try:
        import torch
        from chronos import BaseChronosPipeline

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe = BaseChronosPipeline.from_pretrained(
                model, device_map="cpu", torch_dtype=torch.float32
            )
        _pipe_cache[model] = pipe
        return pipe
    except Exception as exc:
        log.warning("Chronos unavailable (%s): %s", model, exc)
        _pipe_cache[model] = None
        return None


async def chronos_directional_baseline(
    symbols: list[str],
    horizon: int = 5,
    model: str = _DEFAULT_MODEL,
    max_points: int = 30,
    start: str = "2015-01-01",
) -> dict:
    """
    Zero-shot directional accuracy of Chronos over sampled dates. Returns
    {status, accuracy, drift, edge, n, model}. status='unavailable' if Chronos isn't installed.
    """
    import torch

    pipe = _load_pipeline(model)
    if pipe is None:
        return {"status": "unavailable", "model": model}

    from ..db import get_history

    correct = total = ups = 0
    for sym in symbols:
        rows = await get_history(sym, start)
        if len(rows) < _CONTEXT + horizon + 10:
            continue
        df = pd.DataFrame(rows)
        close = df["adj_close"].astype(float).values
        lo, hi = _CONTEXT, len(close) - horizon - 1
        step = max(horizon, (hi - lo) // max_points)
        for t in range(lo, hi, step):
            ctx = torch.tensor(close[t - _CONTEXT : t], dtype=torch.float32)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    q, mean = pipe.predict_quantiles(
                        ctx, prediction_length=horizon, quantile_levels=[0.5]
                    )
                fc = float(mean[0, -1])  # forecasted price at t+horizon
            except Exception:
                continue
            actual = close[t + horizon] / close[t] - 1.0
            pred = fc / close[t] - 1.0
            correct += int((pred > 0) == (actual > 0))
            ups += int(actual > 0)
            total += 1
    if not total:
        return {"status": "no_data", "model": model}
    acc, drift = correct / total, ups / total
    return {
        "status": "ok",
        "model": model,
        "n": total,
        "accuracy": float(acc),
        "drift": float(drift),
        "edge": float(acc - drift),
    }


# ── StatsForecast classical baselines (Phase 9) ───────────────────────────────────
_SF_MODELS = ("AutoARIMA", "AutoETS", "AutoTheta")


async def statsforecast_directional_baseline(
    symbols: list[str],
    horizon: int = 5,
    max_points: int = 20,
    window: int = 500,
    start: str = "2008-01-01",
) -> dict:
    """
    Directional accuracy of fast classical models (StatsForecast AutoARIMA / AutoETS / AutoTheta),
    fit ON A TRAILING WINDOW per sampled date (leak-free) and scored on the sign of the h-step move.

    Returns {status, n, drift, per_model:{name:{accuracy,edge}}, best_edge, best_model}. Each
    model is fit on the trailing `window` log-prices < t, so there is no look-ahead. Bounded by
    `max_points` per symbol so it stays cheap. status='unavailable' if statsforecast isn't installed.
    """
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA, AutoETS, AutoTheta
    except Exception as exc:  # not installed → honest skip
        log.warning("StatsForecast unavailable: %s", exc)
        return {"status": "unavailable", "models": list(_SF_MODELS)}

    import warnings

    from ..db import get_history

    models = [AutoARIMA(), AutoETS(), AutoTheta()]
    name_of = {"AutoARIMA": "AutoARIMA", "AutoETS": "AutoETS", "AutoTheta": "AutoTheta"}
    hit = {m: 0 for m in name_of}
    total = ups = 0
    for sym in symbols:
        rows = await get_history(sym, start)
        if len(rows) < window + horizon + 10:
            continue
        df = pd.DataFrame(rows)
        close = df["adj_close"].astype(float).values
        lo, hi = window, len(close) - horizon - 1
        step = max(horizon, (hi - lo) // max_points)
        for t in range(lo, hi, step):
            y_ctx = np.log(close[t - window : t])  # trailing log-price window (< t)
            train_df = pd.DataFrame({"unique_id": sym, "ds": np.arange(window), "y": y_ctx})
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sf = StatsForecast(models=models, freq=1, n_jobs=1)
                    fc = sf.forecast(df=train_df, h=horizon)
            except Exception:
                continue
            actual = close[t + horizon] / close[t] - 1.0
            ups += int(actual > 0)
            total += 1
            last = float(y_ctx[-1])
            for col in name_of:
                if col in fc.columns:
                    pred = float(fc[col].iloc[-1]) - last  # forecast log-move at t+h
                    hit[col] += int((pred > 0) == (actual > 0))
    if not total:
        return {"status": "no_data", "models": list(_SF_MODELS)}
    drift = ups / total
    per_model = {m: {"accuracy": hit[m] / total, "edge": hit[m] / total - drift} for m in name_of}
    best_model = max(per_model, key=lambda m: per_model[m]["edge"])
    return {
        "status": "ok",
        "n": total,
        "drift": float(drift),
        "per_model": per_model,
        "best_model": best_model,
        "best_edge": float(per_model[best_model]["edge"]),
    }


if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    async def _demo():
        from backend.db import init_db

        await init_db()
        print("Loading Chronos + scoring a small sample (first run downloads the tiny model)...")
        r = await chronos_directional_baseline(["AAPL", "MSFT", "NVDA"], horizon=5, max_points=15)
        if r["status"] != "ok":
            print("  status:", r["status"])
            return
        print(
            f"  Chronos zero-shot directional accuracy: {r['accuracy']:.4f} "
            f"(vs same-sample drift {r['drift']:.4f} -> edge {r['edge']:+.4f}; n={r['n']})"
        )
        print("  (Our model's OOF edge-over-drift is the number to beat — see model_meta.json.)")

        print("\nStatsForecast classical baselines (AutoARIMA/ETS/Theta; sampled)...")
        s = await statsforecast_directional_baseline(
            ["AAPL", "MSFT", "NVDA"], horizon=5, max_points=8
        )
        if s["status"] != "ok":
            print("  status:", s["status"], "(pip install statsforecast to enable)")
        else:
            for m, v in s["per_model"].items():
                print(f"  {m:10}: acc {v['accuracy']:.4f}  edge-over-drift {v['edge']:+.4f}")
            print(f"  best: {s['best_model']} (edge {s['best_edge']:+.4f}; n={s['n']})")

    asyncio.run(_demo())
