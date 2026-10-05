"""
FLUX Prediction — GARCH Volatility (band shaping, Layer 2b)
==========================================================
Volatility clusters: calm days follow calm days, wild days follow wild days. A constant-width
prediction band ignores that — too loose in calm markets, too tight right when risk spikes.
GARCH(1,1) gives a *time-varying* volatility estimate that we use to SHAPE the conformal band
(conformal.py then GUARANTEES its coverage rate). GARCH shapes; conformal guarantees.

What this module exposes (all returns are plain decimal log-returns, not %):
  • conditional_vol(returns)        -> in-sample 1-day conditional σ_t series (causal, for
                                       normalising historical residuals during training)
  • forecast_h_vol(returns, h)      -> scalar σ of the cumulative h-day return (for serving)
  • horizon_scale(sigma_1d, h)      -> σ_1d · √h  (convert a 1-day σ to an h-day scale)

LEAK-SAFETY: GARCH's `conditional_volatility[t]` is the volatility the model expects for day t
given returns up to t-1 — it does not peek forward. The forecast for serving uses only the
in-sample fit on data ≤ today. If `arch` is unavailable or a fit fails to converge, we fall
back to an EWMA σ (RiskMetrics-style), so the pipeline never hard-depends on GARCH.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

log = logging.getLogger("flux.prediction.garch")

# arch wants returns on a ~percent scale for numerical conditioning; we scale in/out by this.
_SCALE = 100.0
_EWMA_SPAN = 20  # fallback σ span (≈ RiskMetrics λ=0.94 → span ~32; 20 is a touch faster)
_MIN_OBS = 250  # below this a GARCH fit is unreliable → use EWMA


def _to_returns(series: pd.Series) -> pd.Series:
    """Coerce a price OR return series to clean log-returns (drops the leading NaN)."""
    s = pd.Series(series).astype(float)
    # A price series is strictly positive; a return series straddles 0. No level check: coins
    # priced under $1 (TRX, ADA, DOGE) are still prices.
    if (s > 0).all():
        s = np.log(s / s.shift())
    return s.replace([np.inf, -np.inf], np.nan).dropna()


def _ewma_vol(returns: pd.Series, span: int = _EWMA_SPAN) -> pd.Series:
    """RiskMetrics-style EWMA volatility (the universal fallback)."""
    return returns.ewm(span=span, min_periods=min(span, 10)).std()


def conditional_vol(series: pd.Series) -> pd.Series:
    """
    1-day conditional volatility σ_t, indexed like the input returns. Uses GARCH(1,1) when
    available/convergent, otherwise EWMA. Causal: σ_t depends only on returns < t.
    """
    ret = _to_returns(series)
    if len(ret) < _MIN_OBS:
        return _ewma_vol(ret)
    try:
        from arch import arch_model

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(ret.values * _SCALE, mean="Zero", vol="Garch", p=1, q=1, dist="normal")
            res = am.fit(disp="off", show_warning=False)
        cv = pd.Series(res.conditional_volatility / _SCALE, index=ret.index)
        # Guard against a degenerate fit (all-equal / nan) → fall back.
        if not np.isfinite(cv.values).all() or cv.std() == 0:
            return _ewma_vol(ret)
        return cv
    except Exception as exc:  # arch missing or fit blew up
        log.debug("GARCH conditional_vol fallback to EWMA: %s", exc)
        return _ewma_vol(ret)


def forecast_h_vol(series: pd.Series, horizon: int) -> float:
    """
    σ of the cumulative `horizon`-day return, forecast from today (the last observation).
    Used at serving time to set the band width. Falls back to EWMA·√h.
    """
    ret = _to_returns(series)
    if len(ret) == 0:
        return float("nan")
    if len(ret) >= _MIN_OBS:
        try:
            from arch import arch_model

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                am = arch_model(
                    ret.values * _SCALE, mean="Zero", vol="Garch", p=1, q=1, dist="normal"
                )
                res = am.fit(disp="off", show_warning=False)
                fc = res.forecast(horizon=horizon, reindex=False)
            # Var of the h-day sum ≈ Σ daily forecast variances (innovations ~uncorrelated).
            var_h = float(np.nansum(fc.variance.values[-1])) / (_SCALE**2)
            if np.isfinite(var_h) and var_h > 0:
                return float(np.sqrt(var_h))
        except Exception as exc:
            log.debug("GARCH forecast fallback to EWMA: %s", exc)
    sigma_1d = float(_ewma_vol(ret).iloc[-1])
    return horizon_scale(sigma_1d, horizon)


def horizon_scale(sigma_1d: float, horizon: int) -> float:
    """Scale a 1-day σ to an h-day σ under the √-time rule (i.i.d. innovations)."""
    return float(sigma_1d) * np.sqrt(max(1, int(horizon)))


if __name__ == "__main__":
    # Smoke test: GARCH σ should track a synthetic volatility cluster and beat a flat guess.
    rng = np.random.default_rng(0)
    vol = np.concatenate([np.full(400, 0.01), np.full(200, 0.04), np.full(400, 0.01)])
    rets = pd.Series(rng.normal(0, vol), index=pd.date_range("2015-01-01", periods=1000, freq="B"))
    cv = conditional_vol(rets)
    print(
        f"GARCH sigma -- calm head {cv.iloc[100:300].mean():.4f}  "
        f"vs stressed mid {cv.iloc[450:550].mean():.4f}  (mid should be larger)"
    )
    print(f"5-day-ahead sigma forecast: {forecast_h_vol(rets, 5):.4f}")
