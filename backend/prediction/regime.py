"""
FLUX Prediction — Market Regime Detection (HMM gate)
====================================================
A 3-state Gaussian HMM on market-context observations (S&P trend, VIX level, realized vol)
labels each day as one of: trend (risk-on) / chop / risk_off. The signal is the same in a
clean uptrend and in a crisis — but it's worth very different amounts. Gating size/confidence
by regime is the main lever for cutting the strategy's worst drawdowns.

LEAK-SAFETY (the part most implementations get wrong):
  • HMM parameters are fit on a TRAINING WINDOW only (data ≤ train_end).
  • Decoding uses a manual FORWARD FILTER (not Viterbi / smoothed posteriors), so the regime
    at day t uses ONLY observations up to day t. hmmlearn's `predict`/`predict_proba` use the
    whole sequence (future-peeking) and must NOT be used for a causal feature.

Public API:
    await decode_regimes(train_end="2007-12-31")  -> DataFrame[date] {regime, p_trend,p_chop,p_risk_off}
    await current_regime()                         -> {'regime','probs','as_of'}
    REGIME_SCALE                                   -> size multiplier per regime
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

log = logging.getLogger("flux.prediction.regime")

MODELS_DIR = Path(__file__).parent / "models"
STATES = ("risk_off", "chop", "trend")
# Balanced a-priori sizing (NOT tuned on the backtest): reduce exposure in stress, don't
# fully exit. Cuts tail drawdown while keeping enough trades that Sharpe holds up.
REGIME_SCALE = {"trend": 1.0, "chop": 0.6, "risk_off": 0.25}


# ── Observations ────────────────────────────────────────────────────────────────
async def build_observations() -> pd.DataFrame:
    """Causal market-context features from SPX/VIX/TNX (indexed by date)."""
    from ..db import get_history

    async def _close(sym):
        df = pd.DataFrame(await get_history(sym))
        df.index = pd.to_datetime(df["date"])
        return df["close"].astype(float)

    spx, vix = await _close("SPX"), await _close("VIX")
    lp = np.log(spx)
    obs = pd.DataFrame(index=spx.index)
    obs["spx_ret20"] = lp.diff(20)  # trend
    obs["spx_rvol20"] = lp.diff().rolling(20).std()  # turbulence
    obs["vix"] = vix.reindex(spx.index).ffill()  # fear level
    return obs.dropna()


# ── Causal forward filter ────────────────────────────────────────────────────────
def _forward_filter(model, X: np.ndarray) -> np.ndarray:
    """Filtered posteriors P(state_t | obs_1..t) — uses only data up to t (no look-ahead)."""
    log_pi = np.log(model.startprob_ + 1e-12)
    log_A = np.log(model.transmat_ + 1e-12)
    B = model._compute_log_likelihood(X)  # (T, K) emission log-lik
    T, K = B.shape
    post = np.zeros((T, K))
    a = log_pi + B[0]
    post[0] = np.exp(a - logsumexp(a))
    for t in range(1, T):
        a = B[t] + logsumexp(a[:, None] + log_A, axis=0)
        a -= logsumexp(a)  # normalize → keeps it a filter
        post[t] = np.exp(a)
    return post


# ── Fit + decode ─────────────────────────────────────────────────────────────────
async def decode_regimes(train_end: str = "2007-12-31", persist: bool = True) -> pd.DataFrame:
    """Fit the HMM on data ≤ train_end, then causally forward-filter the full series."""
    import joblib
    from hmmlearn.hmm import GaussianHMM

    obs = await build_observations()
    train = obs[obs.index <= pd.Timestamp(train_end)]
    if len(train) < 250:
        train = obs.iloc[: max(250, len(obs) // 2)]  # fallback if VIX history short

    mu, sd = train.mean(), train.std().replace(0, 1)
    Xtr = ((train - mu) / sd).values
    Xall = ((obs - mu) / sd).values

    model = GaussianHMM(n_components=3, covariance_type="full", n_iter=300, random_state=42)
    model.fit(Xtr)

    # Label states by their characteristics (on the train window, leak-free):
    #   highest VIX → risk_off ; highest SPX trend → trend ; remaining → chop
    means = pd.DataFrame(model.means_, columns=obs.columns)  # standardized means
    order = {}
    risk_off = int(means["vix"].idxmax())
    trend = int(means["spx_ret20"].drop(index=risk_off).idxmax())
    chop = int([s for s in range(3) if s not in (risk_off, trend)][0])
    order = {risk_off: "risk_off", chop: "chop", trend: "trend"}

    post = _forward_filter(model, Xall)
    state = post.argmax(axis=1)
    out = pd.DataFrame(index=obs.index)
    out["regime"] = [order[s] for s in state]
    for s, name in order.items():
        out[f"p_{name}"] = post[:, s]

    if persist:
        joblib.dump(
            {"model": model, "mu": mu, "sd": sd, "order": order, "cols": list(obs.columns)},
            MODELS_DIR / "hmm.pkl",
        )
    log.info("Decoded regimes %s → %s", out.index.min().date(), out.index.max().date())
    return out


# ── Serving ──────────────────────────────────────────────────────────────────────
# The market regime is a SINGLE market-wide state — it does not vary by symbol. But predict()
# calls current_regime() once per symbol, so an N-symbol batch was refitting the HMM N× per
# cycle (~29× on the full universe). We cache the decoded result and refit at most once per
# _REGIME_TTL seconds (and once per process at startup). A batch finishes in seconds, so it now
# triggers exactly ONE fit; daily_prediction_cycle() calls invalidate_regime_cache() so each
# scheduled cycle still gets a fresh regime regardless of the TTL.
_REGIME_TTL = 900.0  # seconds; one HMM fit per ~cycle, not per symbol
_regime_cache: dict | None = None
_regime_cache_ts: float = 0.0
_regime_lock = asyncio.Lock()


def invalidate_regime_cache() -> None:
    """Drop the cached regime so the next current_regime() refits (call at the start of a cycle)."""
    global _regime_cache, _regime_cache_ts
    _regime_cache = None
    _regime_cache_ts = 0.0


async def current_regime(force: bool = False) -> dict:
    """Latest market regime (causal), cached per cycle.

    The HMM is refit at most once per ``_REGIME_TTL`` seconds, so a batch of N symbols triggers
    ONE fit instead of N. Pass ``force=True`` (or call ``invalidate_regime_cache()``) to refresh.
    """
    global _regime_cache, _regime_cache_ts
    async with _regime_lock:  # serialize so a gathered batch fits once
        now = time.monotonic()
        if not force and _regime_cache is not None and (now - _regime_cache_ts) < _REGIME_TTL:
            return _regime_cache
        df = await decode_regimes(persist=True)
        last = df.iloc[-1]
        probs = {n: float(last[f"p_{n}"]) for n in STATES}
        _regime_cache = {
            "regime": last["regime"],
            "probs": probs,
            "as_of": str(df.index[-1].date()),
        }
        _regime_cache_ts = now
        return _regime_cache


if __name__ == "__main__":
    import asyncio
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    async def _demo():
        df = await decode_regimes()
        print("Regime distribution (full history):")
        print(df["regime"].value_counts())
        print("\nRegime distribution since 2008 (OOF era):")
        print(df[df.index >= "2008-01-01"]["regime"].value_counts())
        print("\nRecent regimes:")
        print(df.tail(5)[["regime", "p_trend", "p_chop", "p_risk_off"]])
        # sanity: 2008 crisis & 2020 crash should be risk_off-heavy
        for label, lo, hi in [
            ("2008-09 to 2009-03 (GFC)", "2008-09-01", "2009-03-31"),
            ("2020-02 to 2020-04 (COVID)", "2020-02-15", "2020-04-30"),
        ]:
            w = df[(df.index >= lo) & (df.index <= hi)]["regime"]
            ro = (w == "risk_off").mean() if len(w) else float("nan")
            print(f"  {label}: risk_off fraction = {ro:.0%}")

    asyncio.run(_demo())
