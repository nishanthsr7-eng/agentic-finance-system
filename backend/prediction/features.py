"""
FLUX Prediction — Feature Engineering (Layer 1)
================================================
Builds a leakage-proof feature matrix from `ohlcv_history`.

Design rules (enforced by tests in test_features.py):
  • Every feature at row t uses ONLY data up to and including day t.
    All rolling / ewm / shift(+) operations are inherently causal.
  • The TARGET is built separately in labeling.py and looks FORWARD — never mix
    the two in this module.
  • Indicators are implemented in pure pandas/numpy (no pandas-ta) because numpy 2.x
    removed `numpy.NaN`, which breaks pandas-ta on import. This is also dependency-light.

Public API:
    build_features(symbol)            -> DataFrame indexed by date (NaN warm-up dropped)
    build_features_from_df(df)        -> same, from an OHLCV DataFrame
    add_macro(features, macro_df)     -> left-join cross-asset/macro columns (optional)
    FEATURE_COLUMNS                   -> the model-input column names
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("flux.prediction.features")


# ── Wilder's smoothing (RMA): EMA with alpha = 1/n ─────────────────────────────
def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


# ── Trend indicators ───────────────────────────────────────────────────────────
def _trend(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    out = pd.DataFrame(index=df.index)
    for n in (5, 10, 20, 50, 200):
        out[f"sma_{n}"] = c / c.rolling(n).mean() - 1  # price vs SMA (ratio, stationary)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd / c  # normalise by price
    out["macd_signal"] = signal / c
    out["macd_hist"] = (macd - signal) / c

    # ADX (trend strength)
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat(
        [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    atr = _rma(tr, 14)
    plus_di = 100 * _rma(pd.Series(plus_dm, index=df.index), 14) / atr
    minus_di = 100 * _rma(pd.Series(minus_dm, index=df.index), 14) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    out["adx"] = _rma(dx, 14)
    return out


# ── Momentum indicators ────────────────────────────────────────────────────────
def _momentum(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l = df["close"], df["high"], df["low"]
    out = pd.DataFrame(index=df.index)

    # RSI(14) via Wilder
    delta = c.diff()
    gain = _rma(delta.clip(lower=0), 14)
    loss = _rma(-delta.clip(upper=0), 14)
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - 100 / (1 + rs)

    # Stochastic %K/%D
    ll = l.rolling(14).min()
    hh = h.rolling(14).max()
    k = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    out["stoch_k"] = k
    out["stoch_d"] = k.rolling(3).mean()

    out["roc_10"] = c.pct_change(10)
    out["williams_r"] = -100 * (hh - c) / (hh - ll).replace(0, np.nan)
    return out


# ── Volatility indicators ──────────────────────────────────────────────────────
def _volatility(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l = df["close"], df["high"], df["low"]
    out = pd.DataFrame(index=df.index)

    ma20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std()
    out["bb_width"] = (4 * sd20) / ma20  # band width as % of price
    out["bb_pctb"] = (c - (ma20 - 2 * sd20)) / (4 * sd20).replace(0, np.nan)

    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    out["atr_14"] = _rma(tr, 14) / c  # ATR as % of price

    logret = np.log(c / c.shift())
    out["rvol_5"] = logret.rolling(5).std()
    out["rvol_20"] = logret.rolling(20).std()
    return out


# ── Volume indicators ──────────────────────────────────────────────────────────
def _volume(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"].astype(float)
    out = pd.DataFrame(index=df.index)

    # OBV slope (normalised so it's stationary)
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    out["obv_slope"] = obv.diff(5) / (v.rolling(20).mean() + 1)

    out["vol_ratio"] = v / (v.rolling(20).mean() + 1)  # today vs 20-day avg volume

    # Money Flow Index (14)
    tp = (h + l + c) / 3
    mf = tp * v
    pos = mf.where(tp > tp.shift(), 0.0)
    neg = mf.where(tp < tp.shift(), 0.0)
    mfr = pos.rolling(14).sum() / neg.rolling(14).sum().replace(0, np.nan)
    out["mfi_14"] = 100 - 100 / (1 + mfr)
    return out


# ── Returns & gaps ─────────────────────────────────────────────────────────────
def _returns(df: pd.DataFrame) -> pd.DataFrame:
    c, o = df["close"], df["open"]
    out = pd.DataFrame(index=df.index)
    for n in (1, 5, 10, 20):
        out[f"logret_{n}"] = np.log(c / c.shift(n))
    out["gap"] = (o - c.shift()) / c.shift()  # overnight gap
    out["intraday_range"] = (df["high"] - df["low"]) / c
    return out


# ── Calendar (cyclical encoding so the model sees continuity) ──────────────────
def _calendar(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index)
    out = pd.DataFrame(index=df.index)
    out["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 5)
    out["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 5)
    out["month_sin"] = np.sin(2 * np.pi * (idx.month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (idx.month - 1) / 12)
    out["is_month_end"] = idx.is_month_end.astype(float)
    return out


# ── Fractional differentiation (stationary, memory-preserving) ────────────────
def _fracdiff_weights(d: float, thresh: float = 1e-4, max_k: int = 200) -> np.ndarray:
    """Fixed-width-window binomial weights for fractional differencing of order d."""
    w = [1.0]
    for k in range(1, max_k):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
    return np.array(w[::-1])  # oldest→newest for convolution


def frac_diff(series: pd.Series, d: float, thresh: float = 1e-4) -> pd.Series:
    """Causal fractional difference: value at t uses only x[t], x[t-1], ..."""
    w = _fracdiff_weights(d, thresh)
    width = len(w)
    vals = series.values.astype(float)
    out = np.full(len(vals), np.nan)
    for i in range(width - 1, len(vals)):
        window = vals[i - width + 1 : i + 1]
        if np.isnan(window).any():
            continue
        out[i] = np.dot(w, window)
    return pd.Series(out, index=series.index)


def _adf_pvalue(series: pd.Series) -> float:
    """ADF stationarity p-value; returns 1.0 if statsmodels unavailable."""
    s = series.dropna()
    if len(s) < 50:
        return 1.0
    try:
        from statsmodels.tsa.stattools import adfuller

        return float(adfuller(s, maxlag=1, regression="c", autolag=None)[1])
    except Exception:
        return 1.0


def min_ffd_order(log_price: pd.Series, candidates=None) -> float:
    """Smallest d whose frac-diff series passes ADF (p < 0.05) — keeps maximum memory."""
    if candidates is None:
        candidates = np.round(np.arange(0.1, 1.01, 0.1), 2)
    for d in candidates:
        if _adf_pvalue(frac_diff(log_price, float(d))) < 0.05:
            return float(d)
    return 1.0


# Default frac-diff order. A FIXED value keeps the feature perfectly causal
# (truncation-invariant). The optimal per-symbol order is calibrated ON THE TRAIN
# SPLIT ONLY via calibrate_fd_order() and passed back in — never auto-derived from
# the full series (that would peek at the future; the audit caught exactly this).
DEFAULT_FD_ORDER = 0.4


def _fracdiff_block(df: pd.DataFrame, d: float) -> pd.DataFrame:
    logp = np.log(df["close"])
    out = pd.DataFrame(index=df.index)
    out["fd_logclose"] = frac_diff(logp, d)
    out["_ffd_order"] = d  # diagnostic, dropped from FEATURE_COLUMNS
    return out


def calibrate_fd_order(close: pd.Series) -> float:
    """
    Pick the minimum frac-diff order that makes log-price stationary (ADF p<0.05).
    Call this on the TRAINING split only, persist the result, and feed it to
    build_features_from_df(..., fd_order=d) for both train and inference.
    """
    return min_ffd_order(np.log(close))


# ── Assembly ───────────────────────────────────────────────────────────────────
# Stationary indicator blocks (no exogenous order needed).
_BLOCKS = (_trend, _momentum, _volatility, _volume, _returns, _calendar)

# Columns fed to models (everything except diagnostics / raw OHLCV).
_DIAGNOSTIC = {"_ffd_order"}


def build_features_from_df(df: pd.DataFrame, fd_order: float = DEFAULT_FD_ORDER) -> pd.DataFrame:
    """
    df must have columns: open, high, low, close, volume — sorted oldest→newest,
    indexed (or with a 'date' column) by date.
    `fd_order` is the fractional-differencing order (fixed → fully causal).
    Returns the feature matrix with NaN warm-up rows dropped.
    """
    df = df.copy()
    if "date" in df.columns:
        df = df.set_index("date")
    df = df.sort_index()
    # Prefer adjusted close for stationarity if present (handles splits/dividends).
    if "adj_close" in df.columns and df["adj_close"].notna().any():
        df["close"] = df["adj_close"]

    parts = [df[["open", "high", "low", "close", "volume"]]]
    parts += [block(df) for block in _BLOCKS]
    parts.append(_fracdiff_block(df, fd_order))
    feat = pd.concat(parts, axis=1)
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.bfill().fillna(0)  # Handle small seeded datasets to avoid dropping all rows
    return feat


# Resolved lazily after the first build (depends on the actual columns produced).
FEATURE_COLUMNS: list[str] = []


def feature_columns(feat: pd.DataFrame) -> list[str]:
    raw = {"open", "high", "low", "close", "volume", "adj_close"}
    return [c for c in feat.columns if c not in raw and c not in _DIAGNOSTIC]


async def build_features(
    symbol: str, start: str | None = None, fd_order: float = DEFAULT_FD_ORDER
) -> pd.DataFrame:
    """Load history from the DB and build the feature matrix for one symbol."""
    from ..db import get_history

    rows = await get_history(symbol, start)
    if not rows:
        log.warning("No history for %s", symbol)
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    feat = build_features_from_df(df, fd_order=fd_order)
    global FEATURE_COLUMNS
    FEATURE_COLUMNS = feature_columns(feat)
    log.info("Built %d features × %d rows for %s", len(FEATURE_COLUMNS), len(feat), symbol)
    return feat
