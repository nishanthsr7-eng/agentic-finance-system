"""
FLUX Prediction — Triple-Barrier Labeling & Sample Weights
==========================================================
Implements the labeling scheme from López de Prado, *Advances in Financial ML*:

  • daily_vol          — EWMA volatility used to size barriers per-symbol, per-day
  • triple_barrier     — label = which barrier (profit / stop / time-out) is hit FIRST,
                         scaled by volatility (so "1%" means the same in calm & wild markets)
  • uniqueness_weights — down-weight overlapping labels (concurrency) → less overfit
  • meta_labels        — binary "should I act on the primary side?" target (precision booster)

The labels deliberately look FORWARD (that is their purpose). Each label also records `t1`,
the date it resolved — the CV splitter (cv.py) uses `t1` to PURGE training rows whose label
window overlaps the test window, killing the overlapping-label leakage.

All functions are causal-safe in the sense that the resulting columns are only ever used as
TARGETS / weights, never as model inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Volatility for barrier sizing ──────────────────────────────────────────────
def daily_vol(close: pd.Series, span: int = 20) -> pd.Series:
    """EWMA std of daily log returns (per-day volatility estimate)."""
    ret = np.log(close / close.shift())
    return ret.ewm(span=span, min_periods=span).std()


# ── Triple-barrier labeling ────────────────────────────────────────────────────
def triple_barrier(
    close: pd.Series,
    vol: pd.Series | None = None,
    horizon: int = 5,
    pt: float = 2.0,
    sl: float = 2.0,
    min_ret: float = 0.0,
) -> pd.DataFrame:
    """
    For each day t, set profit-take (+pt·vol) and stop (-sl·vol) barriers plus a vertical
    time-out at t+horizon. Return one row per event with:
        label    : +1 profit hit | -1 stop hit | 0 flat (time-out near zero)
        ret      : realized return at the resolving bar
        barrier  : 'pt' | 'sl' | 'timeout'
        t1       : index/date the label resolved (for purged CV)
    `min_ret` (in vol units) collapses tiny time-out moves to label 0 (FLAT).
    """
    close = close.sort_index()
    idx = close.index
    px = close.values.astype(float)
    n = len(px)
    if vol is None:
        vol = daily_vol(close)
    v = vol.reindex(idx).values.astype(float)

    labels = np.full(n, np.nan)
    rets = np.full(n, np.nan)
    barriers = np.empty(n, dtype=object)
    t1_pos = np.full(n, -1, dtype=int)

    for t in range(n):
        vt = v[t]
        if not np.isfinite(vt) or vt <= 0:
            continue
        end = min(t + horizon, n - 1)
        if end <= t:
            continue
        up = pt * vt
        dn = -sl * vt
        resolved = False
        for j in range(t + 1, end + 1):
            r = px[j] / px[t] - 1.0
            if r >= up:
                labels[t], rets[t], barriers[t], t1_pos[t] = 1, r, "pt", j
                resolved = True
                break
            if r <= dn:
                labels[t], rets[t], barriers[t], t1_pos[t] = -1, r, "sl", j
                resolved = True
                break
        if not resolved:
            r = px[end] / px[t] - 1.0
            thr = min_ret * vt
            lab = 1 if r > thr else (-1 if r < -thr else 0)
            labels[t], rets[t], barriers[t], t1_pos[t] = lab, r, "timeout", end

    out = pd.DataFrame(
        {
            "label": labels,
            "ret": rets,
            "barrier": barriers,
            "t1": [idx[p] if p >= 0 else pd.NaT for p in t1_pos],
        },
        index=idx,
    )
    return out.dropna(subset=["label"])


# ── Sample-uniqueness weights (concurrency-based) ─────────────────────────────
def uniqueness_weights(events: pd.DataFrame, close_index: pd.Index) -> pd.Series:
    """
    Weight each label by its average uniqueness = mean(1 / concurrency) over the bars it spans.
    Overlapping labels (crowded periods) get down-weighted so the model doesn't over-count them.
    `events` must have index = event start date and column `t1` = resolve date.
    """
    idx = close_index.sort_values()
    pos = {d: i for i, d in enumerate(idx)}
    concurrency = np.zeros(len(idx))

    spans: list[tuple[int, int]] = []
    for t0, t1 in events["t1"].items():
        if pd.isna(t1) or t0 not in pos or t1 not in pos:
            spans.append((-1, -1))
            continue
        a, b = pos[t0], pos[t1]
        concurrency[a : b + 1] += 1
        spans.append((a, b))

    inv = np.divide(1.0, concurrency, out=np.zeros_like(concurrency), where=concurrency > 0)
    weights = np.zeros(len(spans))
    for i, (a, b) in enumerate(spans):
        weights[i] = inv[a : b + 1].mean() if a >= 0 else 0.0

    w = pd.Series(weights, index=events.index)
    # Normalise so weights average 1.0 (keeps learning-rate scale intuitive).
    return w / w[w > 0].mean() if (w > 0).any() else w


# ── Meta-labeling ──────────────────────────────────────────────────────────────
def meta_labels(events: pd.DataFrame, side: pd.Series) -> pd.Series:
    """
    Binary target for the META model: 1 if acting on the primary `side` (+1/-1) would have
    been profitable under the triple-barrier outcome, else 0.
    Train the meta-model on this; at inference, trade only when meta_prob > threshold.
    """
    aligned = side.reindex(events.index)
    pnl = events["ret"] * aligned
    return (pnl > 0).astype(int)


# ── Convenience: full label set for the direction model ───────────────────────
def make_labels(
    close: pd.Series,
    horizon: int = 5,
    pt: float = 2.0,
    sl: float = 2.0,
    vol_span: int = 20,
) -> pd.DataFrame:
    """
    One call → events with triple-barrier `label`/`ret`/`barrier`/`t1`, a 3-class
    direction target `y` in {0=DOWN, 1=FLAT, 2=UP}, and per-sample `weight`.
    """
    vol = daily_vol(close, span=vol_span)
    ev = triple_barrier(close, vol, horizon=horizon, pt=pt, sl=sl)
    ev["y"] = ev["label"].map({-1: 0, 0: 1, 1: 2}).astype(int)
    ev["weight"] = uniqueness_weights(ev, close.index)
    return ev
