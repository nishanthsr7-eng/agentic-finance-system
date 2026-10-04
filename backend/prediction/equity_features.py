"""
FLUX Prediction — Equity Fundamentals & Events Feature Block (Phase 4)
=====================================================================
Turns the one point-in-time datasource loader (``datasources/sec_fundamentals.py``) into a
per-(symbol, date) feature frame capturing the information price/technical factors *can't* see:
**earnings surprise** (SUE), **post-earnings drift** (PEAD), **revenue revision** (YoY growth),
**valuation** (earnings-yield z-score) and **accruals** (earnings quality). This is the equity-side
mirror of ``crypto_features.py`` and Phase 4 of the FLUX-X workflow.

GATE-4 OUTCOME — FAILS (within-noise tie), so the block is DEFAULT OFF. Read before changing it:
  On the identical pooled panel (scripts/gate4_equity_fundamentals_eval.py, like-for-like on shared OOF rows) the
  equity-subset OOF AUC is 0.5107 -> 0.5099 (-0.0008) vs the Phase-3 pooled model — a wash, no
  structural lift. Quarterly fundamentals are slowly-varying step functions with little directional
  power at the 5-day triple-barrier horizon (they matter at monthly+ horizons), so per the honesty
  contract they stay self-gated OFF for the cross-sectional ranker (``FLUX_EQUITY_FEATURES=1`` to opt
  in). Aside: the joint fit nudged pooled +0.0023 / crypto +0.0046 through shared-tree coupling, but
  GATE-4 is the equity metric and it didn't lift. Possible future home: an equity-only-book Sharpe
  input or a longer-horizon model (cf. how crypto-features helped a crypto-only Sharpe despite AUC).

Same honesty / leak-safety contract as ``crypto_features.py``, ``options.py`` and ``fred.py``:
  • POINT-IN-TIME. Every fundamental is keyed on its SEC **filing date** (``date``), never the
    reporting period — a quarter's numbers are public only when the 10-Q/10-K is filed, weeks after
    period end. A row for day D therefore only ever carries fundamentals filed on/before D, and the
    training pipeline reads row D to predict the move starting D+1, so there is never a forward peek.
    Every rolling/shift here is causal (filing order or trailing calendar window). The ``valuation_z``
    earnings-yield z-score uses a causal trailing window of the symbol's own daily price. (Verified by
    test_equity_features.py with a strict truncation-invariance test.)
  • NEUTRAL FOR NON-EQUITY. Crypto (and any symbol absent from the SEC universe) gets an all-zero
    block — *not* NaN — exactly like the crypto block returns zeros for equities. ``load_dataset``
    ends with a ``dropna()``, so NaN columns would silently wipe every crypto row; ``0`` is the
    sensible neutral (no surprise, average valuation, flat revision) and the tree isolates the
    "has-fundamentals vs not" partition on its own.

Cumulative-figure handling (the one subtlety): the SEC income-statement figures (revenue, net_income,
op_income, eps_diluted) are reported **YTD-cumulative within a fiscal year** — a 10-Q's Q2 value is the
H1 total, Q3 is 9-month, the 10-K FY is the full year. We reconstruct the **single-quarter** figure by
differencing within each fiscal year (Q1 = itself, Q2 = H1−Q1, Q3 = 9mo−H1, Q4 = FY−9mo). Balance-sheet
figures (assets, liabilities, equity, cash) are as-of period-end levels and need no decomposition.

Why these features (orthogonal to price):
  earnings_surprise  SUE — standardised seasonal-random-walk EPS surprise (ΔEPS vs same quarter a year
                     ago, scaled by its own recent volatility). The classic earnings-momentum signal.
  earnings_drift     PEAD — sign(SUE) decayed over the ~quarter following the filing; captures the
                     well-documented post-earnings-announcement drift the surprise step alone misses.
  rev_revision       YoY single-quarter revenue growth (same-quarter prior year) — the top-line trend.
  valuation_z        trailing-z-score of TTM earnings yield (ttm_eps / price); a causal value tilt that
                     varies cross-sectionally (it uses each symbol's own daily price).
  accruals           Δ(net operating assets)/avg-assets — the Sloan accruals anomaly (earnings backed
                     by accruals, not cash, mean-revert); higher = lower-quality earnings.

Coverage: all 15 equities in the universe have SEC fundamentals (foreign filers BABA/TSM file annual
20-F only → their quarterly decomposition collapses to the annual figure, handled gracefully). Missing
metrics for a filing leave that feature NaN → neutral-filled at the daily layer.

Public API:
    compute_equity_features(symbol, price) -> DataFrame          # aligned to price.index
    EQUITY_FEATURE_COLS                                          # the 5 column names
    EQUITY_SYMBOLS                                               # symbols with SEC fundamentals
"""

from __future__ import annotations

import logging
from functools import cache, lru_cache

import numpy as np
import pandas as pd

from .datasources.sec_fundamentals import load_sec_fundamentals

log = logging.getLogger("flux.prediction.equity_features")

EQUITY_FEATURE_COLS = [
    "earnings_surprise",  # SUE: standardised YoY EPS surprise (step, held until next filing)
    "earnings_drift",  # PEAD: sign(SUE) decayed over the quarter after the filing (daily)
    "rev_revision",  # YoY single-quarter revenue growth (step)
    "valuation_z",  # trailing z-score of TTM earnings yield (daily, uses price)
    "accruals",  # Δ(net operating assets)/avg-assets — earnings quality (step)
]

# Cumulative income-statement metrics (need within-fiscal-year differencing to a single quarter).
_CUM_METRICS = ["revenue", "net_income", "op_income", "eps_diluted"]

# Tunables (kept conservative; clipping stops one bad filing from owning a tree split).
_CLIP = 1.0  # ratio/growth features clamped to ±100%
_CLIP_Z = 4.0  # z-scores clamped to ±4σ
_SUE_WIN = 8  # filings (~2y) for the surprise-volatility scale
_SUE_MIN = 4
_DRIFT_TAU = 30.0  # PEAD decay constant (calendar days)
_DRIFT_MAX = 90  # drift fully off ~one quarter after the filing
_VAL_WIN = 252  # trading days for the earnings-yield z-score
_VAL_MIN = 60


# ── Cached fundamentals panel + per-symbol filing-level feature frame ──────────────
@lru_cache(maxsize=1)
def _fund_panel() -> pd.DataFrame:
    """The cached point-in-time SEC fundamentals panel (read once, sliced per symbol)."""
    return load_sec_fundamentals()


@lru_cache(maxsize=1)
def _equity_symbols() -> frozenset[str]:
    p = _fund_panel()
    return frozenset() if p.empty or "symbol" not in p else frozenset(p["symbol"].unique())


def clear_cache() -> None:
    """Drop memoised panels (call after refreshing Dataset/ within a live process)."""
    _fund_panel.cache_clear()
    _equity_symbols.cache_clear()
    _symbol_quarterly.cache_clear()


def _ttm(values: np.ndarray, periods: pd.Series, min_n: int) -> np.ndarray:
    """Trailing-12-month sum of a single-quarter series, by reporting *period* (causal).

    For a quarterly filer this sums the trailing 4 single quarters (= full year); for an annual
    filer it is the latest annual figure. Returns NaN until ``min_n`` filings fall inside the window,
    so a partial (e.g. 3-quarter) TTM is never emitted (which would jump the earnings yield).
    """
    p = pd.to_datetime(periods).values.astype("datetime64[ns]")
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        lo = p[i] - np.timedelta64(360, "D")
        m = (p > lo) & (p <= p[i])
        v = values[m]
        v = v[np.isfinite(v)]
        if len(v) >= min_n and np.isfinite(values[i]):
            out[i] = v.sum()
    return out


@cache
def _symbol_quarterly(symbol: str) -> pd.DataFrame | None:
    """Filing-level fundamentals features for one symbol (None if not in the SEC universe).

    Columns: date (filing), sue, rev_revision, accruals, ttm_eps. Each row is known on its filing
    date; the daily layer forward-fills from there.
    """
    panel = _fund_panel()
    if panel.empty:
        return None
    d = panel[panel["symbol"] == symbol.upper()].copy()
    if d.empty:
        return None
    return _quarterly_features(d)


def _quarterly_features(d: pd.DataFrame) -> pd.DataFrame:
    """Pure filing-level feature core (one symbol's raw SEC filings -> [date, sue, rev_revision,
    accruals, ttm_eps]). Split out from ``_symbol_quarterly`` so the leakage test can truncate the
    raw filings and assert earlier rows are invariant (every op is filing-order / period causal)."""
    d = d.copy()
    # 1. Single-quarter reconstruction: difference cumulative metrics within each fiscal year.
    #    First filing of a fiscal year (Q1, or an annual-only filer's FY) keeps its own value.
    d = d.sort_values(["fy", "period"])
    for m in _CUM_METRICS:
        if m in d:
            d[m + "_q"] = d.groupby("fy")[m].diff()
            d[m + "_q"] = d[m + "_q"].fillna(d[m])

    # 2. Order by reporting period for trailing/YoY ops. Same-fp shift(1) = the prior fiscal year's
    #    matching quarter (de-seasonalised YoY) — robust to quarterly vs annual filers alike.
    d = d.sort_values("period").reset_index(drop=True)
    is_quarterly = bool(d["fp"].isin(["Q1", "Q2", "Q3"]).any())

    eps_q = d["eps_diluted_q"] if "eps_diluted_q" in d else pd.Series(np.nan, index=d.index)
    rev_q = d["revenue_q"] if "revenue_q" in d else pd.Series(np.nan, index=d.index)

    # earnings_surprise (SUE): ΔEPS vs same quarter a year ago, scaled by its own recent volatility.
    eps_base = d.groupby("fp")["eps_diluted_q"].shift(1) if "eps_diluted_q" in d else np.nan
    surprise = eps_q - eps_base
    scale = surprise.rolling(_SUE_WIN, min_periods=_SUE_MIN).std()
    d["sue"] = (surprise / scale.replace(0, np.nan)).clip(-_CLIP_Z, _CLIP_Z)

    # rev_revision: YoY single-quarter revenue growth.
    rev_base = d.groupby("fp")["revenue_q"].shift(1) if "revenue_q" in d else np.nan
    d["rev_revision"] = (rev_q / rev_base.replace(0, np.nan) - 1.0).clip(-_CLIP, _CLIP)

    # accruals: Δ net-operating-assets / average assets (Sloan). NOA ex-cash = equity − cash
    #   (≡ assets − cash − liabilities by the accounting identity, so it needs only equity & cash).
    if {"equity", "cash", "assets"} <= set(d.columns):
        noa = d["equity"] - d["cash"]
        noa_base = noa.groupby(d["fp"]).shift(1)
        assets_base = d.groupby("fp")["assets"].shift(1)
        avg_assets = (d["assets"] + assets_base) / 2.0
        d["accruals"] = ((noa - noa_base) / avg_assets.replace(0, np.nan)).clip(-_CLIP, _CLIP)
    else:
        d["accruals"] = np.nan

    # ttm_eps for the valuation yield (trailing full year of single-quarter EPS).
    d["ttm_eps"] = _ttm(eps_q.values.astype(float), d["period"], min_n=4 if is_quarterly else 1)

    return (
        d[["date", "sue", "rev_revision", "accruals", "ttm_eps"]]
        .sort_values("date")
        .reset_index(drop=True)
    )


# ── Assembly ─────────────────────────────────────────────────────────────────────
def compute_equity_features(symbol: str, price: pd.DataFrame) -> pd.DataFrame:
    """
    Equity fundamentals features aligned to ``price.index`` (a per-symbol DatetimeIndex).

    Args:
        symbol : DB symbol (AAPL, …, or a crypto ticker).
        price  : per-symbol frame indexed by date with a ``close`` column (the feature matrix from
                 ``build_features_from_df``); ``close`` is the raw contemporaneous price feeding the
                 point-in-time earnings yield (NOT ``adj_close`` — EPS is as-reported, so it must be
                 divided by the un-back-adjusted price; splits cancel in the ratio).

    Returns a frame indexed exactly by ``price.index`` with all ``EQUITY_FEATURE_COLS``, finite and
    neutral-(0)-filled. Crypto / unknown symbols (no SEC fundamentals) get an all-zero block.
    """
    out = pd.DataFrame(0.0, index=price.index, columns=EQUITY_FEATURE_COLS)
    idx = pd.DatetimeIndex(pd.to_datetime(price.index))

    if symbol.upper() not in _equity_symbols():
        return out  # non-equity -> all neutral

    qf = _symbol_quarterly(symbol)
    if qf is None or qf.empty:
        return out

    # Step features: the filing's value is known from its filing date onward (ffill, no future peek).
    fdate = pd.DatetimeIndex(pd.to_datetime(qf["date"]))

    def _step(col: str) -> pd.Series:
        s = pd.Series(qf[col].values, index=fdate)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        return s.reindex(idx, method="ffill")

    out["earnings_surprise"] = _step("sue").fillna(0.0).values
    out["rev_revision"] = _step("rev_revision").fillna(0.0).values
    out["accruals"] = _step("accruals").fillna(0.0).values

    # earnings_drift (PEAD): sign of the most recent surprise, decayed over the ~quarter after filing.
    sue_step = _step("sue").values
    last_filed = pd.Series(fdate.values, index=fdate)
    last_filed = (
        last_filed[~last_filed.index.duplicated(keep="last")]
        .sort_index()
        .reindex(idx, method="ffill")
    )
    days = (idx.values - last_filed.values).astype("timedelta64[D]").astype(float)
    decay = np.where(
        np.isfinite(days) & (days >= 0) & (days <= _DRIFT_MAX),
        np.exp(-np.clip(days, 0, None) / _DRIFT_TAU),
        0.0,
    )
    out["earnings_drift"] = np.sign(np.nan_to_num(sue_step)) * decay

    # valuation_z: causal trailing z-score of the TTM earnings yield (uses the symbol's own price).
    ttm_eps = _step("ttm_eps").values
    close = price["close"].astype(float).values if "close" in price else np.full(len(idx), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ey = pd.Series(ttm_eps / np.where(close > 0, close, np.nan), index=idx)
    mu = ey.rolling(_VAL_WIN, min_periods=_VAL_MIN).mean()
    sd = ey.rolling(_VAL_WIN, min_periods=_VAL_MIN).std()
    out["valuation_z"] = (
        ((ey - mu) / sd.replace(0, np.nan)).clip(-_CLIP_Z, _CLIP_Z).fillna(0.0).values
    )

    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    async def _demo():
        from backend.db import get_history, init_db
        from backend.prediction.features import build_features_from_df, calibrate_fd_order

        await init_db()

        for sym in ("AAPL", "NVDA", "JPM", "TSM", "BTC"):
            rows = await get_history(sym)
            if not rows:
                print(f"  {sym:5} no history")
                continue
            df = pd.DataFrame(rows)
            d = calibrate_fd_order(pd.Series(df["adj_close"].astype(float).values[: len(df) // 2]))
            feat = build_features_from_df(df, fd_order=d)
            ef = compute_equity_features(sym, feat)
            assert ef.index.equals(feat.index), "index misalignment"
            assert np.isfinite(ef.values).all(), "non-finite equity feature"
            nz = [c for c in EQUITY_FEATURE_COLS if (ef[c].abs() > 1e-9).any()]
            t = ef.iloc[-1]
            print(
                f"  {sym:5} active={len(nz)}/{len(EQUITY_FEATURE_COLS)}  "
                f"sue={t['earnings_surprise']:+.2f} drift={t['earnings_drift']:+.2f} "
                f"rev_rev={t['rev_revision']:+.3f} val_z={t['valuation_z']:+.2f} "
                f"accr={t['accruals']:+.3f}"
            )
        print(
            "OK — crypto returns an all-zero (neutral) block; equity rows are populated & finite."
        )

    asyncio.run(_demo())
