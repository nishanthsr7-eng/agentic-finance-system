"""
FLUX Prediction — Options-Implied Volatility & Skew (orthogonal signal source)
==============================================================================
Per-symbol signals from the listed options market: ATM implied vol (the market's forward risk
estimate), put/call **skew** (a clean fear gauge — puts bid up before downside), and the IV
**term slope** (contango vs backwardation). These are *orthogonal* to price/indicator features:
they encode forward-looking, positioning-derived information that the price series alone can't.

HONESTY / LEAKAGE NOTE — why this is a LIVE feature, not a training feature:
    Free historical option chains don't exist at scale, so IV/skew **cannot** be back-filled into
    the leak-free OOF training set without cheating. Instead we follow the same pattern as live
    sentiment: snapshot the chain daily into `options_iv` (the flywheel), expose the latest value
    in the prediction output for auditing / the LLM layer, and log it alongside each prediction so
    that — once enough real history accrues — it can be validated and promoted into the model.
    Until then it stays INERT w.r.t. the confidence number (the self-gating contract).

Equity-only: crypto/altcoin tickers have no listed chain on yfinance, so they return None and the
rest of the pipeline proceeds unchanged.

Public API:
    await compute_iv_features(symbol)      -> dict | None   # live chain → {atm_iv, skew, term_slope,...}
    await snapshot_iv(symbols)             -> int            # persist daily snapshots (scheduler)
    await symbol_iv(symbol)                -> dict           # latest stored snapshot (serving)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

log = logging.getLogger("flux.prediction.options")

_NEAR_DAYS = 30  # target tenor for the ATM/skew read (1-month standard)
_FAR_DAYS = 90  # target tenor for the far leg of the term-structure slope
_PUT_MONEY = 0.90  # OTM put strike ≈ 0.90·spot  (skew proxy — not a true 25Δ)
_CALL_MONEY = 1.10  # OTM call strike ≈ 1.10·spot
_IV_LO, _IV_HI = (
    0.01,
    5.0,
)  # drop junk/stale IVs outside this band (yfinance returns ~1e-5 for dead strikes)


def _clean_iv(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with a plausible IV and some liquidity (bid>0 or open interest)."""
    if df is None or df.empty:
        return pd.DataFrame()
    d = df[["strike", "impliedVolatility", "bid", "openInterest"]].copy()
    d = d[(d["impliedVolatility"] > _IV_LO) & (d["impliedVolatility"] < _IV_HI)]
    liquid = (d["bid"].fillna(0) > 0) | (d["openInterest"].fillna(0) > 0)
    d = d[liquid]
    return d.dropna(subset=["strike", "impliedVolatility"])


def _iv_at(df: pd.DataFrame, target_strike: float) -> float | None:
    """IV of the contract whose strike is nearest target_strike (None if none)."""
    if df.empty:
        return None
    i = (df["strike"] - target_strike).abs().idxmin()
    return float(df.loc[i, "impliedVolatility"])


def _atm_iv(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> float | None:
    """ATM IV = mean of the call and put IV at the strike nearest spot."""
    cv, pv = _iv_at(calls, spot), _iv_at(puts, spot)
    vals = [v for v in (cv, pv) if v is not None]
    return float(np.mean(vals)) if vals else None


def _pick_expiry(expiries: list[str], target_days: int) -> str | None:
    """Expiry whose calendar distance to target_days is smallest (forward-dated only)."""
    today = date.today()
    fut = [(e, (datetime.strptime(e, "%Y-%m-%d").date() - today).days) for e in expiries]
    fut = [(e, d) for e, d in fut if d >= 1]
    if not fut:
        return None
    return min(fut, key=lambda ed: abs(ed[1] - target_days))[0]


def _compute_iv_sync(symbol: str) -> dict | None:
    """Blocking yfinance read → IV/skew features. Returns None if no usable chain."""
    import yfinance as yf

    t = yf.Ticker(symbol)
    try:
        expiries = list(t.options or [])
    except Exception as exc:
        log.debug("no option chain for %s: %s", symbol, exc)
        return None
    if not expiries:
        return None

    # Spot from fast_info, fall back to last daily close.
    spot = None
    try:
        spot = t.fast_info.get("last_price")
    except Exception:
        spot = None
    if not spot:
        try:
            spot = float(t.history(period="1d")["Close"].iloc[-1])
        except Exception:
            return None
    spot = float(spot)
    if spot <= 0:
        return None

    near = _pick_expiry(expiries, _NEAR_DAYS)
    if near is None:
        return None
    nch = t.option_chain(near)
    calls, puts = _clean_iv(nch.calls), _clean_iv(nch.puts)
    n_contracts = len(calls) + len(puts)
    if n_contracts == 0:
        return None

    atm = _atm_iv(calls, puts, spot)
    put_otm = _iv_at(puts, _PUT_MONEY * spot)
    call_otm = _iv_at(calls, _CALL_MONEY * spot)
    skew = (put_otm - call_otm) if (put_otm is not None and call_otm is not None) else None

    # Term slope: far-tenor ATM IV − near-tenor ATM IV (None if no distinct far expiry).
    term_slope = None
    far = _pick_expiry([e for e in expiries if e != near], _FAR_DAYS)
    if far is not None and atm is not None:
        fch = t.option_chain(far)
        fatm = _atm_iv(_clean_iv(fch.calls), _clean_iv(fch.puts), spot)
        if fatm is not None:
            term_slope = fatm - atm

    if atm is None and skew is None:
        return None
    return {
        "symbol": symbol.upper(),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "spot": round(spot, 4),
        "atm_iv": round(atm, 4) if atm is not None else None,
        "skew": round(skew, 4) if skew is not None else None,
        "term_slope": round(term_slope, 4) if term_slope is not None else None,
        "n_contracts": int(n_contracts),
        "snapshot_at": int(datetime.now(timezone.utc).timestamp() * 1000),
    }


def _is_optionable(symbol: str) -> bool:
    """True only for US-listed equities with options. CRITICAL guard: a crypto ticker like 'BTC'
    resolves on yfinance to an UNRELATED listed security (an ETF/stock that happens to share the
    ticker), whose option chain would silently poison the feature. The authoritative equity set is
    ingestion.STOCK_META — crypto/index symbols are not in it and correctly return None."""
    try:
        from ..ingestion import STOCK_META

        return symbol.upper() in STOCK_META
    except Exception:
        return False


async def compute_iv_features(symbol: str) -> dict | None:
    """Live IV/skew features for `symbol`. Returns None for non-equities (crypto/index) and when no
    usable chain exists. The yfinance network call runs off the event loop."""
    if not _is_optionable(symbol):
        return None
    try:
        return await asyncio.to_thread(_compute_iv_sync, symbol)
    except Exception as exc:
        log.warning("IV compute failed for %s: %s", symbol, exc)
        return None


async def snapshot_iv(symbols: list[str]) -> int:
    """Compute + persist a daily IV/skew snapshot for each symbol. Returns rows written."""
    from ..db import insert_options_iv

    rows = []
    for s in symbols:
        feat = await compute_iv_features(s)
        if feat:
            rows.append(feat)
    await insert_options_iv(rows)
    if rows:
        log.info("Snapshotted options IV for %d/%d symbols", len(rows), len(symbols))
    return len(rows)


async def symbol_iv(symbol: str) -> dict:
    """Latest stored IV/skew snapshot for serving. Neutral dict if none recorded yet."""
    from ..db import get_latest_options_iv

    row = await get_latest_options_iv(symbol)
    if not row:
        return {"atm_iv": None, "skew": None, "term_slope": None, "as_of": None, "available": False}
    return {
        "atm_iv": row["atm_iv"],
        "skew": row["skew"],
        "term_slope": row["term_slope"],
        "as_of": row["date"],
        "available": True,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    async def _demo():
        from backend.db import init_db

        await init_db()
        for sym in ("AAPL", "NVDA", "TSLA", "BTC"):  # BTC is crypto -> None (graceful)
            f = await compute_iv_features(sym)
            if f:
                print(
                    f"  {sym:5} ATM IV {f['atm_iv']}  skew {f['skew']}  term {f['term_slope']}  "
                    f"(spot {f['spot']}, n={f['n_contracts']}, {f['date']})"
                )
            else:
                print(f"  {sym:5} no usable option chain (crypto/illiquid) -> skipped")

    asyncio.run(_demo())
