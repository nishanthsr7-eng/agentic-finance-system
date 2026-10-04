"""
Leakage & sanity tests for the Phase-2 crypto-native feature block (crypto_features.py).

Run directly:    python backend/prediction/test_crypto_features.py
Or with pytest:  pytest backend/prediction/test_crypto_features.py -q

The decisive test is `test_source_causality`: every per-source transform at day t MUST NOT change
when future source rows are removed. If it does, a rolling/shift window is peeking at the future.
The integration tests then confirm the assembled block is finite, index-aligned, and correctly
NEUTRAL (all-zero) for non-crypto symbols.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.prediction import crypto_features as cf
from backend.prediction.crypto_features import (
    CRYPTO_FEATURE_COLS,
    _dvol_feats,
    _funding_feats,
    _oi_feats,
    _onchain_feats,
    _sym_frame,
    compute_crypto_features,
)
from backend.prediction.features import build_features_from_df, calibrate_fd_order


# ── 1. Source-transform causality: truncating future rows can't move earlier values ──
def _check_source_causality() -> None:
    cut_frac = 0.6
    checks = 0

    def invariant(name, frame, fn, **kw):
        nonlocal checks
        if frame.empty:
            return
        cut = int(len(frame) * cut_frac)
        full = fn(frame, **kw)
        trunc_kw = {k: (v.iloc[:cut] if isinstance(v, pd.Series) else v) for k, v in kw.items()}
        trunc = fn(frame.iloc[:cut], **trunc_kw)
        common = full.index.intersection(trunc.index)[-150:]
        a = full.loc[common].to_numpy()
        b = trunc.loc[common].to_numpy()
        d = np.nanmax(np.abs(a - b)) if a.size else 0.0
        assert d < 1e-9, f"{name}: value changed when future removed (d={d:.2e}) -> LEAK"
        checks += 1

    f = _sym_frame(cf._funding_panel(), "BTC")
    invariant("funding", f, _funding_feats)

    oi = _sym_frame(cf._oi_panel(), "BTC")
    if not oi.empty:
        dv = pd.Series(1e9, index=oi.index)  # constant $-vol → tests OI rolling only
        invariant("oi", oi, _oi_feats, dollar_vol=dv)

    d = _sym_frame(cf._dvol_panel(), "BTC")
    invariant("dvol", d, _dvol_feats)

    c = _sym_frame(cf._onchain_panel(), "BTC")
    invariant("onchain", c, _onchain_feats)

    if checks == 0:
        pytest.skip("no seeded crypto source data available (funding/OI/dvol/onchain panels empty)")
    assert checks >= 3, "too few source panels available to test causality"
    print(f"  [1] source-causality       OK   ({checks} transforms invariant to truncation)")


# ── 2. Assembled block: finite, index-aligned, populated on crypto ────────────
def _check_crypto_block(price_btc, price_aapl, btc_ret) -> None:
    cb = compute_crypto_features("BTC", price_btc, btc_ret=btc_ret)
    assert list(cb.columns) == CRYPTO_FEATURE_COLS, "column set/order drift"
    assert cb.index.equals(price_btc.index), "index not aligned to price frame"
    assert np.isfinite(cb.to_numpy()).all(), "non-finite value in crypto block"
    active = int((cb.abs() > 1e-9).any().sum())
    assert active >= 10, f"BTC should populate most columns, got {active}"
    print(
        f"  [2] crypto-block-finite    OK   (BTC {active}/{len(CRYPTO_FEATURE_COLS)} cols active, all finite)"
    )


# ── 3. Equities are NEUTRAL (all-zero) — so load_dataset's dropna() spares their rows ──
def _check_equity_neutral(price_aapl, btc_ret) -> None:
    eq = compute_crypto_features("AAPL", price_aapl, btc_ret=btc_ret)
    assert eq.index.equals(price_aapl.index), "equity index not aligned"
    assert (eq.to_numpy() == 0.0).all(), "equity crypto block must be all-zero (neutral)"
    unknown = compute_crypto_features("NOTASYMBOL", price_aapl)
    assert (unknown.to_numpy() == 0.0).all(), "unknown symbol must be neutral"
    print("  [3] equity-neutral         OK   (AAPL & unknown symbol -> all-zero block)")


# ── 4. btc_lead_lag is causal: row D == BTC's return on D (no shift into the future) ──
def _check_btc_lead_lag(price_btc, btc_ret) -> None:
    cb = compute_crypto_features("ETH", price_btc, btc_ret=btc_ret)  # ETH borrows BTC's beta
    ref = pd.Series(btc_ret).copy()
    ref.index = pd.DatetimeIndex(ref.index)
    aligned = (
        ref.reindex(pd.DatetimeIndex(pd.to_datetime(price_btc.index)))
        .clip(-1, 1)
        .fillna(0)
        .to_numpy()
    )
    d = np.nanmax(np.abs(cb["btc_lead_lag"].to_numpy() - aligned))
    assert d < 1e-9, f"btc_lead_lag misaligned with BTC return (d={d:.2e})"
    print(f"  [4] btc-lead-lag-causal    OK   (matches close-of-D BTC return, max delta {d:.1e})")


async def _setup():
    from backend.db import get_history, init_db

    await init_db()

    async def _price(sym):
        rows = await get_history(sym)
        df = pd.DataFrame(rows)
        d = calibrate_fd_order(pd.Series(df["adj_close"].astype(float).values[: len(df) // 2]))
        return build_features_from_df(df, fd_order=d)

    brows = await get_history("BTC")
    if not brows:
        pytest.skip(
            "no seeded market data for BTC (flux_market.db not populated in this environment)"
        )
    b = pd.DataFrame(brows)
    b.index = pd.to_datetime(b["date"])
    btc_ret = np.log(b["adj_close"].astype(float)).diff()
    return await _price("BTC"), await _price("AAPL"), btc_ret


def _run() -> None:
    price_btc, price_aapl, btc_ret = asyncio.run(_setup())
    print("Running leakage & sanity tests for crypto_features:")
    _check_source_causality()
    _check_crypto_block(price_btc, price_aapl, btc_ret)
    _check_equity_neutral(price_aapl, btc_ret)
    _check_btc_lead_lag(price_btc, btc_ret)
    print("All tests passed.")


# pytest entry points (set up the frames once)
_FRAMES = None


def _frames():
    global _FRAMES
    if _FRAMES is None:
        _FRAMES = asyncio.run(_setup())
    return _FRAMES


def test_source_causality():
    _check_source_causality()


def test_crypto_block():
    _check_crypto_block(*_frames())


def test_equity_neutral():
    p_btc, p_aapl, br = _frames()
    _check_equity_neutral(p_aapl, br)


def test_btc_lead_lag():
    p_btc, p_aapl, br = _frames()
    _check_btc_lead_lag(p_btc, br)


if __name__ == "__main__":
    _run()
