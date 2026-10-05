"""
FLUX Prediction — Crypto-native Feature Block (Phase 2)
=======================================================
Turns the leak-safe datasource loaders (``datasources/``) into a per-(symbol, date) feature frame
that captures the information equity factors *can't* see: perpetual-swap **funding**, **open
interest**, the crypto implied-vol index (**DVOL**), **on-chain** valuation/activity, and
market-wide **sentiment / BTC beta**. This is the single biggest accuracy lever for the crypto
subset of the pooled ranker.

GATE-2 OUTCOME — AUC fails, but the deployable SHARPE *helps* (an AUC-vs-Sharpe divergence, the
mirror image of FRED). Read before changing the default:
  • AUC gate FAILS. On the identical pooled sample/splits the full block lowers crypto-subset OOF AUC
    (0.5209 -> 0.5036); the best greedy subset {btc_lead_lag, dvol_level} reaches only +0.0055 < the
    +0.01 target. (scripts/experiments/gate2_crypto_features_ablation.py, scripts/experiments/gate2_crypto_features_subset_selection.py)
  • DEPLOYABLE SHARPE HELPS. On the crypto-only long-only book the block lifts Sharpe consistently:
    top-33% 1.04->1.14, top-20% 0.97->1.07, top-10% 0.67->0.89 (+0.10..+0.22); only long/short dips
    slightly (-0.02). (scripts/experiments/gate2_crypto_features_sharpe_ablation.py, 416 rebalances)
  • WHY they disagree: global pooled AUC mixes cross-date + cross-symbol (and equity) pairs and barely
    moves, but the crypto-only portfolio only needs better WITHIN-DATE ranking among crypto names —
    which the per-symbol funding/OI/DVOL/btc-beta columns sharpen. (Unlike FRED, whose features were
    cross-sectionally CONSTANT; here only ``fng`` is, and it's the one AUC drag.)
  • FULL-UNIVERSE SHARPE (the production decider) HURTS. The crypto-only win does NOT carry to the
    mixed 33-symbol book: deployable long-only Sharpe 0.835 -> 0.670 (every config down, regime-gated
    -0.335). The pooled model is SHARED and equity events outnumber crypto ~2.7:1, so 14 columns that
    are zero on equities + noisy on crypto shift the shared fit and degrade the dominant equity
    rankings. (scripts/experiments/gate2_crypto_features_sharpe_full_universe.py)
DECISION (final): default OFF (``FLUX_CRYPTO_FEATURES=1`` to opt in). The locked full-universe model
is production and the block regresses it, so it stays OFF there. It IS a net positive on a CRYPTO-ONLY
book — enable the flag if/when a separate crypto book is run, or revisit as a regime/sizing-layer input
(Phase 6). The module is leak-safe & tested. See the four scripts/experiments/gate2_crypto_features_* scripts.

Same contract / honesty discipline as ``options.py`` and ``fred.py``:
  • LEAK-SAFE. Every value on row ``D`` is finalised by the end of UTC day ``D`` (end-of-day OI,
    the day's funding sum, daily on-chain totals, the day's F&G print, BTC's close-of-D return).
    The training pipeline uses row ``D``'s features to predict the move that *starts* on D+1, so a
    row is only ever read with a ≥1-day lag — never a forward peek. Every rolling/shift/ewm op here
    is causal. (Verified by test_crypto_features.py with a strict no-future-leak shuffle test.)
  • NEUTRAL FOR NON-CRYPTO. Equities (and any symbol outside ``CRYPTO_SYMBOLS``) get an all-zero
    block — *not* NaN. This is deliberate: ``load_dataset`` does a final ``dropna()``, so injecting
    NaN columns would silently wipe every equity row. ``0`` is a sensible neutral for each feature
    (no funding, average z-score, flat growth, neutral sentiment), and the tree can isolate the
    "crypto vs equity" partition on its own.

Why these features (orthogonal to price):
  funding      fund_level/z/flips/cum8  — leverage & directional crowding in the perp market
  open int.    oi_chg/oi_to_vol/oi_z    — position build-up vs churn (new money vs rotation)
  DVOL         dvol_level/dvol_chg      — the BTC/ETH "VIX": forward risk priced by options
  on-chain     nvt/addr_growth/tx_growth— valuation vs network usage & adoption momentum
  market       fng/btc_lead_lag         — risk appetite regime + BTC as the sector's beta driver

NOTE on coverage: DVOL is published for BTC/ETH only; Coin Metrics community on-chain covers 11 of
the 14 perps (AVAX/SOL/SHIB absent). Missing sources stay neutral (0) for those symbols — no error.
``dvol_term`` (the IV term-structure slope) is intentionally omitted: only a single DVOL tenor is
available, so a term column would be a constant 0 carrying no signal. blockchain.com's BTC series
(``blockchain_onchain.py``) is redundant here since Coin Metrics already covers BTC and is left for
optional enrichment.

Public API:
    compute_crypto_features(symbol, price, btc_ret=None) -> DataFrame   # aligned to price.index
    CRYPTO_FEATURE_COLS                                                 # the 14 column names
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import pandas as pd

from .datasources import CRYPTO_SYMBOLS
from .datasources.binance_funding import load_binance_funding
from .datasources.binance_oi import load_binance_oi
from .datasources.coinmetrics import load_coinmetrics
from .datasources.deribit_dvol import load_deribit_dvol
from .datasources.fng import load_fng

log = logging.getLogger("flux.prediction.crypto_features")

CRYPTO_FEATURE_COLS = [
    # funding (perpetual-swap carry / crowding)
    "fund_level",
    "fund_z20",
    "fund_sign_flips",
    "fund_cum8",
    # open interest (position build-up vs churn)
    "oi_chg",
    "oi_to_vol",
    "oi_z",
    # DVOL (crypto implied-vol index — BTC/ETH only)
    "dvol_level",
    "dvol_chg",
    # on-chain (valuation & network activity)
    "nvt",
    "addr_growth",
    "tx_growth",
    # market-wide (regime + BTC beta)
    "fng",
    "btc_lead_lag",
]

_CLIP = (
    1.0  # clamp ratio/return-style features to ±100% so a single bad print can't dominate a split
)


# ── Cached raw panels (read each source file ONCE, then slice per symbol) ─────────
# load_coinmetrics reads a ~4 MB CSV; without caching it would be re-read for every crypto symbol.
@lru_cache(maxsize=1)
def _funding_panel() -> pd.DataFrame:
    return load_binance_funding()


@lru_cache(maxsize=1)
def _oi_panel() -> pd.DataFrame:
    return load_binance_oi()


@lru_cache(maxsize=1)
def _dvol_panel() -> pd.DataFrame:
    return load_deribit_dvol()


@lru_cache(maxsize=1)
def _onchain_panel() -> pd.DataFrame:
    return load_coinmetrics()


@lru_cache(maxsize=1)
def _fng_panel() -> pd.DataFrame:
    return load_fng()


def clear_cache() -> None:
    """Drop the memoised panels (call after refreshing Dataset/ within a live process)."""
    for fn in (_funding_panel, _oi_panel, _dvol_panel, _onchain_panel, _fng_panel):
        fn.cache_clear()


def _sym_frame(panel: pd.DataFrame, sym: str) -> pd.DataFrame:
    """Rows of `panel` for one symbol, indexed by date (empty frame if none)."""
    if panel.empty or "symbol" not in panel:
        return pd.DataFrame()
    d = panel[panel["symbol"] == sym]
    if d.empty:
        return pd.DataFrame()
    return d.set_index("date").sort_index()


def _z(s: pd.Series, win: int, min_p: int) -> pd.Series:
    """Causal rolling z-score (mean/σ from the trailing `win` window only)."""
    mu = s.rolling(win, min_periods=min_p).mean()
    sd = s.rolling(win, min_periods=min_p).std()
    return (s - mu) / sd.replace(0, np.nan)


# ── Per-source feature transforms (pure; computed on the source's own daily index) ──
def _funding_feats(f: pd.DataFrame) -> pd.DataFrame:
    """funding panel [funding, funding_last] -> fund_level, fund_z20, fund_sign_flips, fund_cum8."""
    out = pd.DataFrame(index=f.index)
    funding = f["funding"]
    out["fund_level"] = f["funding_last"]  # level going into the next day
    out["fund_z20"] = _z(funding, 20, 10)  # is today's carry unusually rich/cheap?
    sign = np.sign(funding)
    flip = (sign != sign.shift()) & sign.shift().notna() & (sign != 0)
    out["fund_sign_flips"] = flip.rolling(8, min_periods=1).sum()  # regime churn over ~8 days
    out["fund_cum8"] = funding.rolling(8, min_periods=1).sum()  # cumulative 8-day carry
    return out


def _oi_feats(oi: pd.DataFrame, dollar_vol: pd.Series) -> pd.DataFrame:
    """OI panel [oi, oi_value, ...] + traded $-volume -> oi_chg, oi_to_vol, oi_z."""
    out = pd.DataFrame(index=oi.index)
    out["oi_chg"] = oi["oi"].pct_change().clip(-_CLIP, _CLIP)  # day-over-day position growth
    dv = dollar_vol.reindex(oi.index)
    out["oi_to_vol"] = (oi["oi_value"] / dv.replace(0, np.nan)).clip(
        0, 50
    )  # open positions vs turnover
    out["oi_z"] = _z(oi["oi"], 20, 10)  # OI extension vs trailing norm
    return out


def _dvol_feats(d: pd.DataFrame) -> pd.DataFrame:
    """DVOL panel [dvol, dvol_open] -> dvol_level (as a fraction), dvol_chg (log)."""
    out = pd.DataFrame(index=d.index)
    out["dvol_level"] = d["dvol"] / 100.0  # % -> fraction (keep scale modest)
    out["dvol_chg"] = np.log(d["dvol"] / d["dvol"].shift()).clip(-_CLIP, _CLIP)
    return out


def _onchain_feats(c: pd.DataFrame) -> pd.DataFrame:
    """On-chain panel -> nvt (z-scored), addr_growth, tx_growth (7-day log growth of 7-day means).

    NVT proxy: the Coin Metrics *community* tier exposes a transaction COUNT (TxCnt), not transfer
    VALUE, so the textbook NVT (mktcap / transfer-USD) isn't computable. We use market-cap-per-tx as
    a stand-in and z-score it (30d) so it's stationary and comparable across assets.
    """
    out = pd.DataFrame(index=c.index)
    tx = c["tx_cnt"].where(c["tx_cnt"] > 0)
    mcap_per_tx = c["cap_mkt_usd"] / tx
    out["nvt"] = _z(np.log(mcap_per_tx.where(mcap_per_tx > 0)), 30, 10)
    adr = c["adr_act"].where(c["adr_act"] > 0).rolling(7, min_periods=3).mean()
    out["addr_growth"] = np.log(adr / adr.shift(7)).clip(-_CLIP, _CLIP)
    txm = tx.rolling(7, min_periods=3).mean()
    out["tx_growth"] = np.log(txm / txm.shift(7)).clip(-_CLIP, _CLIP)
    return out


# ── Assembly ─────────────────────────────────────────────────────────────────────
def compute_crypto_features(
    symbol: str, price: pd.DataFrame, btc_ret: pd.Series | None = None
) -> pd.DataFrame:
    """
    Crypto-native features aligned to ``price.index`` (a per-symbol DatetimeIndex).

    Args:
        symbol   : DB symbol (BTC, ETH, …, or an equity ticker).
        price    : per-symbol frame indexed by date with ``close`` & ``volume`` (the feature
                   matrix from ``build_features_from_df``; ``close`` may be adjusted — that's fine,
                   it only feeds the ``oi_to_vol`` $-turnover denominator).
        btc_ret  : BTC daily log-return Series indexed by date (the sector beta driver). Broadcast
                   to every crypto symbol — including BTC, where it equals BTC's own return. ``None``
                   leaves ``btc_lead_lag`` neutral.

    Returns a frame indexed exactly by ``price.index`` with all ``CRYPTO_FEATURE_COLS``, finite and
    neutral-(0)-filled. Equities (symbol ∉ CRYPTO_SYMBOLS) get an all-zero block.
    """
    # Preserve price.index verbatim (it may be string-typed) so the caller's `feat.join(...)` aligns;
    # use a parallel DatetimeIndex only for date-based alignment of the (datetime-keyed) source panels.
    out = pd.DataFrame(0.0, index=price.index, columns=CRYPTO_FEATURE_COLS)
    idx = pd.DatetimeIndex(pd.to_datetime(price.index))
    sym = symbol.upper()
    if sym not in CRYPTO_SYMBOLS:
        return out  # non-crypto -> all neutral

    blocks: list[pd.DataFrame] = []

    f = _sym_frame(_funding_panel(), sym)
    if not f.empty:
        blocks.append(_funding_feats(f))

    oi = _sym_frame(_oi_panel(), sym)
    if not oi.empty and {"close", "volume"} <= set(price.columns):
        dollar_vol = price["close"].astype(float) * price["volume"].astype(float)
        blocks.append(_oi_feats(oi, dollar_vol))

    d = _sym_frame(_dvol_panel(), sym)
    if not d.empty:
        blocks.append(_dvol_feats(d))

    c = _sym_frame(_onchain_panel(), sym)
    if not c.empty:
        blocks.append(_onchain_feats(c))

    # Market-wide: F&G (broadcast across crypto), centred so neutral sentiment == 0 (matches the
    # equity neutral-fill). btc_lead_lag = BTC's close-of-D return, the sector's beta driver.
    fng = _fng_panel()
    if not fng.empty:
        fser = fng.set_index("date")["fng"].sort_index()
        out["fng"] = ((fser.reindex(idx, method="ffill", limit=5) - 50.0) / 50.0).values
    if btc_ret is not None and len(btc_ret):
        br = pd.Series(btc_ret).copy()
        br.index = pd.DatetimeIndex(br.index)
        out["btc_lead_lag"] = br.reindex(idx).clip(-_CLIP, _CLIP).values

    # Reindex each per-source block onto the price index (small ffill bridges occasional gaps —
    # causal: row D carries the most recent value finalised on/before D), then write into `out`.
    for blk in blocks:
        aligned = blk.reindex(idx, method="ffill", limit=5)
        for col in aligned.columns:
            out[col] = aligned[col].values

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

        brows = await get_history("BTC")
        b = pd.DataFrame(brows)
        b.index = pd.to_datetime(b["date"])
        btc_ret = np.log(b["adj_close"].astype(float)).diff()

        for sym in ("BTC", "ETH", "SOL", "AAPL"):
            rows = await get_history(sym)
            if not rows:
                print(f"  {sym:5} no history")
                continue
            df = pd.DataFrame(rows)
            d = calibrate_fd_order(pd.Series(df["adj_close"].astype(float).values[: len(df) // 2]))
            feat = build_features_from_df(df, fd_order=d)
            cf = compute_crypto_features(sym, feat, btc_ret=btc_ret)
            nz = (cf.abs() > 1e-9).any()
            active = [c for c in CRYPTO_FEATURE_COLS if nz.get(c, False)]
            assert cf.index.equals(feat.index), "index misalignment"
            assert np.isfinite(cf.values).all(), "non-finite crypto feature"
            tail = cf.iloc[-1]
            print(
                f"  {sym:5} active={len(active):>2}/{len(CRYPTO_FEATURE_COLS)}  "
                f"fund_level={tail['fund_level']:+.5f} oi_z={tail['oi_z']:+.2f} "
                f"dvol={tail['dvol_level']:.3f} nvt={tail['nvt']:+.2f} "
                f"fng={tail['fng']:+.2f} btc_ll={tail['btc_lead_lag']:+.4f}"
            )
        print(
            "OK — equities return an all-zero (neutral) block; crypto rows are populated & finite."
        )

    asyncio.run(_demo())
