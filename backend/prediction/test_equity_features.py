"""
Leakage & sanity tests for the Phase-4 equity fundamentals block (equity_features.py).

Run directly:    python -m backend.prediction.test_equity_features
Or with pytest:  pytest backend/prediction/test_equity_features.py -q

The decisive test is `test_filing_causality`: the filing-level features at row t MUST NOT change when
future filings are removed, and the assembled daily block at date t MUST NOT change when future price
rows are removed. If either moves, a rolling/shift/trailing window is peeking at the future. The
integration tests then confirm the assembled block is finite, index-aligned, populated on equities,
and correctly NEUTRAL (all-zero) for crypto / unknown symbols.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.prediction import equity_features as ef
from backend.prediction.equity_features import (
    EQUITY_FEATURE_COLS,
    _fund_panel,
    _quarterly_features,
    compute_equity_features,
)
from backend.prediction.features import build_features_from_df, calibrate_fd_order


# ── 1. Causality: truncating future filings / price rows can't move earlier values ──
def _check_filing_causality(price_aapl) -> None:
    panel = _fund_panel()
    raw = panel[panel["symbol"] == "AAPL"].sort_values("date").reset_index(drop=True)
    assert len(raw) > 12, "need enough AAPL filings to test"
    cut = int(len(raw) * 0.6)

    full = _quarterly_features(raw).set_index("date")
    trunc = _quarterly_features(raw.iloc[:cut]).set_index("date")
    common = full.index.intersection(trunc.index)
    a = full.loc[common].to_numpy(dtype=float)
    b = trunc.loc[common].to_numpy(dtype=float)
    d = np.nanmax(np.abs(np.nan_to_num(a) - np.nan_to_num(b))) if a.size else 0.0
    assert d < 1e-9, f"filing-level feature changed when future filings removed (d={d:.2e}) -> LEAK"

    # Daily block: truncate the future price tail, earlier daily rows must be identical.
    pcut = int(len(price_aapl) * 0.6)
    full_d = compute_equity_features("AAPL", price_aapl)
    trunc_d = compute_equity_features("AAPL", price_aapl.iloc[:pcut])
    idx = trunc_d.index
    dd = np.nanmax(np.abs(full_d.loc[idx].to_numpy() - trunc_d.to_numpy()))
    assert dd < 1e-9, f"daily block changed when future price removed (d={dd:.2e}) -> LEAK"
    print(
        f"  [1] causality              OK   (filing d={d:.1e}, daily d={dd:.1e} -> invariant to truncation)"
    )


# ── 2. Assembled block: finite, index-aligned, populated on an equity ─────────────
def _check_equity_block(price_aapl) -> None:
    eb = compute_equity_features("AAPL", price_aapl)
    assert list(eb.columns) == EQUITY_FEATURE_COLS, "column set/order drift"
    assert eb.index.equals(price_aapl.index), "index not aligned to price frame"
    assert np.isfinite(eb.to_numpy()).all(), "non-finite value in equity block"
    active = int((eb.abs() > 1e-9).any().sum())
    assert active >= 4, f"AAPL should populate most columns, got {active}"
    print(
        f"  [2] equity-block-finite    OK   (AAPL {active}/{len(EQUITY_FEATURE_COLS)} cols active, all finite)"
    )


# ── 3. Crypto / unknown symbols are NEUTRAL (all-zero) — so dropna() spares their rows ──
def _check_crypto_neutral(price_btc) -> None:
    cb = compute_equity_features("BTC", price_btc)
    assert cb.index.equals(price_btc.index), "crypto index not aligned"
    assert (cb.to_numpy() == 0.0).all(), "crypto equity block must be all-zero (neutral)"
    unknown = compute_equity_features("NOTASYMBOL", price_btc)
    assert (unknown.to_numpy() == 0.0).all(), "unknown symbol must be neutral"
    print("  [3] crypto-neutral         OK   (BTC & unknown symbol -> all-zero block)")


# ── 4. Point-in-time: a feature only turns on at/after its filing date (no pre-dating) ──
def _check_point_in_time(price_aapl) -> None:
    eb = compute_equity_features("AAPL", price_aapl)
    qf = ef._symbol_quarterly("AAPL")
    first_filed = pd.to_datetime(qf["date"]).min()
    idx = pd.DatetimeIndex(pd.to_datetime(eb.index))
    before = eb.loc[idx < first_filed]
    # step/level features must be exactly neutral before the first filing is public.
    cols = ["earnings_surprise", "rev_revision", "accruals", "earnings_drift"]
    assert (before[cols].to_numpy() == 0.0).all(), (
        "fundamentals leaked before the first filing date"
    )
    print(
        f"  [4] point-in-time          OK   (all features neutral before first filing {first_filed.date()})"
    )


# ── 5. Single-quarter reconstruction + YoY: synthetic filings with known cumulative figures ──
def _check_quarterly_reconstruction() -> None:
    """The SEC income statement is YTD-cumulative within a fiscal year; the block reconstructs the
    single-quarter figure by within-`fy` differencing (Q1=itself, Q2=H1−Q1, …) before computing the
    YoY surprise/revision. Feed three fiscal years of KNOWN cumulative EPS/revenue and assert the
    derived features carry the right sign and magnitude — a regression guard on the differencing,
    fiscal-year grouping, and same-quarter YoY shift that the integration tests only cover indirectly.
    No DB / network: calls the pure filing-level core ``_quarterly_features``."""
    # Cumulative (YTD) EPS per filing → single-quarter EPS after differencing within each fy:
    #   2021: cum [1.0,2.0,3.0,4.0]  → q [1.0,1.0,1.0,1.0]
    #   2022: cum [1.2,2.6,4.2,6.0]  → q [1.2,1.4,1.6,1.8]   (YoY surprise vs 2021 all > 0)
    #   2023: cum [1.5,3.2,5.1,7.2]  → q [1.5,1.7,1.9,2.1]
    # Cumulative revenue → single-quarter revenue (clean +20% then +25% YoY growth):
    #   2021 q=10 each ; 2022 q=12 each (+20%) ; 2023 q=15 each (+25%)
    eps_cum = {2021: [1.0, 2.0, 3.0, 4.0], 2022: [1.2, 2.6, 4.2, 6.0], 2023: [1.5, 3.2, 5.1, 7.2]}
    rev_cum = {2021: [10, 20, 30, 40], 2022: [12, 24, 36, 48], 2023: [15, 30, 45, 60]}
    fps = ["Q1", "Q2", "Q3", "FY"]
    rows = []
    for fy in (2021, 2022, 2023):
        for i, fp in enumerate(fps):
            period = pd.Timestamp(year=fy, month=3 * (i + 1), day=28)
            rows.append(
                {
                    "symbol": "TEST",
                    "date": period + pd.Timedelta(days=40),
                    "period": period,
                    "fy": fy,
                    "fp": fp,
                    "form": "10-Q" if fp != "FY" else "10-K",
                    "revenue": rev_cum[fy][i] * 1e6,
                    "eps_diluted": eps_cum[fy][i],
                    "net_income": eps_cum[fy][i] * 1e6,
                    "op_income": eps_cum[fy][i] * 1e6,
                    # constant balance sheet → accruals ≈ 0 (no NOA change), just must not error/NaN-explode
                    "assets": 1000e6,
                    "liabilities": 400e6,
                    "equity": 600e6,
                    "cash": 100e6,
                }
            )
    qf = _quarterly_features(pd.DataFrame(rows))  # date-sorted; row i == build order i (filings are
    #   strictly chronological, incl. the FY filing that is *filed* the next calendar year). Select by
    #   filing position, NOT index.year — a Q4/FY filing lands in the next year and would mix fiscal years.
    rr = qf["rev_revision"].to_numpy()
    assert np.allclose(rr[4:8], 0.20, atol=1e-9), f"FY2022 YoY revenue revision != +20% ({rr[4:8]})"
    assert np.allclose(rr[8:12], 0.25, atol=1e-9), (
        f"FY2023 YoY revenue revision != +25% ({rr[8:12]})"
    )
    # First fiscal year has no prior-year base → revision is NaN here (neutral-0-filled at the daily layer).
    assert np.isnan(rr[0:4]).all(), "first fiscal year revision must be undefined (no YoY base)"

    # earnings_surprise (SUE): EPS grows every year → FY2023 surprises strictly positive once the
    # vol-scale window (≥4 filings) fills. Sign is the load-bearing property for the drift signal.
    sue_2023 = qf["sue"].to_numpy()[8:12]
    sue_2023 = sue_2023[~np.isnan(sue_2023)]
    assert len(sue_2023) >= 1 and (sue_2023 > 0).all(), (
        f"growing EPS must yield positive SUE ({sue_2023})"
    )

    # ttm_eps: trailing-4-quarter sum = the fiscal-year total once 4 quarters are in the window.
    assert np.isclose(qf["ttm_eps"].to_numpy()[7], 6.0, atol=1e-6), (
        f"TTM EPS at FY2022 filing != annual 6.0 ({qf['ttm_eps'].to_numpy()[7]})"
    )
    print(
        "  [5] quarterly-reconstruction OK   (YTD->single-quarter diff, YoY +20%/+25%, SUE>0, TTM=FY)"
    )


async def _setup():
    from backend.db import get_history, init_db

    await init_db()

    async def _price(sym):
        rows = await get_history(sym)
        if not rows:
            pytest.skip(
                f"no seeded market data for {sym} (flux_market.db not populated in this environment)"
            )
        df = pd.DataFrame(rows)
        d = calibrate_fd_order(pd.Series(df["adj_close"].astype(float).values[: len(df) // 2]))
        return build_features_from_df(df, fd_order=d)

    return await _price("AAPL"), await _price("BTC")


def _run() -> None:
    price_aapl, price_btc = asyncio.run(_setup())
    print("Running leakage & sanity tests for equity_features:")
    _check_filing_causality(price_aapl)
    _check_equity_block(price_aapl)
    _check_crypto_neutral(price_btc)
    _check_point_in_time(price_aapl)
    _check_quarterly_reconstruction()
    print("All tests passed.")


# pytest entry points (set up the frames once)
_FRAMES = None


def _frames():
    global _FRAMES
    if _FRAMES is None:
        _FRAMES = asyncio.run(_setup())
    return _FRAMES


def test_filing_causality():
    p_aapl, _ = _frames()
    _check_filing_causality(p_aapl)


def test_equity_block():
    p_aapl, _ = _frames()
    _check_equity_block(p_aapl)


def test_crypto_neutral():
    _, p_btc = _frames()
    _check_crypto_neutral(p_btc)


def test_point_in_time():
    p_aapl, _ = _frames()
    _check_point_in_time(p_aapl)


def test_quarterly_reconstruction():
    _check_quarterly_reconstruction()


if __name__ == "__main__":
    _run()
