"""
FLUX Prediction — FRED Macro Features (Phase 3.3)
=================================================
Macro regime is one of the strongest *and* most overlooked free inputs. This pulls the key
series from FRED and turns them into causal daily features:

  • term_spread = 10Y − 2Y Treasury (DGS10 − DGS2) — the classic recession/regime signal
  • dgs10       — 10Y yield level (Δ already captured cross-asset via TNX; level adds context)
  • fed_funds   — effective Fed Funds rate (policy stance)
  • cpi_yoy     — year-over-year CPI inflation (CPIAUCSL, % change vs 12 months prior)

All series are forward-filled to a daily index and lagged by one day at use so row t only sees
data released on/before t (CPI is monthly + released with a lag, so ffill is naturally causal).

Activation: needs a free FRED_API_KEY in .env. WITHOUT a key this is a clean no-op — it returns
an empty frame and the training/serving macro block is unchanged. Add the key and retrain to
fold these columns in (the feature set is fixed at train time).

Security:
  • series ids are taken only from the FRED_SERIES whitelist (no arbitrary id into the URL).
  • key is read from settings, sent only to FRED over https, never logged or returned.

Public API:
    await build_fred_features(start) -> DataFrame[date]  ( empty if no key )
    compute_fred_features(raw)       -> DataFrame         ( pure transform, unit-tested )
"""

from __future__ import annotations

import logging

import httpx
import pandas as pd

from ..config import settings

log = logging.getLogger("flux.prediction.fred")

# Whitelist: FRED series id -> internal column name. Only these ids are ever requested.
FRED_SERIES = {
    "DGS10": "dgs10",  # 10-Year Treasury yield
    "DGS2": "dgs2",  # 2-Year Treasury yield
    "FEDFUNDS": "fed_funds",  # Effective Federal Funds rate
    "CPIAUCSL": "cpi",  # CPI (level → we derive YoY)
    "BAA10Y": "credit_spread",  # Moody's Baa corporate yield − 10Y Treasury = credit risk premium
}
_FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


async def _fetch_series(series_id: str, start: str) -> pd.Series:
    """One FRED series as a date-indexed float Series. Empty Series on failure / no key."""
    if series_id not in FRED_SERIES or not settings.FRED_API_KEY:
        return pd.Series(dtype=float)
    try:
        async with httpx.AsyncClient(timeout=20.0) as cli:
            r = await cli.get(
                _FRED_URL,
                params={
                    "series_id": series_id,
                    "api_key": settings.FRED_API_KEY,
                    "file_type": "json",
                    "observation_start": start,
                },
            )
            r.raise_for_status()
            obs = r.json().get("observations", [])
    except Exception as exc:
        log.warning("FRED fetch %s failed: %s", series_id, exc)
        return pd.Series(dtype=float)
    idx, vals = [], []
    for o in obs:
        v = o.get("value")
        if v in (None, ".", ""):  # FRED marks missing as "."
            continue
        try:
            vals.append(float(v))
            idx.append(pd.Timestamp(o["date"]))
        except (ValueError, KeyError):
            continue
    return pd.Series(vals, index=pd.DatetimeIndex(idx), name=FRED_SERIES[series_id])


def compute_fred_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Pure transform (no I/O, unit-tested): raw columns {dgs10,dgs2,fed_funds,cpi} indexed by date
    → daily causal features {term_spread, dgs10, fed_funds, cpi_yoy}, forward-filled.
    """
    if raw.empty:
        return pd.DataFrame()
    df = raw.sort_index().copy()
    daily = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(daily).ffill()
    out = pd.DataFrame(index=daily)
    if {"dgs10", "dgs2"} <= set(df.columns):
        out["term_spread"] = df["dgs10"] - df["dgs2"]
    if "credit_spread" in df.columns:
        out["credit_spread"] = df["credit_spread"]  # Baa − 10Y, already a spread (daily)
    if "dgs10" in df.columns:
        out["dgs10"] = df["dgs10"]
    if "fed_funds" in df.columns:
        out["fed_funds"] = df["fed_funds"]
    if "cpi" in df.columns:
        out["cpi_yoy"] = df["cpi"].pct_change(365) * 100.0  # YoY % (daily index → 365d)
    return out.dropna(how="all")


async def build_fred_features(start: str = "2000-01-01") -> pd.DataFrame:
    """Fetch all whitelisted series and build the causal macro feature frame (empty if no key)."""
    if not settings.FRED_API_KEY:
        return pd.DataFrame()
    series = {}
    for sid in FRED_SERIES:
        s = await _fetch_series(sid, start)
        if not s.empty:
            series[FRED_SERIES[sid]] = s
    if not series:
        return pd.DataFrame()
    raw = pd.DataFrame(series)
    return compute_fred_features(raw)


if __name__ == "__main__":
    import asyncio

    async def _demo():
        # Live fetch if a key is configured; otherwise unit-test the transform on synthetic data.
        live = await build_fred_features("2018-01-01")
        if not live.empty:
            print("Live FRED features (tail):")
            print(live.tail(3))
            return
        print("No FRED_API_KEY set -> verifying the transform on synthetic data:")
        idx = pd.to_datetime(["2020-01-01", "2020-07-01", "2021-01-01", "2021-07-01"])
        raw = pd.DataFrame(
            {
                "dgs10": [1.8, 0.7, 1.1, 1.5],
                "dgs2": [1.5, 0.2, 0.2, 0.3],
                "fed_funds": [1.6, 0.1, 0.1, 0.1],
                "cpi": [258.0, 259.0, 262.0, 271.0],
            },
            index=idx,
        )
        feat = compute_fred_features(raw)
        ts = feat["term_spread"].dropna()
        print(f"  term_spread sample: {ts.iloc[0]:.2f} (expect 0.30 on 2020-01-01)")
        print(f"  cpi_yoy on 2021-01-01: {feat.loc['2021-01-01', 'cpi_yoy']:.2f}% (expect ~1.55%)")
        print(f"  columns: {list(feat.columns)}  rows: {len(feat)}")
        assert abs(ts.iloc[0] - 0.30) < 1e-6, "term spread transform wrong"
        print("  transform OK")

    asyncio.run(_demo())
