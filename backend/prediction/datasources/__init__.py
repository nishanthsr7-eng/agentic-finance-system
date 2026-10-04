"""
FLUX Prediction — Datasource Loaders (Phase 0)
==============================================
Thin, leak-safe readers that turn the files staged under ``Dataset/`` into **tidy daily frames
keyed by (symbol, date)** — the common shape every downstream feature block (crypto_features,
sec fundamentals, …) consumes. They do *raw alignment only*; signal engineering (z-scores,
NVT, growth rates, surprises) lives in the Phase-2+ feature modules so the loaders stay testable
and reusable.

Contract (honoured by every loader here):
  • PER-SYMBOL sources return a long DataFrame with columns ``[symbol, date, <values…>]`` where
    ``symbol`` matches the DB universe (BTC, ETH, AAPL, …) and ``date`` is a tz-naive UTC
    ``datetime64[ns]`` floored to the day.
  • MARKET-WIDE sources (Fear & Greed) have no symbol — they return ``[date, <values…>]`` and the
    feature layer broadcasts them across crypto symbols.
  • One row per (symbol, day); duplicates collapsed; sorted by (symbol, date).

Leakage: every value on row ``date=D`` is finalised by the end of UTC day D (end-of-day OI/DVOL,
the day's funding sum, on-chain daily totals). The training pipeline lags features by ≥1 day at
use, so a row for D is only ever read when predicting for D+1 onward. The SEC loader is the one
point-in-time exception and keys on the **filing date**, not the reporting period — see its docstring.

Public API per module:  ``load_<name>() -> pd.DataFrame``  (+ pure ``compute_*`` transforms where useful).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Dataset/ lives at the repo root:  datasources → prediction → backend → <root>/Dataset
DATASET_DIR = Path(__file__).resolve().parents[3] / "Dataset"

# DB symbol -> Coin Metrics asset id (lowercase). Also the crypto universe these loaders cover.
CRYPTO_SYMBOLS = [
    "BTC",
    "ETH",
    "BNB",
    "SOL",
    "XRP",
    "ADA",
    "AVAX",
    "DOT",
    "LINK",
    "LTC",
    "TRX",
    "UNI",
    "DOGE",
    "SHIB",
]


def _to_day(ts) -> pd.Series:
    """Coerce a ms-epoch / iso / datetime column to a tz-naive day-floored Timestamp.

    Numeric input is read as **milliseconds since epoch** (Binance calc_time/create_time);
    everything else is parsed as a datetime string (ISO, YYYY-MM-DD, …)."""
    ser = pd.Series(ts)
    if pd.api.types.is_numeric_dtype(ser):
        s = pd.to_datetime(ser, unit="ms", utc=True, errors="coerce")
    else:
        s = pd.to_datetime(ser, utc=True, errors="coerce")
    return s.dt.tz_localize(None).dt.normalize()


def tidy(df: pd.DataFrame, has_symbol: bool = True) -> pd.DataFrame:
    """Final shaping shared by every loader: drop empty/dup rows, sort by key, reset index."""
    if df.empty:
        return df
    key = ["symbol", "date"] if has_symbol else ["date"]
    df = df.dropna(subset=key).drop_duplicates(key, keep="last")
    return df.sort_values(key).reset_index(drop=True)
