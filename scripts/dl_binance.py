"""
Download Binance USDⓈ-M futures FUNDING RATE + 1d KLINES (keyless, public bucket).
Writes one combined CSV per symbol into Dataset/binance/.

    python scripts/dl_binance.py                 # funding + klines, all symbols, from 2019
    python scripts/dl_binance.py --start 2021    # narrower history
    python scripts/dl_binance.py --oi            # ALSO pull open interest (heavy: daily files)

Funding rate is the primary crypto signal; klines add perp OHLCV/volume/taker-flow.
Open interest history only exists as per-DAY files on the bucket, so it's opt-in (slow).
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

BASE = "https://data.binance.vision/data"
OUT = Path(__file__).resolve().parents[1] / "Dataset" / "binance"

# our crypto universe -> Binance USDⓈ-M perp pair (SHIB trades as 1000SHIB on futures)
PAIRS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "BNB": "BNBUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
    "ADA": "ADAUSDT",
    "AVAX": "AVAXUSDT",
    "DOT": "DOTUSDT",
    "LINK": "LINKUSDT",
    "LTC": "LTCUSDT",
    "TRX": "TRXUSDT",
    "UNI": "UNIUSDT",
    "DOGE": "DOGEUSDT",
    "SHIB": "1000SHIBUSDT",
}
KLINE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]
FUNDING_COLS = ["calc_time", "funding_interval_hours", "last_funding_rate"]
SESSION = requests.Session()


def months(start_year: int):
    s = dt.date(start_year, 1, 1)
    e = dt.date.today()
    while s <= e:
        yield s.strftime("%Y-%m")
        s = (s.replace(day=28) + dt.timedelta(days=4)).replace(day=1)


def days(start_year: int):
    s = dt.date(start_year, 1, 1)
    e = dt.date.today()
    while s <= e:
        yield s.strftime("%Y-%m-%d")
        s += dt.timedelta(days=1)


def _read_zip_csv(raw: bytes, names: list[str]) -> pd.DataFrame | None:
    """Read the single CSV inside a Binance .zip; drop a header row if present; assign `names`."""
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        inner = z.read(z.namelist()[0])
    if not inner.strip():
        return None
    df = pd.read_csv(io.BytesIO(inner), header=None)
    try:
        float(str(df.iloc[0, 0]))  # numeric → no header row
    except ValueError:
        df = df.iloc[1:].reset_index(drop=True)  # drop header row
    if df.shape[1] == len(names):
        df.columns = names
    return df


def _grab(url: str, names: list[str]) -> pd.DataFrame | None:
    r = SESSION.get(url, timeout=60)
    if r.status_code != 200:
        return None  # month/day not published → skip
    try:
        return _read_zip_csv(r.content, names)
    except Exception as e:
        print(f"    warn: {url.split('/')[-1]}: {e}")
        return None


def pull_series(sym: str, pair: str, kind: str, start_year: int) -> int:
    if kind == "funding":
        url = lambda ym: f"{BASE}/futures/um/monthly/fundingRate/{pair}/{pair}-fundingRate-{ym}.zip"
        names, iterator = FUNDING_COLS, months(start_year)
    else:  # klines
        url = lambda ym: f"{BASE}/futures/um/monthly/klines/{pair}/1d/{pair}-1d-{ym}.zip"
        names, iterator = KLINE_COLS, months(start_year)
    frames = [df for df in (_grab(url(ym), names) for ym in iterator) if df is not None]
    if not frames:
        print(f"  {sym:5} {kind:8}: no data")
        return 0
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    d = OUT / kind
    d.mkdir(parents=True, exist_ok=True)
    fp = d / f"{kind}_{sym}.csv"
    out.to_csv(fp, index=False)
    print(f"  {sym:5} {kind:8}: {len(out):>6} rows -> {fp.relative_to(OUT.parent)}")
    return len(out)


def pull_oi(sym: str, pair: str, start_year: int) -> int:
    """Open interest from per-day metrics files (heavy)."""
    OI = [
        "create_time",
        "symbol",
        "sum_open_interest",
        "sum_open_interest_value",
        "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio",
        "count_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    ]
    url = lambda d: f"{BASE}/futures/um/daily/metrics/{pair}/{pair}-metrics-{d}.zip"
    frames = [df for df in (_grab(url(d), OI) for d in days(start_year)) if df is not None]
    if not frames:
        print(f"  {sym:5} oi      : no data")
        return 0
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    d = OUT / "open_interest"
    d.mkdir(parents=True, exist_ok=True)
    out.to_csv(d / f"oi_{sym}.csv", index=False)
    print(f"  {sym:5} oi      : {len(out):>6} rows")
    return len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2019)
    ap.add_argument("--oi", action="store_true", help="also pull open interest (slow: daily files)")
    ap.add_argument("--symbols", nargs="*", help="subset of symbols (default: all)")
    a = ap.parse_args()
    syms = a.symbols or list(PAIRS)
    print(f"Binance download -> {OUT}  (start {a.start}, {len(syms)} symbols, oi={a.oi})")
    tot = 0
    for sym in syms:
        pair = PAIRS.get(sym.upper())
        if not pair:
            print(f"  {sym}: not a known crypto pair, skip")
            continue
        tot += pull_series(sym.upper(), pair, "funding", a.start)
        tot += pull_series(sym.upper(), pair, "klines", a.start)
        if a.oi:
            tot += pull_oi(sym.upper(), pair, max(a.start, 2021))
    print(f"DONE — {tot} total rows under {OUT}")


if __name__ == "__main__":
    sys.exit(main())
