"""
Download Deribit DVOL (implied-volatility index) history — keyless, BTC & ETH only.
Writes Dataset/deribit/dvol_<CCY>.csv with columns: date, ts, open, high, low, close.

    python scripts/dl_deribit.py                # ~6 years, daily
    python scripts/dl_deribit.py --years 8 --resolution 1D

`close` is the DVOL value (annualized implied vol, %). This is the crypto analog of VIX and,
unlike equity option chains, it is fully backfillable → can enter leak-free training.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
OUT = Path(__file__).resolve().parents[1] / "Dataset" / "deribit"


def fetch(ccy: str, years: int, resolution: str) -> pd.DataFrame:
    # Deribit returns at most ~1000 points and gives the MOST RECENT window in [start,end],
    # so we paginate BACKWARD: keep `start` fixed and walk `end` back to each batch's earliest ts.
    now = int(time.time() * 1000)
    start = now - int(years * 365.25 * 24 * 3600 * 1000)
    rows, end = [], now
    while end > start:
        p = {
            "currency": ccy,
            "start_timestamp": start,
            "end_timestamp": end,
            "resolution": resolution,
        }
        r = requests.get(URL, params=p, timeout=30).json()
        data = r.get("result", {}).get("data", [])
        if not data:
            break
        rows += data
        earliest = min(row[0] for row in data)
        if earliest <= start or earliest >= end:  # reached inception / no progress
            break
        end = earliest - 1
        time.sleep(0.2)
    df = (
        pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"])
        .drop_duplicates("ts")
        .sort_values("ts")
    )
    df.insert(0, "date", pd.to_datetime(df["ts"], unit="ms").dt.strftime("%Y-%m-%d"))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=6)
    ap.add_argument("--resolution", default="1D", help="1 | 60 | 1D")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Deribit DVOL -> {OUT}  (years {a.years}, resolution {a.resolution})")
    for ccy in ("BTC", "ETH"):
        df = fetch(ccy, a.years, a.resolution)
        fp = OUT / f"dvol_{ccy}.csv"
        df.to_csv(fp, index=False)
        span = f"{df['date'].iloc[0]}..{df['date'].iloc[-1]}" if len(df) else "empty"
        print(f"  {ccy}: {len(df):>5} rows  {span}")
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())
