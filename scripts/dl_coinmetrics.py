"""
Download Coin Metrics COMMUNITY on-chain metrics — keyless, multi-asset, historical.
(The free, no-key, multi-coin replacement for Glassnode.)
Writes Dataset/coinmetrics/onchain.csv (long format: time, asset, <metrics>).

    python scripts/dl_coinmetrics.py
    python scripts/dl_coinmetrics.py --metrics AdrActCnt,TxCnt,FeeTotUSD

Metrics: AdrActCnt (active addresses), TxCnt (tx count), FeeTotUSD (fees), SplyCur (supply).
Catalog: https://docs.coinmetrics.io/api/v4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
OUT = Path(__file__).resolve().parents[1] / "Dataset" / "coinmetrics"
# our crypto universe as Coin Metrics asset ids (lowercase); some altcoins may be unsupported -> skipped
ASSETS = [
    "btc",
    "eth",
    "bnb",
    "sol",
    "xrp",
    "ada",
    "avax",
    "dot",
    "link",
    "ltc",
    "trx",
    "uni",
    "doge",
    "shib",
]


def _fetch(asset: str, metrics: list[str]) -> list[dict] | None:
    """Paginated fetch for ONE asset. Returns None if the combo is forbidden (so caller retries)."""
    rows, token = [], None
    while True:
        params = {
            "assets": asset,
            "metrics": ",".join(metrics),
            "frequency": "1d",
            "page_size": 10000,
        }
        if token:
            params["next_page_token"] = token
        r = requests.get(BASE, params=params, timeout=60).json()
        if r.get("error"):
            return None
        rows += r.get("data", [])
        token = r.get("next_page_token")
        if not token:
            break
        time.sleep(0.3)
    return rows


def fetch(assets: list[str], metrics: list[str]) -> pd.DataFrame:
    # Batch all metrics per asset; if that asset lacks one (→ forbidden), retry it metric-by-metric.
    acc: dict[tuple, dict] = {}
    for a in assets:
        rows = _fetch(a, metrics)
        if rows is None:
            rows = []
            for m in metrics:
                rws = _fetch(a, [m])
                if rws:
                    rows += rws
        if not rows:
            print(f"  {a:5}: no community data")
            continue
        got = set()
        for r in rows:
            key = (r["asset"], r["time"])
            d = acc.setdefault(key, {"asset": r["asset"], "time": r["time"]})
            for k, v in r.items():
                if k not in ("asset", "time"):
                    d[k] = v
                    got.add(k)
        print(f"  {a:5}: {len(rows):>5} rows, metrics {sorted(got)}")
    return pd.DataFrame(list(acc.values()))


def main():
    ap = argparse.ArgumentParser()
    # Broad list; metrics not on the free community tier are auto-skipped per-asset/metric.
    ap.add_argument("--metrics", default="AdrActCnt,TxCnt,SplyCur,CapMrktCurUSD,PriceUSD,FeeTotUSD")
    ap.add_argument("--assets", default=",".join(ASSETS))
    a = ap.parse_args()
    assets = [s.strip() for s in a.assets.split(",") if s.strip()]
    metrics = [s.strip() for s in a.metrics.split(",") if s.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Coin Metrics community -> {OUT}  ({len(assets)} assets, {len(metrics)} metrics)")
    df = fetch(assets, metrics)
    if df.empty:
        print("  no data returned")
        return 0
    fp = OUT / "onchain.csv"
    df.to_csv(fp, index=False)
    got = sorted(df["asset"].unique()) if "asset" in df else []
    missing = [x for x in assets if x not in got]
    print(f"  {len(df)} rows, assets covered: {got}")
    if missing:
        print(f"  not available on community tier: {missing}")
    print(f"DONE -> {fp}")


if __name__ == "__main__":
    sys.exit(main())
