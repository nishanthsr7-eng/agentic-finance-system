"""
Datasource — Crypto Fear & Greed Index  →  tidy daily (date, fng).  MARKET-WIDE (no symbol).

Source file: ``Dataset/fng.json``  (alternative.me schema):
    {"data": [{"value": "12", "value_classification": "Extreme Fear", "timestamp": "1780704000"}, …]}

This is a single market-level sentiment/regime gauge (0 = extreme fear, 100 = extreme greed), so it
has no per-symbol dimension — the crypto feature layer broadcasts it across all crypto symbols. The
index value for day D is published during day D, so a row dated D is causal for a D+1 prediction.

    • ``fng``        — index value 0–100
    • ``fng_class``  — categorical label (Extreme Fear … Extreme Greed)

    load_fng()                -> DataFrame[date, fng, fng_class]
"""

from __future__ import annotations

import json

import pandas as pd

from . import DATASET_DIR, tidy

_FP = DATASET_DIR / "fng.json"


def load_fng() -> pd.DataFrame:
    """Daily Fear & Greed index. Empty frame if the file is missing."""
    cols = ["date", "fng", "fng_class"]
    if not _FP.exists():
        return pd.DataFrame(columns=cols)
    data = json.loads(_FP.read_text()).get("data", [])
    if not data:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(data)
    df["date"] = (
        pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="s", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    df["fng"] = pd.to_numeric(df["value"], errors="coerce")
    df["fng_class"] = df.get("value_classification")
    return tidy(df[cols], has_symbol=False)


if __name__ == "__main__":
    df = load_fng()
    assert not df.empty, "no fng.json found in Dataset/"
    assert df["fng"].between(0, 100).all(), "F&G out of 0..100 range"
    assert df.duplicated(["date"]).sum() == 0
    print(f"Fear & Greed: {len(df):,} daily rows ({df.date.min().date()}..{df.date.max().date()})")
    print(df.tail(3).to_string(index=False))
    print("OK")
