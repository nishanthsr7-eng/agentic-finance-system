"""
Datasource — blockchain.com BTC ON-CHAIN series  →  tidy daily (symbol=BTC, date, …).

Source files (Dataset root, headerless ``datetime,value``):
    hash-rate.csv · n-transactions.csv · n-unique-addresses.csv · transaction-fees.csv

These complement Coin Metrics with BTC-native chain health. The raw series are sampled
irregularly (every few days), so we align them on a common daily index and FORWARD-FILL — which
is causal: row D carries the most recent value published on/before D, never a future one.

    • ``hashrate``       — network hash rate            • ``n_unique_addr`` — unique active addresses
    • ``n_tx``           — daily transaction count      • ``tx_fees``       — total transaction fees (BTC)

    load_blockchain_onchain() -> DataFrame[symbol, date, hashrate, n_tx, n_unique_addr, tx_fees]
"""

from __future__ import annotations

import pandas as pd

from . import DATASET_DIR, tidy

_FILES = {
    "hash-rate.csv": "hashrate",
    "n-transactions.csv": "n_tx",
    "n-unique-addresses.csv": "n_unique_addr",
    "transaction-fees.csv": "tx_fees",
}


def _read(fname: str, col: str) -> pd.Series:
    fp = DATASET_DIR / fname
    if not fp.exists():
        return pd.Series(dtype=float, name=col)
    raw = pd.read_csv(fp, header=None, names=["date", col])
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    raw[col] = pd.to_numeric(raw[col], errors="coerce")
    return raw.dropna(subset=["date"]).drop_duplicates("date", keep="last").set_index("date")[col]


def load_blockchain_onchain() -> pd.DataFrame:
    """BTC chain-health metrics on a daily ffilled index. Empty if no source files present."""
    cols = ["symbol", "date", *_FILES.values()]
    series = {col: _read(f, col) for f, col in _FILES.items()}
    series = {k: v for k, v in series.items() if not v.empty}
    if not series:
        return pd.DataFrame(columns=cols)
    wide = pd.DataFrame(series).sort_index()
    daily = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    wide = wide.reindex(daily).ffill().dropna(how="all")
    out = wide.reset_index(names="date")
    out.insert(0, "symbol", "BTC")
    return tidy(out[[c for c in cols if c in out.columns]])


if __name__ == "__main__":
    df = load_blockchain_onchain()
    assert not df.empty, "no blockchain.com CSVs found in Dataset/"
    assert (df["symbol"] == "BTC").all()
    assert df.duplicated(["symbol", "date"]).sum() == 0
    print(
        f"blockchain.com BTC: {len(df):,} daily rows "
        f"({df.date.min().date()}..{df.date.max().date()})"
    )
    print(df.tail(3).to_string(index=False))
    print("OK")
