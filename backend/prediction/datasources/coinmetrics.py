"""
Datasource — Coin Metrics community ON-CHAIN metrics  →  tidy daily (symbol, date, …).

Source file: ``Dataset/coinmetrics/onchain.csv``  (``scripts/dl_coinmetrics.py``) — long format:
    asset (lowercase), time (iso), AdrActCnt, TxCnt, SplyCur, CapMrktCurUSD, PriceUSD.

Renamed to stable internal columns and keyed to DB symbols:
    • ``adr_act``      — active addresses              • ``cap_mkt_usd`` — market cap (USD)
    • ``tx_cnt``       — transaction count             • ``price_cm``    — Coin Metrics ref price (USD)
    • ``sply_cur``     — current circulating supply
Daily totals are finalised at UTC midnight of day D. NVT / addr-growth / tx-growth are derived in
crypto_features.py. The community tier covers 11 of our 14 perps (AVAX/SOL/SHIB are absent and are
simply missing — the feature layer fills them neutral).

Genesis-era rows where the chain had ~no activity (AdrActCnt==0 & TxCnt==0) are dropped so a
forward-fill can't carry zeros into the training window.

    load_coinmetrics()        -> DataFrame[symbol, date, adr_act, tx_cnt, sply_cur, cap_mkt_usd, price_cm]
"""

from __future__ import annotations

import pandas as pd

from . import CRYPTO_SYMBOLS, DATASET_DIR, _to_day, tidy

_FP = DATASET_DIR / "coinmetrics" / "onchain.csv"
_RENAME = {
    "AdrActCnt": "adr_act",
    "TxCnt": "tx_cnt",
    "SplyCur": "sply_cur",
    "CapMrktCurUSD": "cap_mkt_usd",
    "PriceUSD": "price_cm",
}


def load_coinmetrics(symbols: list[str] | None = None) -> pd.DataFrame:
    """Daily on-chain metrics per symbol. Empty frame if the file is missing."""
    want = {s.upper() for s in (symbols or CRYPTO_SYMBOLS)}
    cols = ["symbol", "date", *_RENAME.values()]
    if not _FP.exists():
        return pd.DataFrame(columns=cols)
    raw = pd.read_csv(_FP)
    if raw.empty or "asset" not in raw:
        return pd.DataFrame(columns=cols)
    raw["symbol"] = raw["asset"].str.upper()
    raw = raw[raw["symbol"].isin(want)].copy()
    raw["date"] = _to_day(raw["time"])
    raw = raw.rename(columns=_RENAME)
    for c in _RENAME.values():
        raw[c] = pd.to_numeric(raw.get(c), errors="coerce")
    # Drop genesis-era inactivity so ffill never propagates structural zeros.
    raw = raw[~((raw["adr_act"].fillna(0) == 0) & (raw["tx_cnt"].fillna(0) == 0))]
    return tidy(raw[cols])


if __name__ == "__main__":
    df = load_coinmetrics()
    assert not df.empty, "no Coin Metrics data — run scripts/dl_coinmetrics.py first"
    assert df.duplicated(["symbol", "date"]).sum() == 0
    print(
        f"coinmetrics: {len(df):,} rows over {df.symbol.nunique()} symbols "
        f"({df.date.min().date()}..{df.date.max().date()})"
    )
    print("symbols:", sorted(df.symbol.unique()))
    print(df[df.symbol == "BTC"].tail(3).to_string(index=False))
    print("OK")
