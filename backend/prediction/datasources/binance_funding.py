"""
Datasource — Binance USDⓈ-M perpetual FUNDING RATE  →  tidy daily (symbol, date, funding).

Source files: ``Dataset/binance/funding/funding_<SYM>.csv``  (from ``scripts/dl_binance.py``)
    columns: calc_time (ms epoch), funding_interval_hours (=8), last_funding_rate.

Funding is charged three times a UTC day (00:00 / 08:00 / 16:00). We collapse each day to:
    • ``funding``      — the day's TOTAL funding (sum of the 3 settlements) = the actual daily carry
    • ``funding_last`` — the last settlement of the day (the level going into the next day)
Both are finalised by 16:00 UTC of day D, so a row dated D leaks nothing into a D+1 prediction.

Signal engineering (fund_z20, sign-flips, fund_cum8 …) is deliberately left to crypto_features.py.

    load_binance_funding()            -> DataFrame[symbol, date, funding, funding_last]
    load_binance_funding(["BTC"])     -> same, subset
"""

from __future__ import annotations

import pandas as pd

from . import CRYPTO_SYMBOLS, DATASET_DIR, _to_day, tidy

_DIR = DATASET_DIR / "binance" / "funding"


def _load_one(sym: str) -> pd.DataFrame:
    fp = _DIR / f"funding_{sym}.csv"
    if not fp.exists():
        return pd.DataFrame()
    raw = pd.read_csv(fp)
    if raw.empty or "calc_time" not in raw:
        return pd.DataFrame()
    raw["date"] = _to_day(raw["calc_time"])
    raw["last_funding_rate"] = pd.to_numeric(raw["last_funding_rate"], errors="coerce")
    g = raw.dropna(subset=["date", "last_funding_rate"]).groupby("date")["last_funding_rate"]
    out = pd.DataFrame({"funding": g.sum(), "funding_last": g.last()}).reset_index()
    out.insert(0, "symbol", sym)
    return out


def load_binance_funding(symbols: list[str] | None = None) -> pd.DataFrame:
    """Daily funding per symbol. Missing files are skipped (graceful)."""
    syms = symbols or CRYPTO_SYMBOLS
    frames = [d for d in (_load_one(s.upper()) for s in syms) if not d.empty]
    if not frames:
        return pd.DataFrame(columns=["symbol", "date", "funding", "funding_last"])
    return tidy(pd.concat(frames, ignore_index=True))


if __name__ == "__main__":
    df = load_binance_funding()
    assert not df.empty, "no funding loaded — run scripts/dl_binance.py first"
    assert list(df.columns) == ["symbol", "date", "funding", "funding_last"]
    assert df.duplicated(["symbol", "date"]).sum() == 0, "duplicate (symbol,date) rows"
    n = df.groupby("symbol").size()
    print(
        f"funding: {len(df):,} rows over {df.symbol.nunique()} symbols "
        f"({df.date.min().date()}..{df.date.max().date()})"
    )
    print(df[df.symbol == "BTC"].tail(3).to_string(index=False))
    print(f"rows/symbol: {n.min()}..{n.max()}  | OK")
