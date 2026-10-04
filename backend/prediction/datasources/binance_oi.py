"""
Datasource — Binance USDⓈ-M perpetual OPEN INTEREST  →  tidy daily (symbol, date, oi…).

Source files: ``Dataset/binance/open_interest/oi_<SYM>.csv``  (``scripts/dl_binance.py --oi``)
    columns: create_time, symbol, sum_open_interest, sum_open_interest_value,
             count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio,
             count_long_short_ratio, sum_taker_long_short_vol_ratio.

The bucket ships intraday (5-min) snapshots, so we collapse each UTC day to its END-OF-DAY state:
    • ``oi``        — last sum_open_interest of the day (contracts)
    • ``oi_value``  — last sum_open_interest_value of the day (USD notional)
    • ``ls_ratio``  — day-mean taker long/short volume ratio (positioning tilt)
    • ``top_ls``    — day-mean top-trader long/short account ratio
End-of-day values are known at 23:55 UTC of day D → no leak into a D+1 prediction.
ΔOI, OI/vol, OI-z are derived later in crypto_features.py.

    load_binance_oi()         -> DataFrame[symbol, date, oi, oi_value, ls_ratio, top_ls]
"""

from __future__ import annotations

import pandas as pd

from . import CRYPTO_SYMBOLS, DATASET_DIR, _to_day, tidy

_DIR = DATASET_DIR / "binance" / "open_interest"
_LAST = {"sum_open_interest": "oi", "sum_open_interest_value": "oi_value"}
_MEAN = {"sum_taker_long_short_vol_ratio": "ls_ratio", "sum_toptrader_long_short_ratio": "top_ls"}


def _load_one(sym: str) -> pd.DataFrame:
    fp = _DIR / f"oi_{sym}.csv"
    if not fp.exists():
        return pd.DataFrame()
    raw = pd.read_csv(fp)
    if raw.empty or "create_time" not in raw:
        return pd.DataFrame()
    raw["date"] = _to_day(raw["create_time"])
    for c in (*_LAST, *_MEAN):
        if c in raw:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values("create_time")
    g = raw.groupby("date")
    out = g[list(_LAST)].last().rename(columns=_LAST)
    out = out.join(g[[c for c in _MEAN if c in raw]].mean().rename(columns=_MEAN))
    out = out.reset_index()
    out.insert(0, "symbol", sym)
    return out


def load_binance_oi(symbols: list[str] | None = None) -> pd.DataFrame:
    """Daily open interest per symbol. Missing files skipped (OI is an opt-in download)."""
    syms = symbols or CRYPTO_SYMBOLS
    frames = [d for d in (_load_one(s.upper()) for s in syms) if not d.empty]
    cols = ["symbol", "date", "oi", "oi_value", "ls_ratio", "top_ls"]
    if not frames:
        return pd.DataFrame(columns=cols)
    return tidy(pd.concat(frames, ignore_index=True))[cols]


if __name__ == "__main__":
    df = load_binance_oi()
    if df.empty:
        print("open_interest: no files yet — run `python scripts/dl_binance.py --oi` (slow).")
    else:
        assert df.duplicated(["symbol", "date"]).sum() == 0, "duplicate (symbol,date) rows"
        print(
            f"open interest: {len(df):,} rows over {df.symbol.nunique()} symbols "
            f"({df.date.min().date()}..{df.date.max().date()})"
        )
        print(df[df.symbol == "BTC"].tail(3).to_string(index=False))
        print("OK")
