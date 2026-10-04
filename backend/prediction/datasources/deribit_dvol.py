"""
Datasource — Deribit DVOL (crypto implied-vol index, the BTC/ETH analog of VIX).

Source files: ``Dataset/deribit/dvol_<CCY>.csv``  (``scripts/dl_deribit.py``)
    columns: date, ts (ms), open, high, low, close.  Already daily.

DVOL is the annualised implied vol (%). Unlike equity option chains it is fully backfillable, so
it can enter leak-free training. Only BTC and ETH are published.

    • ``dvol``       — daily close (the headline implied-vol level)
    • ``dvol_open``  — daily open (kept so dvol_chg = close/open can be derived intraday-causally)
Per-symbol level/term/Δ features are built in crypto_features.py.

    load_deribit_dvol()       -> DataFrame[symbol, date, dvol, dvol_open]
"""

from __future__ import annotations

import pandas as pd

from . import DATASET_DIR, _to_day, tidy

_DIR = DATASET_DIR / "deribit"
_CCY = {"BTC": "BTC", "ETH": "ETH"}  # file currency -> DB symbol (identity here)


def _load_one(ccy: str, sym: str) -> pd.DataFrame:
    fp = _DIR / f"dvol_{ccy}.csv"
    if not fp.exists():
        return pd.DataFrame()
    raw = pd.read_csv(fp)
    if raw.empty or "close" not in raw:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "symbol": sym,
            "date": _to_day(raw["date"]),
            "dvol": pd.to_numeric(raw["close"], errors="coerce"),
            "dvol_open": pd.to_numeric(raw["open"], errors="coerce"),
        }
    )
    return out.dropna(subset=["dvol"])


def load_deribit_dvol(symbols: list[str] | None = None) -> pd.DataFrame:
    """Daily DVOL for BTC/ETH. Other symbols have no DVOL and are simply absent."""
    want = {s.upper() for s in symbols} if symbols else set(_CCY)
    frames = [_load_one(ccy, sym) for ccy, sym in _CCY.items() if sym in want]
    frames = [d for d in frames if not d.empty]
    if not frames:
        return pd.DataFrame(columns=["symbol", "date", "dvol", "dvol_open"])
    return tidy(pd.concat(frames, ignore_index=True))


if __name__ == "__main__":
    df = load_deribit_dvol()
    assert not df.empty, "no DVOL loaded — run scripts/dl_deribit.py first"
    assert set(df.symbol.unique()) <= {"BTC", "ETH"}
    assert df.duplicated(["symbol", "date"]).sum() == 0
    print(f"DVOL: {len(df):,} rows ({df.date.min().date()}..{df.date.max().date()})")
    print(df.groupby("symbol").size().to_dict())
    print(df[df.symbol == "BTC"].tail(3).to_string(index=False))
    print("OK")
