"""
FLUX — One-off historical OHLCV backfill
-----------------------------------------
Pulls *years* of daily OHLCV for all 15 stocks + 15 crypto and stores them in
the `ohlcv_history` table (created by backend.db). Training data lives here,
separate from the 30-day `ohlcv_daily` the live app uses.

Run ONCE from the project root:

    python scripts/backfill_history.py

After it finishes, verify with the printed summary (or in SQL):

    SELECT symbol, COUNT(*), MIN(date), MAX(date) FROM ohlcv_history GROUP BY symbol;

Re-running is safe — rows are upserted on (symbol, date), so nothing duplicates.
Use --symbols AAPL BTC to backfill a subset (e.g. a quick smoke test).
"""

import argparse
import asyncio
import math
import sys
import time
from pathlib import Path

import yfinance as yf

# Make `backend` importable when run as `python scripts/backfill_history.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.db import history_summary, init_db, insert_history  # noqa: E402
from backend.ingestion import STOCK_META  # noqa: E402

# ── Symbol universe ───────────────────────────────────────────────────────────
# Stocks: reuse the live catalogue. DB label == ticker.
STOCK_SYMBOLS: dict[str, str] = {sym: sym for sym in STOCK_META}

# Crypto: CoinGecko ids (ingestion.CRYPTO_IDS) → yfinance ticker → clean DB label.
CRYPTO_SYMBOLS: dict[str, str] = {
    "BTC-USD": "BTC",
    "ETH-USD": "ETH",
    "USDT-USD": "USDT",  # stablecoin — stored for completeness, skip when training
    "BNB-USD": "BNB",
    "SOL-USD": "SOL",
    "XRP-USD": "XRP",
    "DOGE-USD": "DOGE",
    "ADA-USD": "ADA",
    "AVAX-USD": "AVAX",
    "DOT-USD": "DOT",
    "LINK-USD": "LINK",
    "UNI7083-USD": "UNI",  # plain UNI-USD is delisted on yfinance; this is real Uniswap
    "LTC-USD": "LTC",
    "SHIB-USD": "SHIB",
    "TRX-USD": "TRX",
}

# Macro / market-context series (free, no key). Not prediction targets — used to build
# cross-asset features (market return, volatility regime, rates) in train.load_dataset.
MACRO_SYMBOLS: dict[str, str] = {
    "^GSPC": "SPX",  # S&P 500 — broad market return
    "^VIX": "VIX",  # volatility index — risk-on/off regime
    "^TNX": "TNX",  # 10-year Treasury yield
}

# yfinance ticker → (DB label, is_crypto)
UNIVERSE: dict[str, tuple[str, bool]] = {
    **{tkr: (label, False) for tkr, label in STOCK_SYMBOLS.items()},
    **{tkr: (label, True) for tkr, label in CRYPTO_SYMBOLS.items()},
    **{tkr: (label, False) for tkr, label in MACRO_SYMBOLS.items()},
}

_REQUEST_DELAY = 1.0  # seconds between symbols → avoid yfinance rate limits


def fetch_symbol(yf_ticker: str, label: str, is_crypto: bool) -> list[dict]:
    """Pull full history for one symbol and shape it into insert_history rows."""
    df = yf.Ticker(yf_ticker).history(period="max", interval="1d", auto_adjust=False)
    if df.empty:
        return []

    has_adj = "Adj Close" in df.columns

    def clean(x) -> float | None:
        """Float with enough precision for micro-price tokens; None if NaN."""
        v = float(x)
        if math.isnan(v):
            return None
        # 10 significant figures preserves SHIB-scale prices without float noise.
        return float(f"{v:.10g}")

    rows: list[dict] = []
    for ts, row in df.iterrows():
        close = clean(row["Close"])
        if close is None or close <= 0:
            continue  # skip genuinely empty bars (e.g. early illiquid days)
        # Crypto has no splits/dividends → adj_close == close.
        adj = close if is_crypto else (clean(row["Adj Close"]) if has_adj else close)
        rows.append(
            {
                "symbol": label,
                "date": str(ts.date()),
                "open": clean(row["Open"]),
                "high": clean(row["High"]),
                "low": clean(row["Low"]),
                "close": close,
                "adj_close": adj if adj is not None else close,
                "volume": int(row.get("Volume", 0) or 0),
            }
        )
    return rows


async def main(only: list[str] | None = None) -> None:
    await init_db()  # ensures ohlcv_history exists even if the app never ran

    universe = UNIVERSE
    if only:
        wanted = {s.upper() for s in only}
        universe = {
            tkr: meta
            for tkr, meta in UNIVERSE.items()
            if tkr.upper() in wanted or meta[0].upper() in wanted
        }
        if not universe:
            print(
                f"No matching symbols for {only}. Known labels: "
                f"{sorted(m[0] for m in UNIVERSE.values())}"
            )
            return

    print(f"Backfilling {len(universe)} symbols -> ohlcv_history\n")
    total = 0
    for i, (yf_ticker, (label, is_crypto)) in enumerate(universe.items(), 1):
        try:
            rows = fetch_symbol(yf_ticker, label, is_crypto)
            await insert_history(rows)
            total += len(rows)
            span = f"{rows[0]['date']} -> {rows[-1]['date']}" if rows else "-"
            print(f"  [{i:>2}/{len(universe)}] {label:<6} {len(rows):>6} rows  {span}")
        except Exception as exc:
            print(f"  [{i:>2}/{len(universe)}] {label:<6} FAILED: {exc}")
        time.sleep(_REQUEST_DELAY)

    print(f"\nDone. {total} total rows written.\n")
    print(f"{'SYMBOL':<8}{'ROWS':>8}  {'FIRST':<12}{'LAST':<12}")
    print("-" * 40)
    for r in await history_summary():
        flag = "  (!) thin" if r["rows"] < 750 else ""
        print(f"{r['symbol']:<8}{r['rows']:>8}  {r['first']:<12}{r['last']:<12}{flag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Backfill historical OHLCV into ohlcv_history.")
    ap.add_argument(
        "--symbols", nargs="*", help="Subset of tickers/labels (e.g. AAPL BTC). Default: all 30."
    )
    args = ap.parse_args()
    asyncio.run(main(args.symbols))
