"""
Datasource — GDELT 2.0 news *tone* timeline  ->  point-in-time daily per-symbol sentiment (Phase 5).

GDELT's DOC 2.0 API (https://api.gdeltproject.org/api/v2/doc/doc) exposes, for any free-text query, a
daily TIMELINE of the **average tone** of matching worldwide news coverage (mode=timelinetone) plus the
matching article **volume** (the per-point ``norm``). Tone is roughly [-100, +100] but in practice sits
near [-10, +10] (negative = more negative coverage). It is FREE and KEYLESS, with history back to 2017,
which is what makes it usable as a *trainable* sentiment stream (unlike the real-time-only FinBERT
news scores, whose history is too shallow to enter OOF training).

LEAKAGE: a point dated day ``D`` is the tone of articles *published on D* — known by end of day D. The
training pipeline reads row D to predict the move starting D+1, so a >=1-day lag is always in force; no
forward peek. The feature layer (``sentiment_features.py``) only ever applies causal trailing windows.

Like the other ``datasources/`` loaders this does *raw alignment only* (date, tone, vol); the z-scores /
momentum live in the feature module. The assembled panel is CACHED to ``Dataset/gdelt/gdelt_tone.csv``;
``load_gdelt_tone()`` reads the cache, ``fetch_gdelt_tone(...)`` (or ``--rebuild``) re-pulls from the API.
Everything degrades to an EMPTY frame on no-cache / no-network / throttling (HTTP 429) — never crashes —
so the downstream block simply stays neutral, exactly like a crypto symbol with no SEC fundamentals.

    load_gdelt_tone()                 -> DataFrame[symbol, date, tone, vol]   (cached; empty if none)
    fetch_gdelt_tone(symbols, ...)    -> DataFrame  (re-pull from the API and rewrite the cache)
    GDELT_QUERY                       -> {symbol: free-text query}
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

import pandas as pd

from . import CRYPTO_SYMBOLS, DATASET_DIR, tidy

log = logging.getLogger("flux.prediction.datasources.gdelt")

_GDELT_DIR = DATASET_DIR / "gdelt"
_CACHE = _GDELT_DIR / "gdelt_tone.csv"
_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# Free-text query per symbol. Equities use the company name (quoted phrase) OR'd with the ticker
# in a finance context; crypto uses the coin name OR'd with its ticker. Kept deliberately tight so
# the tone reflects the asset, not an unrelated homonym.
# Distinctive single token / quoted phrase per coin (GDELT ANDs bare words, so multi-word queries
# crush recall; a quoted phrase requires the exact bigram). Kept as the most identifying single term.
_CRYPTO_NAME = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "BNB": "binance",
    "SOL": "solana",
    "XRP": "ripple",
    "ADA": "cardano",
    "AVAX": "avalanche",
    "DOT": "polkadot",
    "LINK": "chainlink",
    "LTC": "litecoin",
    "TRX": "tron",
    "UNI": "uniswap",
    "DOGE": "dogecoin",
    "SHIB": '"shiba inu"',
}


def _equity_query(symbol: str) -> str:
    try:
        from ...ingestion import STOCK_META

        name = STOCK_META.get(symbol.upper(), (symbol, ""))[0]
    except Exception:
        name = symbol
    return f'"{name}"'


def _build_query_map() -> dict[str, str]:
    out = {s: _CRYPTO_NAME[s] for s in CRYPTO_SYMBOLS if s in _CRYPTO_NAME}
    try:
        from ...ingestion import STOCK_META

        for sym in STOCK_META:
            out[sym] = _equity_query(sym)
    except Exception:
        pass
    return out


GDELT_QUERY = _build_query_map()


def _parse_timeline(payload: dict) -> pd.DataFrame:
    """GDELT timelinetone JSON -> DataFrame[date, tone, vol]. Tolerant of the two date encodings."""
    series = (payload or {}).get("timeline") or []
    if not series:
        return pd.DataFrame(columns=["date", "tone", "vol"])
    data = series[0].get("data") or []
    rows = []
    for pt in data:
        raw = pt.get("date")
        ts = pd.to_datetime(str(raw).replace("Z", ""), errors="coerce", utc=False)
        if pd.isna(ts):
            continue
        rows.append({"date": ts.normalize(), "tone": pt.get("value"), "vol": pt.get("norm")})
    df = pd.DataFrame(rows)
    return df if not df.empty else pd.DataFrame(columns=["date", "tone", "vol"])


def _compose(query: str) -> str:
    """GDELT query string. Parens are only legal around OR'd statements, so we never wrap a bare term
    or quoted phrase — just append the English-source filter (a space = AND)."""
    return f"{query} sourcelang:english"


def _fetch_one(
    symbol: str,
    query: str,
    start: str,
    end: str,
    timeout: float,
    retries: int = 4,
    backoff: float = 6.0,
) -> pd.DataFrame:
    """One symbol's daily tone timeline; retries the intermittent 429 with linear backoff; empty
    frame on persistent failure (so a partial backfill still proceeds)."""
    import httpx

    params = {
        "query": _compose(query),
        "mode": "timelinetone",
        "format": "json",
        "startdatetime": start,
        "enddatetime": end,
    }
    empty = pd.DataFrame(columns=["date", "tone", "vol"])
    for attempt in range(retries):
        try:
            r = httpx.get(
                _API,
                params=params,
                timeout=timeout,
                headers={"User-Agent": "flux-market/0.1 (research)"},
            )
            if r.status_code == 429:  # intermittent throttle — back off & retry
                time.sleep(backoff * (attempt + 1))
                continue
            r.raise_for_status()
            df = _parse_timeline(r.json())
            df["symbol"] = symbol.upper()
            return df
        except Exception as exc:  # network / parse
            log.warning("GDELT fetch for %s failed (attempt %d): %s", symbol, attempt + 1, exc)
            time.sleep(backoff)
    log.warning("GDELT %s: gave up after %d attempts (throttled)", symbol, retries)
    return empty


def fetch_gdelt_tone(
    symbols: list[str] | None = None,
    start: str = "20170101000000",
    end: str | None = None,
    pause: float = 1.5,
    timeout: float = 20.0,
) -> pd.DataFrame:
    """Re-pull the daily tone timeline for each symbol from the GDELT API and rewrite the cache.

    Polite by default (a short pause between calls); whatever symbols succeed are cached, so a partial
    pull under throttling still yields a usable (if sparse) panel. Returns the assembled frame.
    """
    qmap = GDELT_QUERY
    syms = [s.upper() for s in (symbols or list(qmap))]
    end = end or datetime.utcnow().strftime("%Y%m%d%H%M%S")
    frames = []
    for s in syms:
        q = qmap.get(s)
        if not q:
            continue
        df = _fetch_one(s, q, start, end, timeout)
        if not df.empty:
            frames.append(df)
            log.info("GDELT %s: %d daily tone points", s, len(df))
        time.sleep(pause)
    if not frames:
        log.warning("GDELT: no data fetched (throttled or offline) - cache left unchanged")
        return load_gdelt_tone()
    out = tidy(pd.concat(frames, ignore_index=True)[["symbol", "date", "tone", "vol"]])
    _GDELT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(_CACHE, index=False)
    return out


def load_gdelt_tone() -> pd.DataFrame:
    """Cached daily per-symbol GDELT tone (empty frame if never fetched / no cache)."""
    cols = ["symbol", "date", "tone", "vol"]
    if not _CACHE.exists():
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(_CACHE, parse_dates=["date"])
    except Exception as exc:
        log.warning("GDELT cache unreadable: %s", exc)
        return pd.DataFrame(columns=cols)
    return df[[c for c in cols if c in df.columns]]


if __name__ == "__main__":
    import sys

    do_fetch = "--fetch" in sys.argv or "--rebuild" in sys.argv
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    if do_fetch:
        # A tiny, polite demo pull (a few symbols, recent window) to prove the pipeline end-to-end.
        syms = [s.upper() for s in only] or ["AAPL", "BTC", "NVDA"]
        start = (datetime.utcnow().replace(microsecond=0)).strftime("%Y") + "0101000000"
        df = fetch_gdelt_tone(syms, start=start)
    else:
        df = load_gdelt_tone()
    if df.empty:
        print("GDELT tone: no data (run with --fetch to pull, or it was throttled/offline).")
        sys.exit(0)
    print(
        f"GDELT tone: {len(df):,} rows over {df.symbol.nunique()} symbols "
        f"({df.date.min().date()}..{df.date.max().date()})"
    )
    print(
        df.groupby("symbol")
        .agg(n=("tone", "size"), mean_tone=("tone", "mean"))
        .round(2)
        .to_string()
    )
