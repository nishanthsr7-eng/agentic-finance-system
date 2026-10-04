"""
Datasource — Alpha Vantage NEWS_SENTIMENT  ->  point-in-time daily per-symbol sentiment (Phase 5).

Alpha Vantage's ``NEWS_SENTIMENT`` endpoint returns a feed of articles, each carrying a model
``ticker_sentiment`` score in [-1, +1] and a ``relevance_score`` per mentioned ticker. We collapse the
feed to one **relevance-weighted daily mean** per (symbol, publication day), plus the article count.
This is an EQUITY-side stream (AV's coverage is US equities/ETFs; crypto tickers are sparse), and it is
genuinely historical (``time_from`` back-fills), so — unlike the real-time FinBERT scores — it can enter
OOF training.

LEAKAGE: a row dated day ``D`` aggregates articles whose ``time_published`` is on D, known by end of D;
the pipeline reads row D to predict D+1 (>=1-day lag). The feature layer applies only causal windows.

Needs ``ALPHA_VANTAGE_API_KEY`` (already in .env). The free tier is heavily rate-limited (≈25 req/day),
so this is a slow back-fill, not a live stream — which is fine, it's cached to
``Dataset/av_news/av_news.csv`` and read from there. No key / no network / throttled → EMPTY frame
(downstream stays neutral), never a crash.

    load_av_news()              -> DataFrame[symbol, date, av_sent, av_relevance, av_n]  (cached)
    fetch_av_news(symbols,...)  -> DataFrame  (re-pull from the API, rewrite the cache)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

import pandas as pd

from . import DATASET_DIR, tidy

log = logging.getLogger("flux.prediction.datasources.av_news")

_AV_DIR = DATASET_DIR / "av_news"
_CACHE = _AV_DIR / "av_news.csv"
_API = "https://www.alphavantage.co/query"


def _aggregate_feed(feed: list[dict], symbol: str) -> pd.DataFrame:
    """AV feed -> DataFrame[date, av_sent, av_relevance, av_n] (relevance-weighted daily mean)."""
    sym = symbol.upper()
    recs = []
    for art in feed or []:
        tp = pd.to_datetime(
            str(art.get("time_published", "")).replace("T", ""),
            format="%Y%m%d%H%M%S",
            errors="coerce",
        )
        if pd.isna(tp):
            continue
        for ts in art.get("ticker_sentiment", []) or []:
            if (ts.get("ticker") or "").upper() != sym:
                continue
            try:
                rel = float(ts.get("relevance_score", 0))
                sc = float(ts.get("ticker_sentiment_score", 0))
            except (TypeError, ValueError):
                continue
            recs.append({"date": tp.normalize(), "rel": rel, "sc": sc})
    if not recs:
        return pd.DataFrame(columns=["date", "av_sent", "av_relevance", "av_n"])
    df = pd.DataFrame(recs)
    df["wsc"] = df["rel"] * df["sc"]
    g = df.groupby("date")
    out = pd.DataFrame(
        {
            "av_sent": g["wsc"].sum() / g["rel"].sum().replace(0, pd.NA),
            "av_relevance": g["rel"].mean(),
            "av_n": g["sc"].size,
        }
    ).reset_index()
    out["av_sent"] = out["av_sent"].fillna(g["sc"].mean().values)
    return out


def _fetch_one(symbol: str, key: str, time_from: str, limit: int, timeout: float) -> pd.DataFrame:
    import httpx

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol.upper(),
        "time_from": time_from,
        "limit": str(limit),
        "sort": "EARLIEST",
        "apikey": key,
    }
    try:
        r = httpx.get(_API, params=params, timeout=timeout)
        r.raise_for_status()
        payload = r.json()
    except Exception as exc:
        log.warning("AV news fetch for %s failed: %s", symbol, exc)
        return pd.DataFrame(columns=["date", "av_sent", "av_relevance", "av_n"])
    if "feed" not in payload:  # rate-limit / info note instead of data
        log.warning("AV news for %s: no feed (%s)", symbol, str(payload)[:120])
        return pd.DataFrame(columns=["date", "av_sent", "av_relevance", "av_n"])
    df = _aggregate_feed(payload["feed"], symbol)
    df["symbol"] = symbol.upper()
    return df


def fetch_av_news(
    symbols: list[str],
    time_from: str = "20220101T0000",
    limit: int = 1000,
    pause: float = 15.0,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Re-pull AV news sentiment per symbol and rewrite the cache. No key -> no-op (cached frame)."""
    from ...config import settings

    key = settings.ALPHA_VANTAGE_API_KEY
    if not key:
        log.warning("AV news: ALPHA_VANTAGE_API_KEY not set — skipping")
        return load_av_news()
    frames = []
    for s in symbols:
        df = _fetch_one(s, key, time_from, limit, timeout)
        if not df.empty:
            frames.append(df)
            log.info("AV news %s: %d daily rows", s, len(df))
        time.sleep(pause)  # free tier: ~25 req/day
    if not frames:
        log.warning("AV news: no data fetched — cache left unchanged")
        return load_av_news()
    out = tidy(
        pd.concat(frames, ignore_index=True)[["symbol", "date", "av_sent", "av_relevance", "av_n"]]
    )
    _AV_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(_CACHE, index=False)
    return out


def load_av_news() -> pd.DataFrame:
    """Cached daily per-symbol AV news sentiment (empty frame if never fetched)."""
    cols = ["symbol", "date", "av_sent", "av_relevance", "av_n"]
    if not _CACHE.exists():
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(_CACHE, parse_dates=["date"])
    except Exception as exc:
        log.warning("AV news cache unreadable: %s", exc)
        return pd.DataFrame(columns=cols)
    return df[[c for c in cols if c in df.columns]]


if __name__ == "__main__":
    import sys

    if "--fetch" in sys.argv or "--rebuild" in sys.argv:
        only = [a.upper() for a in sys.argv[1:] if not a.startswith("-")]
        df = fetch_av_news(
            only or ["AAPL", "NVDA"], time_from=datetime.utcnow().strftime("%Y0101T0000")
        )
    else:
        df = load_av_news()
    if df.empty:
        print(
            "AV news: no data (run with --fetch + an ALPHA_VANTAGE_API_KEY, or it was rate-limited)."
        )
        sys.exit(0)
    print(
        f"AV news: {len(df):,} rows over {df.symbol.nunique()} symbols "
        f"({df.date.min().date()}..{df.date.max().date()})"
    )
    print(
        df.groupby("symbol")
        .agg(n=("av_sent", "size"), mean=("av_sent", "mean"))
        .round(3)
        .to_string()
    )
