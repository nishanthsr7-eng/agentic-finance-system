"""
FLUX Data Ingestion Scheduler
------------------------------
Uses APScheduler (AsyncIOScheduler). Every interval is a setting — the
defaults are shown, and deployment widens them to stay inside free API quotas:
  • CoinGecko  → crypto price snapshots    → SQLite  (INGESTION_INTERVAL_MIN, 5)
  • Finnhub    → stock price snapshots     → SQLite  (INGESTION_INTERVAL_MIN, 5)
  • yfinance   → 30-day daily OHLCV        → SQLite  (OHLCV_INTERVAL_MIN, 30)
  • NewsAPI    → finance headlines         → SQLite  (NEWS_INTERVAL_MIN, 15)

The AI insight cycle has no trigger of its own: it is chained onto the market
cycle (throttled to INSIGHT_INTERVAL_MIN) so it always reads a fresh cache.

All jobs also update the in-memory cache (same TTLs as the live endpoints)
so the existing /market/* routes always return fresh data.
"""

import asyncio
import logging
import math
import time
from datetime import datetime, timedelta
from typing import Any

import httpx
import yfinance as yf

from .cache import cache
from .config import settings
from .db import insert_history, insert_news, insert_snapshots, log_ingestion, upsert_ohlcv

log = logging.getLogger("flux.ingestion")

# ── Shared HTTP client ────────────────────────────────────────────────────────
_client: httpx.AsyncClient | None = None


def _http() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=15.0, follow_redirects=True)
    return _client


async def close_client() -> None:
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()


# ── Status tracking ───────────────────────────────────────────────────────────
_status: dict[str, Any] = {}  # job → {ts, rows, status}


def get_status() -> dict:
    return dict(_status)


def _record(job: str, rows: int, status: str = "ok", msg: str = "") -> None:
    _status[job] = {"ts": int(time.time() * 1000), "rows": rows, "status": status, "msg": msg}


# ── Asset catalogues (mirrors main.py) ───────────────────────────────────────
CRYPTO_IDS = [
    "bitcoin",
    "ethereum",
    "tether",
    "binancecoin",
    "solana",
    "ripple",
    "dogecoin",
    "cardano",
    "avalanche-2",
    "polkadot",
    "chainlink",
    "uniswap",
    "litecoin",
    "shiba-inu",
    "tron",
]

STOCK_META: dict[str, tuple[str, str]] = {
    "AAPL": ("Apple Inc.", "Technology"),
    "MSFT": ("Microsoft Corp.", "Technology"),
    "NVDA": ("NVIDIA Corp.", "Semiconductors"),
    "GOOGL": ("Alphabet Inc.", "Technology"),
    "AMZN": ("Amazon.com Inc.", "Consumer"),
    "TSLA": ("Tesla Inc.", "Automotive"),
    "META": ("Meta Platforms", "Technology"),
    "NFLX": ("Netflix Inc.", "Media"),
    "JPM": ("JPMorgan Chase", "Financials"),
    "AMD": ("Advanced Micro Dev.", "Semiconductors"),
    "TSM": ("Taiwan Semiconductor", "Semiconductors"),
    "ORCL": ("Oracle Corp.", "Technology"),
    "CRM": ("Salesforce Inc.", "Software"),
    "INTC": ("Intel Corp.", "Semiconductors"),
    "BABA": ("Alibaba Group", "Consumer"),
}
STOCK_SYMBOLS = list(STOCK_META.keys())
_FMP = "https://financialmodelingprep.com/image-stock"

# Symbols for OHLCV (yfinance ticker → label stored in DB)
OHLCV_SYMBOLS: dict[str, str] = {
    "BTC-USD": "BTC",
    "ETH-USD": "ETH",
    "SOL-USD": "SOL",
    "AAPL": "AAPL",
    "NVDA": "NVDA",
    "TSLA": "TSLA",
    "MSFT": "MSFT",
}

# ── Training-data universe for ohlcv_history ─────────────────────────────────
# yfinance ticker → (DB label, is_crypto). Mirrors scripts/backfill_history.py's
# UNIVERSE — the 15 stocks above, 15 crypto and 3 macro series (SPX/VIX/TNX).
HISTORY_UNIVERSE: dict[str, tuple[str, bool]] = {
    **{sym: (sym, False) for sym in STOCK_SYMBOLS},
    "BTC-USD": ("BTC", True),
    "ETH-USD": ("ETH", True),
    "USDT-USD": ("USDT", True),
    "BNB-USD": ("BNB", True),
    "SOL-USD": ("SOL", True),
    "XRP-USD": ("XRP", True),
    "DOGE-USD": ("DOGE", True),
    "ADA-USD": ("ADA", True),
    "AVAX-USD": ("AVAX", True),
    "DOT-USD": ("DOT", True),
    "LINK-USD": ("LINK", True),
    "UNI7083-USD": ("UNI", True),  # plain UNI-USD is delisted on yfinance
    "LTC-USD": ("LTC", True),
    "SHIB-USD": ("SHIB", True),
    "TRX-USD": ("TRX", True),
    "^GSPC": ("SPX", False),
    "^VIX": ("VIX", False),
    "^TNX": ("TNX", False),
}

_FINANCE_DOMAINS = (
    "coindesk.com,cointelegraph.com,reuters.com,bloomberg.com,"
    "cnbc.com,marketwatch.com,wsj.com,ft.com,investing.com,"
    "cryptonews.com,decrypt.co,theblock.co"
)
_GENERAL_QUERY = (
    'bitcoin OR ethereum OR cryptocurrency OR "stock market" OR '
    '"Federal Reserve" OR "interest rate" OR "S&P 500"'
)


# ── Job 1: Crypto (CoinGecko) ─────────────────────────────────────────────────
async def ingest_crypto() -> list[dict]:
    ts = int(time.time() * 1000)
    params: dict[str, Any] = {
        "vs_currency": "usd",
        "ids": ",".join(CRYPTO_IDS),
        "order": "market_cap_desc",
        "per_page": len(CRYPTO_IDS),
        "page": 1,
        "sparkline": "true",
        "price_change_percentage": "24h",
    }
    headers = {}
    if settings.COINGECKO_API_KEY:
        headers["x-cg-demo-api-key"] = settings.COINGECKO_API_KEY

    try:
        r = await _http().get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params=params,
            headers=headers,
        )
        r.raise_for_status()
        coins = r.json()
    except Exception as exc:
        log.warning("CoinGecko fetch failed: %s", exc)
        _record("crypto", 0, "error", str(exc))
        await log_ingestion("crypto", "error", 0, str(exc))
        return []

    assets, rows = [], []
    for coin in coins:
        sym = (coin.get("symbol") or "").upper()
        price = float(coin.get("current_price") or 0)
        change = float(coin.get("price_change_percentage_24h") or 0)
        mcap = float(coin.get("market_cap") or 0)
        vol = float(coin.get("total_volume") or 0)

        assets.append(
            {
                "symbol": f"BINANCE:{sym}USDT",
                "name": coin.get("name", sym),
                "sub": sym,
                "price": price,
                "change_pct": change,
                "market_cap": mcap,
                "icon": coin.get("image", ""),
                "sector": "",
                "sparkline_7d": (coin.get("sparkline_in_7d") or {}).get("price", []),
            }
        )
        rows.append(
            {
                "symbol": sym,
                "asset_type": "crypto",
                "name": coin.get("name", sym),
                "price": price,
                "change_pct": change,
                "volume": vol,
                "market_cap": mcap,
                "ts": ts,
            }
        )

    # Update in-memory cache so live endpoints stay fresh
    cache.set("crypto", assets, ttl=settings.CRYPTO_TTL)
    await insert_snapshots(rows)
    _record("crypto", len(rows))
    await log_ingestion("crypto", "ok", len(rows))
    log.info("Ingested %d crypto snapshots", len(rows))
    return assets


# ── Job 2: Stocks (Finnhub) ───────────────────────────────────────────────────
async def _fetch_one_stock(sym: str) -> dict | None:
    try:
        r = await _http().get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": sym, "token": settings.FINNHUB_API_KEY},
        )
        if r.status_code != 200:
            return None
        q = r.json()
        price = float(q.get("c") or 0)
        if price == 0:
            return None
        prev = float(q.get("pc") or price)
        chg = ((price - prev) / prev * 100) if prev else 0
        vol = float(q.get("v") or 0)
        name, sector = STOCK_META.get(sym, (sym, ""))
        return {
            "sym": sym,
            "name": name,
            "sector": sector,
            "price": price,
            "chg": round(chg, 3),
            "vol": vol,
        }
    except Exception as exc:
        log.debug("Finnhub %s: %s", sym, exc)
        return None


async def ingest_stocks() -> list[dict]:
    if not settings.FINNHUB_API_KEY:
        return []
    ts = int(time.time() * 1000)
    results = await asyncio.gather(*[_fetch_one_stock(s) for s in STOCK_SYMBOLS])
    assets, rows = [], []
    for r in results:
        if r is None:
            continue
        assets.append(
            {
                "symbol": r["sym"],
                "name": r["name"],
                "sub": r["sym"],
                "price": r["price"],
                "change_pct": r["chg"],
                "market_cap": 0,
                "icon": f"{_FMP}/{r['sym']}.png",
                "sector": r["sector"],
            }
        )
        rows.append(
            {
                "symbol": r["sym"],
                "asset_type": "stock",
                "name": r["name"],
                "price": r["price"],
                "change_pct": r["chg"],
                "volume": r["vol"],
                "market_cap": 0,
                "ts": ts,
            }
        )

    if assets:
        cache.set("stocks", assets, ttl=settings.STOCKS_TTL)
        await insert_snapshots(rows)

    _record("stocks", len(rows))
    await log_ingestion("stocks", "ok", len(rows))
    log.info("Ingested %d stock snapshots", len(rows))
    return assets


# ── Job 3: OHLCV (yfinance, sync in executor) ────────────────────────────────
def _fetch_ohlcv_sync() -> list[dict]:
    rows = []
    for yf_sym, label in OHLCV_SYMBOLS.items():
        try:
            df = yf.Ticker(yf_sym).history(period="30d", interval="1d", auto_adjust=True)
            if df.empty:
                continue
            for ts, row in df.iterrows():
                rows.append(
                    {
                        "symbol": label,
                        "date": str(ts.date()),
                        "open": float(f"{float(row['Open']):.6g}"),
                        "high": float(f"{float(row['High']):.6g}"),
                        "low": float(f"{float(row['Low']):.6g}"),
                        "close": float(f"{float(row['Close']):.6g}"),
                        "volume": int(row.get("Volume", 0) or 0),
                    }
                )
        except Exception as exc:
            log.warning("yfinance %s: %s", yf_sym, exc)
    return rows


async def ingest_ohlcv() -> None:
    loop = asyncio.get_event_loop()
    try:
        rows = await loop.run_in_executor(None, _fetch_ohlcv_sync)
        await upsert_ohlcv(rows)
        _record("ohlcv", len(rows))
        await log_ingestion("ohlcv", "ok", len(rows))
        log.info("Ingested %d OHLCV rows", len(rows))
    except Exception as exc:
        _record("ohlcv", 0, "error", str(exc))
        await log_ingestion("ohlcv", "error", 0, str(exc))
        log.warning("OHLCV ingestion failed: %s", exc)


# ── Job 9: ohlcv_history daily append (training data) ────────────────────────
def _clean_price(x) -> float | None:
    """Float with enough precision for micro-price tokens; None if NaN."""
    v = float(x)
    if math.isnan(v):
        return None
    return float(f"{v:.10g}")  # 10 sig figs preserves SHIB-scale prices


def _fetch_history_sync(
    period: str = "1mo", universe: dict[str, tuple[str, bool]] | None = None
) -> list[dict]:
    """Pull a recent OHLCV window for the training universe and shape it for
    insert_history. Mirrors scripts/backfill_history.fetch_symbol but with a
    bounded lookback by default — insert_history upserts on (symbol, date) so
    re-fetching overlapping days is harmless. Pass period="max" + a one-symbol
    universe for a one-off backfill of a newly-added symbol."""
    rows: list[dict] = []
    for yf_ticker, (label, is_crypto) in (universe or HISTORY_UNIVERSE).items():
        try:
            df = yf.Ticker(yf_ticker).history(period=period, interval="1d", auto_adjust=False)
            if df.empty:
                continue
            has_adj = "Adj Close" in df.columns
            for ts, row in df.iterrows():
                close = _clean_price(row["Close"])
                if close is None or close <= 0:
                    continue
                adj = close if is_crypto else (_clean_price(row["Adj Close"]) if has_adj else close)
                rows.append(
                    {
                        "symbol": label,
                        "date": str(ts.date()),
                        "open": _clean_price(row["Open"]),
                        "high": _clean_price(row["High"]),
                        "low": _clean_price(row["Low"]),
                        "close": close,
                        "adj_close": adj if adj is not None else close,
                        "volume": int(row.get("Volume", 0) or 0),
                    }
                )
        except Exception as exc:
            log.warning("ohlcv_history fetch %s: %s", yf_ticker, exc)
    return rows


async def ingest_history_daily() -> None:
    """Append the latest daily bar(s) to ohlcv_history for the training universe
    (29 model symbols + macro context), so predictions anchor on a fresh
    close and resolve_due() can mature outcomes."""
    loop = asyncio.get_event_loop()
    try:
        rows = await loop.run_in_executor(None, _fetch_history_sync)
        await insert_history(rows)
        _record("ohlcv_history", len(rows))
        await log_ingestion("ohlcv_history", "ok", len(rows))
        log.info("Appended %d ohlcv_history rows", len(rows))
    except Exception as exc:
        _record("ohlcv_history", 0, "error", str(exc))
        await log_ingestion("ohlcv_history", "error", 0, str(exc))
        log.warning("ohlcv_history ingestion failed: %s", exc)


# ── Job 4: News (NewsAPI) ─────────────────────────────────────────────────────
async def ingest_news() -> list[dict]:
    if not settings.NEWSAPI_KEY:
        return []
    ts = int(time.time() * 1000)
    try:
        r = await _http().get(
            "https://newsapi.org/v2/everything",
            params={
                "q": _GENERAL_QUERY,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 20,
                "apiKey": settings.NEWSAPI_KEY,
                "domains": _FINANCE_DOMAINS,
            },
        )
        r.raise_for_status()
        articles_raw = r.json().get("articles", [])
    except Exception as exc:
        log.warning("NewsAPI fetch failed: %s", exc)
        _record("news", 0, "error", str(exc))
        return []

    articles, rows = [], []
    for a in articles_raw:
        title = (a.get("title") or "").strip()
        url = a.get("url") or ""
        if not title or title == "[Removed]" or not url:
            continue
        art = {
            "title": title,
            "source": a.get("source", {}).get("name", ""),
            "url": url,
            "summary": (a.get("description") or "")[:200].strip(),
            "published_at": a.get("publishedAt", ""),
            "cached_at": ts,
        }
        articles.append(art)
        rows.append(art)

    await insert_news(rows)
    _record("news", len(rows))
    await log_ingestion("news", "ok", len(rows))
    log.info("Ingested %d news articles", len(rows))
    return articles


# ── Combined cycle (used by scheduler) ───────────────────────────────────────
async def full_market_cycle() -> dict:
    """Run crypto + stocks ingestion together. Called every INGESTION_INTERVAL_MIN."""
    crypto = await ingest_crypto()
    stocks = await ingest_stocks()
    return {"crypto": len(crypto), "stocks": len(stocks)}


# ── Scheduler factory ─────────────────────────────────────────────────────────
def build_scheduler(insight_job_fn=None):
    """
    Build and return a configured AsyncIOScheduler.

    Pass insight_job_fn to enable the RAG insight cycle. It is *chained* onto
    the market cycle rather than given a trigger of its own: the insight job
    reads the in-memory cache that ingestion has just written, so scheduling
    the two independently raced — on a cold start the insight run fired against
    an empty cache and silently produced nothing. Chaining also caps how much
    runs at once, which matters more than the schedule on a 512 MB host.

    Every interval comes from settings; nothing here is hardcoded any more.
    """
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.date import DateTrigger
    from apscheduler.triggers.interval import IntervalTrigger

    scheduler = AsyncIOScheduler(timezone="UTC")

    market_min = max(1, settings.INGESTION_INTERVAL_MIN)
    ohlcv_min = max(market_min, settings.OHLCV_INTERVAL_MIN)
    news_min = max(market_min, settings.NEWS_INTERVAL_MIN)
    insight_min = max(market_min, settings.INSIGHT_INTERVAL_MIN)
    heavy_boot = settings.HEAVY_JOBS_ON_STARTUP

    def _boot_run(minutes: int) -> dict:
        """Kwargs for a one-off run shortly after startup, or nothing at all.

        APScheduler treats an explicit next_run_time=None as "create this job
        paused", which is not what we want — so the key has to be absent, not
        None, when HEAVY_JOBS_ON_STARTUP is off. The cron trigger still fires.
        """
        if not heavy_boot:
            return {}
        return {"next_run_time": datetime.utcnow() + timedelta(minutes=minutes)}

    # 1) Market data + (chained) insights.
    #
    # _last_insight is the throttle: the market cycle may run every 5 minutes
    # while insights only need to run every 15, and each insight costs an LLM
    # call per asset. Awaiting the insight fn inline, inside the same job, is
    # deliberate — it guarantees the cache it reads is the one this job just
    # wrote, and a slow LLM delays the next insight instead of stacking one.
    _last_insight = [0.0]

    async def market_cycle_job() -> dict:
        res = await full_market_cycle()
        if insight_job_fn is None:
            return res
        now = time.time()
        if now - _last_insight[0] < insight_min * 60:
            return res
        _last_insight[0] = now
        try:
            await insight_job_fn()
        except Exception as exc:  # never let insights kill ingestion
            log.warning("insight cycle failed: %s", exc)
        return res

    scheduler.add_job(
        market_cycle_job,
        trigger=IntervalTrigger(minutes=market_min),
        id="market_cycle",
        next_run_time=datetime.utcnow(),
        misfire_grace_time=60,
        coalesce=True,
        max_instances=1,
    )

    # 2) OHLCV — first run 2 min after startup
    scheduler.add_job(
        ingest_ohlcv,
        trigger=IntervalTrigger(minutes=ohlcv_min),
        id="ohlcv",
        next_run_time=datetime.utcnow() + timedelta(minutes=2),
        misfire_grace_time=120,
        coalesce=True,
        max_instances=1,
    )

    # 3) News — first run 3 min after startup. NewsAPI's free plan allows 100
    #    requests/day, so anything under ~15 minutes exhausts it before evening.
    scheduler.add_job(
        ingest_news,
        trigger=IntervalTrigger(minutes=news_min),
        id="news",
        next_run_time=datetime.utcnow() + timedelta(minutes=3),
        misfire_grace_time=120,
        coalesce=True,
        max_instances=1,
    )

    # 4) Prune old snapshots and ingestion_log rows daily at midnight UTC
    from .db import prune_ingestion_log, prune_old_snapshots

    async def prune_job():
        await prune_old_snapshots(settings.SNAPSHOT_RETENTION_DAYS)
        await prune_ingestion_log(settings.INGESTION_LOG_RETENTION_DAYS)

    scheduler.add_job(
        prune_job,
        trigger=CronTrigger(hour=0, minute=0, timezone="UTC"),
        id="prune",
        coalesce=True,
    )

    # 4b) Demo user reseed — daily at 00:05 UTC (before the 00:10 / 00:30 jobs),
    #     plus a self-heal ~4 min after boot, once the OHLCV job (+2 min) has
    #     stored the closes its paper trades are priced from. Light work: a
    #     thousand small inserts, no pandas, no network.
    if settings.DEMO_RESEED:

        async def demo_reseed_job(force: bool = True):
            from .demo_seed import ensure_demo_fresh, seed_demo_user

            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, seed_demo_user if force else ensure_demo_fresh)
            except Exception as exc:  # never let it crash the scheduler
                log.warning("demo reseed failed: %s", exc)

        scheduler.add_job(
            demo_reseed_job,
            trigger=CronTrigger(hour=0, minute=5, timezone="UTC"),
            id="demo_reseed",
            misfire_grace_time=600,
            coalesce=True,
            max_instances=1,
        )
        scheduler.add_job(
            demo_reseed_job,
            trigger=DateTrigger(run_date=datetime.utcnow() + timedelta(minutes=4), timezone="UTC"),
            kwargs={"force": False},
            id="demo_reseed_boot",
            misfire_grace_time=600,
        )

    # 5) ohlcv_history daily append — daily at 00:10 UTC, before options_iv
    #    (00:20) and the prediction cycle (00:30) so they read fresh closes.
    scheduler.add_job(
        ingest_history_daily,
        trigger=CronTrigger(hour=0, minute=10, timezone="UTC"),
        id="ohlcv_history",
        **_boot_run(1),
        misfire_grace_time=600,
        coalesce=True,
        max_instances=1,
    )

    # 6) Prediction cycle daily at 00:30 UTC — the full FLUX-X loop: resolve matured predictions
    #    → log a fresh batch (steps 1–7) → construct the cross-sectional book (step 8) → red-team
    #    its top-k (step 9) → paper dry-run (step 10). Runs only if the trained model artifacts are
    #    present (skips cleanly otherwise).
    #
    #    HEAVY_JOBS_ON_STARTUP gates the boot-time run of this and the two jobs
    #    around it. On a 512 MB host it must be false. The imports are cheap
    #    (~17 MB measured); the *run* is not — it loads price history into
    #    dataframes and fits an HMM and a GARCH per asset, on top of a process
    #    already holding FastAPI, pandas and the ONNX embedder. Doing that a
    #    few minutes after every boot is what turns one OOM kill into a restart
    #    loop, because each restart schedules it again. The cron trigger is
    #    unaffected and POST /ingestion/trigger/predictions runs it on demand.
    from pathlib import Path as _Path

    if (_Path(__file__).parent / "prediction" / "models" / "xgb_primary.json").exists():

        async def prediction_job():
            from .prediction.flux_x import run_flux_x

            try:
                res = await run_flux_x()  # dry-run paper by default (no live orders)
                await log_ingestion("predictions", "ok", res.get("logged", 0), str(res))
            except Exception as exc:  # never let it crash the scheduler
                log.warning("prediction cycle failed: %s", exc)
                await log_ingestion("predictions", "error", 0, str(exc))

        scheduler.add_job(
            prediction_job,
            trigger=CronTrigger(hour=0, minute=30, timezone="UTC"),
            id="predictions",
            **_boot_run(8),
            misfire_grace_time=600,
            coalesce=True,
            max_instances=1,
        )

        # 7) Drift-triggered retrain — weekly (Sun 02:00 UTC). Retrains ONLY if live accuracy/
        #    ECE has drifted past threshold (and the model isn't too fresh); otherwise a no-op.
        async def drift_job():
            from .prediction.drift import maybe_retrain

            try:
                res = await maybe_retrain()
                await log_ingestion(
                    "drift_retrain",
                    "ok" if res.get("retrained") else "skip",
                    0,
                    res.get("reason", ""),
                )
            except Exception as exc:
                log.warning("drift retrain failed: %s", exc)
                await log_ingestion("drift_retrain", "error", 0, str(exc))

        scheduler.add_job(
            drift_job,
            trigger=CronTrigger(day_of_week="sun", hour=2, minute=0, timezone="UTC"),
            id="drift_retrain",
            misfire_grace_time=3600,
            coalesce=True,
            max_instances=1,
        )

        # 8) Options IV/skew snapshot — daily at 00:20 UTC, just before the prediction cycle so
        #    predictions read a same-day chain. Equity-only (crypto returns None); builds the
        #    options_iv flywheel that a future training feature can be validated against.
        async def options_iv_job():
            from .prediction.options import snapshot_iv

            try:
                n = await snapshot_iv(STOCK_SYMBOLS)
                await log_ingestion("options_iv", "ok", n, f"{n} symbols snapshotted")
            except Exception as exc:
                log.warning("options IV snapshot failed: %s", exc)
                await log_ingestion("options_iv", "error", 0, str(exc))

        scheduler.add_job(
            options_iv_job,
            trigger=CronTrigger(hour=0, minute=20, timezone="UTC"),
            id="options_iv",
            **_boot_run(5),
            misfire_grace_time=600,
            coalesce=True,
            max_instances=1,
        )

    return scheduler
