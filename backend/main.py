"""
FLUX Market API  —  main.py
FastAPI backend serving live market data to the FLUX frontend.

Endpoints
---------
GET /health                    — liveness + cache stats
GET /market/quotes/crypto      — top 15 coins via CoinGecko
GET /market/quotes/stocks      — 15 curated equities via Finnhub

Response shape (both endpoints)
--------------------------------
{
  "assets": [
    {
      "symbol":     str,   # e.g. "BINANCE:BTCUSDT" / "AAPL"
      "name":       str,   # e.g. "Bitcoin" / "Apple Inc."
      "sub":        str,   # short ticker  "BTC" / "AAPL"
      "price":      float,
      "change_pct": float, # 24h %
      "market_cap": float, # crypto only  (0 for stocks)
      "icon":       str,   # image URL
      "sector":     str,   # stocks only  ("" for crypto)
    },
    ...
  ],
  "source":    str,
  "cached":    bool,
  "timestamp": str,  # ISO-8601
}

Start
-----
    uvicorn backend.main:app --reload --port 8000
"""

import asyncio
import json
import logging
import re
import time
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
import yfinance as yf
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import llm
from .auth import auth_router, ensure_auth_schema, is_admin, require_admin, require_user
from .cache import cache
from .config import settings
from .db import (
    get_calibration_buckets,
    get_ingestion_log,
    get_latest_insights,
    get_latest_predictions,
    get_latest_snapshots,
    get_ohlcv,
    get_prediction_history,
    get_recent_news,
    get_symbol_history,
    init_db,
)
from .payments_api import payments_router
from .rag import chroma_stats, embed_market_snapshots, init_chroma, rag_query
from .ratelimit import rate_limit
from .trading_api import trading_router
from .user_api import db_router

# One shared per-IP allowance for every route that spends LLM quota.
_AI_LIMIT = Depends(rate_limit("ai", 20, 60))

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
log = logging.getLogger("flux.api")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Agentic AI Finance & Stock Prediction System API",
    version="1.0.0",
    description="Market data, predictions and paper trading for the Agentic AI Finance & Stock Prediction System.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],  # DELETE: watchlist + alerts
    allow_headers=["*"],
)


# Security headers on every API response. CSP here protects rendered API
# output (docs pages etc.); the HTML pages carry their own CSP meta tag and,
# in production, should be served behind the same reverse proxy adding these.
@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers.setdefault("Cache-Control", "no-store")
    return resp


# MySQL-backed dataset for all pages (see backend/user_api.py)
app.include_router(db_router)
# Marketplace paper-trading, watchlist & alerts (see backend/trading_api.py)
app.include_router(trading_router)
# Payments page writes — transactions, recurring, toggles, rewards, account switch
app.include_router(payments_router)
# Login / register / me (see backend/auth.py)
app.include_router(auth_router)

# ── Asset Catalogues ──────────────────────────────────────────────────────────

# CoinGecko IDs — ranked by market cap
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

# Stocks: symbols + static metadata (sector rarely changes)
STOCK_META: dict[str, dict] = {
    "AAPL": {"name": "Apple Inc.", "sector": "Technology"},
    "MSFT": {"name": "Microsoft Corp.", "sector": "Technology"},
    "NVDA": {"name": "NVIDIA Corp.", "sector": "Semiconductors"},
    "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology"},
    "AMZN": {"name": "Amazon.com Inc.", "sector": "Consumer"},
    "TSLA": {"name": "Tesla Inc.", "sector": "Automotive"},
    "META": {"name": "Meta Platforms", "sector": "Technology"},
    "NFLX": {"name": "Netflix Inc.", "sector": "Media"},
    "JPM": {"name": "JPMorgan Chase", "sector": "Financials"},
    "AMD": {"name": "Advanced Micro Dev.", "sector": "Semiconductors"},
    "TSM": {"name": "Taiwan Semiconductor", "sector": "Semiconductors"},
    "ORCL": {"name": "Oracle Corp.", "sector": "Technology"},
    "CRM": {"name": "Salesforce Inc.", "sector": "Software"},
    "INTC": {"name": "Intel Corp.", "sector": "Semiconductors"},
    "BABA": {"name": "Alibaba Group", "sector": "Consumer"},
}

STOCK_SYMBOLS = list(STOCK_META.keys())

# Financial Modeling Prep image CDN — purpose-built stock logos, no auth required
_FMP = "https://financialmodelingprep.com/image-stock"
STOCK_LOGOS: dict[str, str] = {
    sym: f"{_FMP}/{sym}.png"
    for sym in [
        "AAPL",
        "MSFT",
        "NVDA",
        "GOOGL",
        "AMZN",
        "TSLA",
        "META",
        "NFLX",
        "JPM",
        "AMD",
        "TSM",
        "ORCL",
        "CRM",
        "INTC",
        "BABA",
    ]
}

# ── HTTP Client ───────────────────────────────────────────────────────────────
# Shared client — reuses TCP connections across requests
_http_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=12.0, follow_redirects=True)
    return _http_client


_scheduler = None  # APScheduler instance


def _cached_assets() -> list[dict]:
    """Flatten the crypto + stock caches into the shape the insight cycle wants.

    Returns [] when ingestion has not populated the cache yet; every caller
    treats that as "nothing to do" rather than an error.
    """
    crypto = cache.get("crypto") or []
    stocks = cache.get("stocks") or []
    return [
        {
            "symbol": a["sub"],
            "name": a["name"],
            "price": a["price"],
            "change_pct": a["change_pct"],
            "asset_type": "crypto",
        }
        for a in crypto
    ] + [
        {
            "symbol": a["sub"],
            "name": a["name"],
            "price": a["price"],
            "change_pct": a["change_pct"],
            "asset_type": "stock",
        }
        for a in stocks
    ]


async def _insight_job() -> None:
    """One AI insight cycle over the freshest cached assets.

    The scheduler awaits this at the end of a market cycle, so by the time it
    runs the cache was written moments ago. It is also what the manual
    /ai/insights/refresh and /ingestion/trigger/insights routes call.
    """
    from .insights import run_insight_cycle

    all_assets = _cached_assets()
    if not all_assets:
        log.debug("Insight job skipped — cache not warm yet")
        return
    generated = await run_insight_cycle(all_assets, settings.INSIGHT_MAX_ASSETS)
    # Also embed the freshest market snapshots into ChromaDB
    embed_market_snapshots(all_assets)
    log.info("Insight cycle complete: %d insights generated", len(generated))


@app.on_event("startup")
async def startup_event():
    # 0. MySQL schema migrations (backend/migrations), then the demo-user
    #    credential seed that needs users.password_hash to exist.
    from .migrate import run_migrations_safely

    run_migrations_safely()
    ensure_auth_schema()

    # 1. SQLite schema
    await init_db()

    # 2. ChromaDB (optional — degrades gracefully)
    init_chroma()

    if not settings.INGESTION_ENABLED:
        log.info("Ingestion scheduler disabled (INGESTION_ENABLED=false)")
        return

    # 3. Build and start the scheduler. The insight cycle is passed in and
    #    chained onto the market cycle there — see ingestion.build_scheduler.
    from .ingestion import build_scheduler

    global _scheduler
    _scheduler = build_scheduler(insight_job_fn=_insight_job)
    _scheduler.start()
    log.info("APScheduler started — %d jobs registered", len(_scheduler.get_jobs()))


@app.on_event("shutdown")
async def shutdown_event():
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        log.info("APScheduler stopped")
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
    from .ingestion import close_client as _close_ingest_client

    await _close_ingest_client()
    from .mysql_db import close_pool

    close_pool()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _wrap(assets: list[dict], source: str, cached: bool) -> dict:
    return {"assets": assets, "source": source, "cached": cached, "timestamp": _now_iso()}


# ── Crypto Fetcher ────────────────────────────────────────────────────────────


async def _fetch_crypto_live() -> list[dict]:
    """Fetch top coins from CoinGecko /coins/markets."""
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
        # Demo key uses x-cg-demo-api-key header
        headers["x-cg-demo-api-key"] = settings.COINGECKO_API_KEY

    r = await get_client().get(
        "https://api.coingecko.com/api/v3/coins/markets",
        params=params,
        headers=headers,
    )
    r.raise_for_status()
    coins = r.json()

    assets = []
    for coin in coins:
        symbol_short = (coin.get("symbol") or "").upper()
        assets.append(
            {
                "symbol": f"BINANCE:{symbol_short}USDT",
                "name": coin.get("name", symbol_short),
                "sub": symbol_short,
                "price": float(coin.get("current_price") or 0),
                "change_pct": float(coin.get("price_change_percentage_24h") or 0),
                "market_cap": float(coin.get("market_cap") or 0),
                "icon": coin.get("image", ""),
                "sector": "",
                "sparkline_7d": (coin.get("sparkline_in_7d") or {}).get("price", []),
            }
        )
    log.info("CoinGecko: fetched %d coins", len(assets))
    return assets


# ── Stocks Fetcher ────────────────────────────────────────────────────────────


async def _fetch_single_stock(client: httpx.AsyncClient, symbol: str) -> dict | None:
    """Fetch a single Finnhub quote. Returns None on error."""
    try:
        r = await client.get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": symbol, "token": settings.FINNHUB_API_KEY},
        )
        if r.status_code != 200:
            return None
        q = r.json()
        price = float(q.get("c") or 0)
        if price == 0:
            return None  # market closed / bad symbol — skip
        prev = float(q.get("pc") or price)
        chg = ((price - prev) / prev * 100) if prev else 0
        meta = STOCK_META.get(symbol, {"name": symbol, "sector": "—"})
        return {
            "symbol": symbol,
            "name": meta["name"],
            "sub": symbol,
            "price": price,
            "change_pct": round(chg, 3),
            "market_cap": 0,
            "icon": STOCK_LOGOS.get(symbol, ""),
            "sector": meta["sector"],
        }
    except Exception as exc:
        log.warning("Finnhub %s failed: %s", symbol, exc)
        return None


async def _fetch_stocks_live() -> list[dict]:
    """Fetch all stock quotes concurrently from Finnhub."""
    client = get_client()
    tasks = [_fetch_single_stock(client, sym) for sym in STOCK_SYMBOLS]
    results = await asyncio.gather(*tasks)
    assets = [r for r in results if r is not None]
    log.info("Finnhub: fetched %d/%d stocks", len(assets), len(STOCK_SYMBOLS))
    return assets


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/market/summary")
async def market_summary():
    """
    Returns 4 highlight assets for the marketplace strip cards:
    top-2 cryptos by market cap, plus the 2 stocks with the biggest absolute 24h move.
    Backed by the same caches as /market/quotes/* — no extra upstream calls.
    """
    crypto = cache.get("crypto")
    if crypto is None:
        try:
            crypto = await _fetch_crypto_live()
            cache.set("crypto", crypto, ttl=settings.CRYPTO_TTL)
        except Exception:
            crypto = []

    stocks = cache.get("stocks")
    if stocks is None and settings.FINNHUB_API_KEY:
        try:
            stocks = await _fetch_stocks_live()
            cache.set("stocks", stocks, ttl=settings.STOCKS_TTL)
        except Exception:
            stocks = []
    stocks = stocks or []

    top_crypto = sorted(crypto, key=lambda a: a.get("market_cap", 0), reverse=True)[:2]
    top_stocks = sorted(stocks, key=lambda a: abs(a.get("change_pct", 0)), reverse=True)[:2]

    def _strip(a: dict, category: str) -> dict:
        return {
            "symbol": a.get("symbol"),
            "name": a.get("name"),
            "sub": a.get("sub"),
            "price": a.get("price"),
            "change_pct": a.get("change_pct"),
            "icon": a.get("icon"),
            "category": category,
        }

    highlights = [_strip(a, "crypto") for a in top_crypto] + [
        _strip(a, "stocks") for a in top_stocks
    ]
    return {"highlights": highlights, "timestamp": _now_iso()}


@app.post("/cache/flush", dependencies=[Depends(require_admin)])
async def cache_flush():
    """Invalidate all market quote caches so the next request fetches fresh data."""
    for key in ("crypto", "stocks"):
        cache.invalidate(key)
    return {"flushed": ["crypto", "stocks"]}


# HEAD as well as GET: uptime monitors default to HEAD because it is cheaper,
# and FastAPI — unlike plain Starlette — does not register it alongside GET, so
# a GET-only route answers 405 and the monitor reports the service as down.
@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    from .ingestion import get_status as ingestion_status

    sched_jobs = []
    if _scheduler and _scheduler.running:
        sched_jobs = [
            {
                "id": j.id,
                "next_run": j.next_run_time.isoformat() if j.next_run_time else None,
            }
            for j in _scheduler.get_jobs()
        ]
    agent_ok, agent_model = await llm.health()

    return {
        "status": "ok",
        "cache": cache.stats(),
        "chroma": chroma_stats(),
        "scheduler": {
            "running": bool(_scheduler and _scheduler.running),
            "jobs": sched_jobs,
        },
        "ingestion": ingestion_status(),
        "agent": {
            "available": agent_ok,
            "model": agent_model,
            "provider": llm.provider(),
        },
        "keys": {
            "coingecko": bool(settings.COINGECKO_API_KEY),
            "finnhub": bool(settings.FINNHUB_API_KEY),
            "newsapi": bool(settings.NEWSAPI_KEY),
        },
    }


# ── §A: Persisted Market Data Endpoints ──────────────────────────────────────


@app.get("/data/snapshots")
async def data_snapshots(limit: int = Query(50, ge=1, le=500)):
    """Latest N price snapshots from SQLite (all symbols, newest first)."""
    rows = await get_latest_snapshots(limit)
    return {"snapshots": rows, "count": len(rows), "timestamp": _now_iso()}


@app.get("/data/history/{symbol}")
async def data_history(
    symbol: str,
    limit: int = Query(100, ge=1, le=1000),
):
    """Price history for a specific symbol from SQLite."""
    rows = await get_symbol_history(symbol.upper(), limit)
    return {"symbol": symbol.upper(), "history": rows, "count": len(rows), "timestamp": _now_iso()}


@app.get("/data/ohlcv/{symbol}")
async def data_ohlcv(
    symbol: str,
    days: int = Query(30, ge=1, le=365),
):
    """30-day daily OHLCV from SQLite (persisted by scheduler)."""
    rows = await get_ohlcv(symbol.upper(), days)
    return {"symbol": symbol.upper(), "ohlcv": rows, "count": len(rows), "timestamp": _now_iso()}


@app.get("/data/news")
async def data_news_cached(limit: int = Query(20, ge=1, le=100)):
    """Recent news articles from SQLite."""
    rows = await get_recent_news(limit)
    return {"articles": rows, "count": len(rows), "timestamp": _now_iso()}


# ── Prediction agent (Layer 2: calibrated direction + conformal band + regime) ────
@app.get("/predict/leaderboard")
async def predict_leaderboard(limit: int = Query(50, ge=1, le=200)):
    """Latest stored prediction per symbol, ranked by calibrated confidence."""
    rows = await get_latest_predictions(None, limit)
    return {"leaderboard": rows, "count": len(rows), "timestamp": _now_iso()}


@app.get("/predict/calibration")
async def predict_calibration(model: str = "live"):
    """Realized hit-rate per confidence bucket — the reliability curve behind the gauge."""
    rows = await get_calibration_buckets(model)
    return {"model": model, "buckets": rows, "timestamp": _now_iso()}


@app.post("/predict/run", dependencies=[Depends(require_admin)])
async def predict_run(symbol: str | None = Query(None)):
    """Generate + persist predictions now (one symbol or all). Also resolves matured ones."""
    from .prediction.serve import resolve_due, run_predictions

    resolved = await resolve_due()
    preds = await run_predictions([symbol.upper()] if symbol else None)
    return {
        "resolved": resolved,
        "logged": len(preds),
        "predictions": preds,
        "timestamp": _now_iso(),
    }


@app.get("/predict/verdicts")
async def predict_verdicts(limit: int = Query(50, ge=1, le=200)):
    """Latest verifier verdict per symbol — feeds leaderboard VETO/downgrade badges."""
    from .db import get_latest_verdicts

    rows = await get_latest_verdicts(limit)
    return {
        "verdicts": [_parse_verdict(r) for r in rows],
        "count": len(rows),
        "timestamp": _now_iso(),
    }


@app.get("/predict/{symbol}")
async def predict_symbol(
    symbol: str, fresh: bool = Query(False), x_admin_token: str | None = Header(default=None)
):
    """
    Latest calibrated prediction for a symbol. By default returns the most recent STORED
    prediction (fast); `fresh=true` recomputes live and logs it (admin only: it runs the
    model on a GET, which crawlers and link previews would otherwise trigger).
    """
    symbol = symbol.upper()
    if fresh and not is_admin(x_admin_token):
        raise HTTPException(status_code=403, detail="fresh=true requires the admin token")
    if fresh:
        from .prediction.serve import predict_and_log

        p = await predict_and_log(symbol)
        if not p:
            raise HTTPException(status_code=404, detail=f"No prediction available for {symbol}")
        return {"symbol": symbol, "prediction": p, "fresh": True, "timestamp": _now_iso()}
    rows = await get_latest_predictions(symbol, 1)
    if not rows:
        raise HTTPException(
            status_code=404, detail=f"No stored prediction for {symbol}; call with ?fresh=true"
        )
    return {"symbol": symbol, "prediction": rows[0], "fresh": False, "timestamp": _now_iso()}


RANGE_HIT_WINDOW = 60  # most recent matured predictions the range hit rate is measured over


@app.get("/predict/{symbol}/history")
async def predict_symbol_history(symbol: str, limit: int = Query(100, ge=1, le=500)):
    """A symbol's past predictions joined with realized outcomes (accuracy over time)."""
    rows = await get_prediction_history(symbol.upper(), limit)
    resolved = [r for r in rows if r.get("correct") is not None]
    acc = round(sum(r["correct"] for r in resolved) / len(resolved), 4) if resolved else None
    # Live range hit rate: share of matured predictions whose close landed inside the 80% range.
    banded = [r["in_band"] for r in resolved if r.get("in_band") is not None][:RANGE_HIT_WINDOW]
    band_hit = round(sum(banded) / len(banded), 4) if banded else None
    return {
        "symbol": symbol.upper(),
        "history": rows,
        "count": len(rows),
        "resolved": len(resolved),
        "realized_accuracy": acc,
        "band_hit_rate": band_hit,
        "band_resolved": len(banded),
        "timestamp": _now_iso(),
    }


_forecast_locks: dict[str, asyncio.Lock] = {}


def _forecast_lock(symbol: str) -> asyncio.Lock:
    return _forecast_locks.setdefault(symbol, asyncio.Lock())


@lru_cache(maxsize=1)
def _point_forecast_quality() -> dict:
    """Out-of-fold point-forecast record of the shipped return model (model_meta.json)."""
    try:
        from .prediction.point_eval import point_forecast_has_skill

        meta = json.loads(
            (Path(__file__).parent / "prediction" / "models" / "model_meta.json").read_text()
        )
        rep = meta.get("conformal_report") or {}
        mae, naive = rep.get("mae"), rep.get("mae_predict_zero")
        skill = round(1 - mae / naive, 4) if mae and naive else None
        return {
            "mae": mae,
            "mae_no_change": naive,
            "skill": skill,
            "has_skill": point_forecast_has_skill(rep),
            "coverage": rep.get("coverage", {}),
        }
    except Exception as exc:
        log.warning("point-forecast record unavailable: %s", exc)
        return {"has_skill": False, "skill": None}


@lru_cache(maxsize=1)
def _model_symbols() -> frozenset[str]:
    """Symbols the live model was trained on (`fd_orders` in model_meta.json)."""
    try:
        meta = json.loads(
            (Path(__file__).parent / "prediction" / "models" / "model_meta.json").read_text()
        )
        return frozenset(meta.get("fd_orders", {}))
    except Exception as exc:
        log.warning("model_meta.json unreadable, forecast recompute disabled: %s", exc)
        return frozenset()


@app.get("/predict/{symbol}/forecast")
async def predict_symbol_forecast(symbol: str, lookback: int = Query(60, ge=10, le=400)):
    """
    Chart-ready forecast bundle for the advisor: the recent daily CLOSE series (the actual-price
    line) plus the latest calibrated prediction (the projected point + conformal band), sourced
    from the SAME `ohlcv_history` series the model anchors on, so the actual line and the
    prediction point are guaranteed consistent. Falls back to a fresh prediction if none stored.
    """
    from .db import get_history

    symbol = symbol.upper()

    rows = await get_history(symbol)  # full series, oldest→newest
    if not rows:
        raise HTTPException(status_code=404, detail=f"No price history for {symbol}")
    tail = rows[-lookback:]
    series = [
        {"date": r["date"], "close": float(r.get("adj_close") or r.get("close"))}
        for r in tail
        if (r.get("adj_close") or r.get("close")) is not None
    ]

    # Reuse today's stored prediction if one exists (one fresh inference per symbol per day);
    # only recompute when stale or missing, so dashboard reloads don't hammer the model.
    stored = await get_latest_predictions(symbol, 1)
    pred = stored[0] if stored else None
    today = datetime.now(timezone.utc).date()
    is_fresh = (
        pred
        and pred.get("generated_at")
        and datetime.fromtimestamp(pred["generated_at"] / 1000, tz=timezone.utc).date() == today
    )
    # Recompute only for symbols the model was trained on, and only one at a
    # time per symbol: concurrent dashboard loads share a single inference.
    lock = _forecast_lock(symbol)
    if not is_fresh and symbol in _model_symbols() and not lock.locked():
        async with lock:
            try:
                from .prediction.serve import predict_and_log

                fresh = await predict_and_log(symbol)
                if fresh:
                    fresh["generated_at"] = int(time.time() * 1000)
                    pred = fresh
            except Exception as exc:  # agent/model offline → fall back to stale/none
                log.warning("forecast predict failed for %s: %s", symbol, exc)

    return {
        "symbol": symbol,
        "series": series,
        "count": len(series),
        "prediction": pred,
        "point_forecast": _point_forecast_quality(),
        "timestamp": _now_iso(),
    }


@app.post("/predict/{symbol}/verify", dependencies=[Depends(require_user), _AI_LIMIT])
async def predict_symbol_verify(symbol: str):
    """
    Run the LLM verifier (Layer 3) over the calibrated signal: it explains the call and may
    DOWNGRADE or VETO it against fresh news/RAG — never raise confidence above the calibrated
    number. Degrades gracefully (verifier='unavailable') if Ollama is down.
    """
    from .prediction.agent import verify_prediction

    v = await verify_prediction(symbol.upper())
    if not v:
        raise HTTPException(status_code=404, detail=f"No prediction available for {symbol}")
    return {"symbol": symbol.upper(), "verdict": v, "timestamp": _now_iso()}


# Parses the free-text `ai_insights.content` string written by verify_prediction(), e.g.
# "BTC — model says UP @ 70% (regime trend). Verifier: VETO → final 35%. <rationale> Risk: <risks>"
_VERDICT_RE = re.compile(
    r"model says (?P<direction>\w+) @ (?P<model_conf>\d+)% \(regime (?P<regime>[^)]*)\)\.\s*"
    r"Verifier:\s*(?P<label>VETO|agree|downgrade)\s*→\s*final\s*(?P<final_conf>\d+)%\.\s*"
    r"(?P<rationale>.*?)\s*Risk:\s*(?P<risks>.*)$",
    re.S,
)


def _parse_verdict(row: dict) -> dict:
    out = {
        "symbol": row["symbol"],
        "label": None,
        "model_confidence": None,
        "final_confidence": row.get("confidence"),
        "direction": None,
        "regime": None,
        "rationale": None,
        "risks": None,
        "generated_at": row.get("generated_at"),
    }
    m = _VERDICT_RE.search(row.get("content") or "")
    if m:
        out.update(
            {
                "direction": m.group("direction"),
                "model_confidence": int(m.group("model_conf")),
                "regime": m.group("regime"),
                "label": m.group("label"),
                "final_confidence": int(m.group("final_conf")),
                "rationale": m.group("rationale").strip(),
                "risks": m.group("risks").strip(),
            }
        )
    return out


@app.get("/predict/{symbol}/verdict")
async def predict_symbol_verdict(symbol: str):
    """
    Latest persisted verifier verdict for a symbol (logged during the daily prediction cycle),
    parsed into structured fields. `verdict: null` if the verifier hasn't run for this symbol yet.
    """
    from .db import get_latest_verdict

    row = await get_latest_verdict(symbol.upper())
    return {
        "symbol": symbol.upper(),
        "verdict": _parse_verdict(row) if row else None,
        "timestamp": _now_iso(),
    }


@app.get("/ingestion/status")
async def ingestion_status_endpoint():
    """Scheduler status and per-job last-run metadata."""
    from .ingestion import get_status

    jobs = []
    if _scheduler and _scheduler.running:
        jobs = [
            {
                "id": j.id,
                "next_run": j.next_run_time.isoformat() if j.next_run_time else None,
            }
            for j in _scheduler.get_jobs()
        ]
    log_rows = await get_ingestion_log(20)
    return {
        "scheduler_running": bool(_scheduler and _scheduler.running),
        "job_status": get_status(),
        "scheduled_jobs": jobs,
        "recent_log": log_rows,
        "timestamp": _now_iso(),
    }


@app.post("/ingestion/trigger/{job}", dependencies=[Depends(require_admin)])
async def ingestion_trigger(job: str):
    """
    Manually trigger a specific ingestion job immediately.
    job: crypto | stocks | ohlcv | news | market | insights | predictions

    predictions is the manual lever for the heavy FLUX-X cycle, which no longer
    runs itself on startup (HEAVY_JOBS_ON_STARTUP) — on a 512 MB host that run
    was the difference between one OOM kill and a restart loop.
    """
    from .ingestion import (
        full_market_cycle,
        ingest_crypto,
        ingest_news,
        ingest_ohlcv,
        ingest_stocks,
    )

    async def _predictions():
        from .prediction.flux_x import run_flux_x

        try:
            res = await run_flux_x()
            log.info("manual prediction cycle: %s", res)
        except Exception as exc:
            log.warning("manual prediction cycle failed: %s", exc)

    job_map = {
        "crypto": ingest_crypto,
        "stocks": ingest_stocks,
        "ohlcv": ingest_ohlcv,
        "news": ingest_news,
        "market": full_market_cycle,
        "insights": _insight_job,
        "predictions": _predictions,
    }
    if job not in job_map:
        raise HTTPException(400, f"Unknown job '{job}'. Valid: {list(job_map)}")
    asyncio.create_task(job_map[job]())
    return {"triggered": job, "timestamp": _now_iso()}


# ── §B: AI Insights Endpoints ─────────────────────────────────────────────────


@app.get("/ai/insights")
async def ai_insights_list(
    limit: int = Query(20, ge=1, le=100),
    symbol: str = Query("", description="Filter by symbol"),
):
    """Latest pre-computed AI insights from SQLite."""
    rows = await get_latest_insights(
        limit=limit,
        symbol=symbol.upper() if symbol else None,
    )
    return {"insights": rows, "count": len(rows), "timestamp": _now_iso()}


@app.post("/ai/insights/refresh", dependencies=[Depends(require_admin)])
async def ai_insights_refresh():
    """Trigger an immediate AI insight generation cycle (non-blocking)."""
    all_assets = _cached_assets()
    if not all_assets:
        raise HTTPException(503, "No asset data in cache yet — wait for the first ingestion cycle")

    asyncio.create_task(_insight_job())
    return {
        "status": "refresh_started",
        "assets": len(all_assets),
        "timestamp": _now_iso(),
    }


# ── §C: RAG Query Endpoint ────────────────────────────────────────────────────


class RagQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    include_context: bool = False


@app.post("/ai/rag/query", dependencies=[Depends(require_user), _AI_LIMIT])
async def ai_rag_query(body: RagQueryRequest):
    """
    RAG-enhanced financial Q&A.
    Retrieves relevant context from ChromaDB, injects it into the Ollama
    system prompt, and returns a grounded answer.
    """
    context, n_chunks = rag_query(body.query)

    system_content = (
        "You are AURA, a professional AI financial advisor. "
        "Answer with data-driven precision. Keep responses under 150 words.\n\n"
    )
    if context:
        system_content += f"=== LIVE MARKET CONTEXT (retrieved) ===\n{context}\n=== END ==="

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": body.query},
    ]
    try:
        answer = await llm.chat(messages, timeout=45.0)
    except llm.LLMError as exc:
        log.error("RAG chat failed: %s", exc)
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        log.error("RAG chat failed: %s", exc)
        raise HTTPException(502, "AI unavailable") from exc

    resp: dict = {
        "answer": answer,
        "rag_available": bool(context),
        "context_chunks": n_chunks,
        "timestamp": _now_iso(),
    }
    if body.include_context:
        resp["context"] = context
    return resp


@app.get("/market/quotes/crypto")
async def quotes_crypto():
    """
    Returns top-15 cryptocurrencies from CoinGecko.
    Cached for CRYPTO_TTL seconds (default 30s).
    """
    cached_data = cache.get("crypto")
    if cached_data is not None:
        return _wrap(cached_data, "coingecko_cache", cached=True)

    try:
        assets = await _fetch_crypto_live()
    except httpx.HTTPStatusError as e:
        log.error("CoinGecko HTTP error: %s", e)
        raise HTTPException(502, f"CoinGecko error: {e.response.status_code}") from e
    except Exception as e:
        log.error("CoinGecko fetch failed: %s", e)
        raise HTTPException(502, "CoinGecko unreachable") from e

    cache.set("crypto", assets, ttl=settings.CRYPTO_TTL)
    return _wrap(assets, "coingecko_live", cached=False)


@app.get("/market/quotes/stocks")
async def quotes_stocks():
    """
    Returns 15 curated equities from Finnhub.
    Cached for STOCKS_TTL seconds (default 60s).
    """
    cached_data = cache.get("stocks")
    if cached_data is not None:
        return _wrap(cached_data, "finnhub_cache", cached=True)

    if not settings.FINNHUB_API_KEY:
        raise HTTPException(503, "FINNHUB_API_KEY not configured")

    try:
        assets = await _fetch_stocks_live()
    except Exception as e:
        log.error("Finnhub fetch failed: %s", e)
        raise HTTPException(502, "Finnhub unreachable") from e

    cache.set("stocks", assets, ttl=settings.STOCKS_TTL)
    return _wrap(assets, "finnhub_live", cached=False)


# ── Portfolio Mark-to-Market Valuation ───────────────────────────────────────


# yfinance ticker per holding: NSE equities/ETFs trade as <SYM>.NS in INR;
# crypto trades as <SYM>-USD and needs the USDINR rate applied.
def _holding_yf_symbol(h: dict) -> str:
    if h.get("asset_type") == "crypto":
        return f"{h['symbol']}-USD"
    if h.get("currency") == "USD":  # US-listed, e.g. SPY
        return h["symbol"]
    return f"{h['symbol']}.NS"


def _holding_needs_fx(h: dict) -> bool:
    """Crypto and US-listed holdings are quoted in USD and converted to INR."""
    return h.get("asset_type") == "crypto" or h.get("currency") == "USD"


def _fetch_last_prices_sync(symbols: list[str]) -> dict[str, float]:
    """Blocking yfinance batch quote — last close per symbol. Symbols that
    fail to resolve are simply absent from the result."""
    out: dict[str, float] = {}
    if not symbols:
        return out
    df = yf.download(
        symbols,
        period="5d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    for sym in symbols:
        try:
            closes = (df[sym]["Close"] if len(symbols) > 1 else df["Close"]).dropna()
            if len(closes):
                out[sym] = float(closes.iloc[-1])
        except Exception:  # noqa: BLE001 — unresolvable symbol → skip
            continue
    return out


@app.get("/portfolio/value")
async def portfolio_value(user_id: int = Depends(require_user)):
    """Mark-to-market portfolio valuation in INR.

    Joins the user's holdings with live last prices (yfinance batch) and the
    live USDINR rate for crypto. Returns per-holding market value, cost basis
    and unrealised P&L. Cached for 60s per user.
    """
    cache_key = f"pf_value_{user_id}"
    cached = cache.get(cache_key)
    if cached is not None:
        return {**cached, "cached": True}

    from . import mysql_db as M

    holdings = M.query("SELECT * FROM portfolio_holdings WHERE user_id=%s", (user_id,))
    if not holdings:
        return {
            "holdings": [],
            "total_value_inr": 0,
            "total_cost_inr": 0,
            "pnl_inr": 0,
            "pnl_pct": 0,
            "fx_usdinr": None,
            "priced": 0,
            "unpriced": [],
            "cached": False,
            "timestamp": _now_iso(),
        }

    loop = asyncio.get_event_loop()
    yf_symbols = sorted({_holding_yf_symbol(h) for h in holdings})
    needs_fx = any(_holding_needs_fx(h) for h in holdings)
    fetch_list = yf_symbols + (["USDINR=X"] if needs_fx else [])
    prices = await loop.run_in_executor(None, _fetch_last_prices_sync, fetch_list)

    fx = prices.get("USDINR=X")
    if needs_fx and not fx:
        # Without FX, pricing crypto in INR would be silently wrong — keep the
        # response honest by leaving those holdings unpriced instead.
        log.warning("portfolio_value: USDINR rate unavailable; USD holdings unpriced")

    rows, unpriced = [], []
    total_value = total_cost = 0.0
    for h in holdings:
        qty = float(h["quantity"])
        cost = qty * float(h["avg_price"])  # cost basis stored in INR
        yf_sym = _holding_yf_symbol(h)
        last = prices.get(yf_sym)
        is_usd = _holding_needs_fx(h)
        if last is not None and (not is_usd or fx):
            value = qty * last * (fx if is_usd else 1.0)
            total_value += value
            total_cost += cost
            rows.append(
                {
                    "symbol": h["symbol"],
                    "name": h["name"],
                    "asset_type": h["asset_type"],
                    "quantity": qty,
                    "avg_price": float(h["avg_price"]),
                    "last_price_inr": round(last * (fx if is_usd else 1.0), 2),
                    "value_inr": round(value, 2),
                    "cost_inr": round(cost, 2),
                    "pnl_pct": round((value - cost) / cost * 100, 2) if cost else 0,
                }
            )
        else:
            unpriced.append(h["symbol"])

    for r in rows:
        r["weight_pct"] = round(r["value_inr"] / total_value * 100, 2) if total_value else 0

    result = {
        "holdings": rows,
        "total_value_inr": round(total_value, 2),
        "total_cost_inr": round(total_cost, 2),
        "pnl_inr": round(total_value - total_cost, 2),
        "pnl_pct": round((total_value - total_cost) / total_cost * 100, 2) if total_cost else 0,
        "fx_usdinr": round(fx, 4) if fx else None,
        "priced": len(rows),
        "unpriced": unpriced,
        "timestamp": _now_iso(),
    }
    cache.set(cache_key, result, ttl=60)
    return {**result, "cached": False}


# ── Candles Endpoint (yfinance) ───────────────────────────────────────────────

# Maps frontend asset key → yfinance ticker symbol
_CANDLE_SYMBOL_MAP: dict[str, str] = {
    "btc": "BTC-USD",
    "eth": "ETH-USD",
    "spy": "SPY",
}

# Maps frontend timeframe → (period, interval) for yfinance
_TF_PARAMS: dict[str, tuple[str, str]] = {
    "1D": ("1d", "5m"),
    "7D": ("7d", "1h"),
    "1M": ("1mo", "1d"),
    "1Y": ("1y", "1wk"),
}

# Candle cache TTLs (seconds)
_CANDLE_TTL: dict[str, int] = {
    "1D": 60,
    "7D": 300,
    "1M": 600,
    "1Y": 3600,
}


def _fetch_candles_sync(yf_symbol: str, period: str, interval: str) -> list[dict]:
    """Blocking yfinance download — must be called via run_in_executor."""
    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        return []
    candles = []
    for ts, row in df.iterrows():
        candles.append(
            {
                "t": int(ts.timestamp() * 1000),
                "o": round(float(row["Open"]), 4),
                "h": round(float(row["High"]), 4),
                "l": round(float(row["Low"]), 4),
                "c": round(float(row["Close"]), 4),
                "v": int(row.get("Volume", 0) or 0),
            }
        )
    return candles


@app.get("/market/candles/{asset}")
async def market_candles(
    asset: str,
    tf: str = Query("1D", regex="^(1D|7D|1M|1Y)$"),
):
    """
    Returns OHLCV candles for a given asset and timeframe.

    asset: btc | eth | spy
    tf:    1D | 7D | 1M | 1Y
    """
    asset_lower = asset.lower()
    yf_symbol = _CANDLE_SYMBOL_MAP.get(asset_lower)
    if yf_symbol is None:
        raise HTTPException(404, f"Unknown asset '{asset}'. Supported: {list(_CANDLE_SYMBOL_MAP)}")

    cache_key = f"candles:{asset_lower}:{tf}"
    cached = cache.get(cache_key)
    if cached is not None:
        return {
            "symbol": asset_lower,
            "tf": tf,
            "candles": cached,
            "cached": True,
            "timestamp": _now_iso(),
        }

    period, interval = _TF_PARAMS[tf]
    try:
        loop = asyncio.get_event_loop()
        candles = await loop.run_in_executor(None, _fetch_candles_sync, yf_symbol, period, interval)
    except Exception as e:
        log.error("yfinance %s %s/%s failed: %s", yf_symbol, period, interval, e)
        raise HTTPException(502, f"yfinance fetch failed: {e}") from e

    if not candles:
        raise HTTPException(404, f"No data returned for {asset_lower} / {tf}")

    ttl = _CANDLE_TTL.get(tf, 300)
    cache.set(cache_key, candles, ttl=ttl)
    log.info("yfinance %s %s/%s: %d candles", yf_symbol, period, interval, len(candles))
    return {
        "symbol": asset_lower,
        "tf": tf,
        "candles": candles,
        "cached": False,
        "timestamp": _now_iso(),
    }


# ── Indicators Endpoint ───────────────────────────────────────────────────────


def _normalize_to_yf(symbol: str) -> str:
    """Map any frontend ticker string to a yfinance symbol."""
    s = symbol.upper().strip()
    # Strip exchange prefixes
    for pfx in ("BINANCE:", "NASDAQ:", "NYSE:", "AMEX:"):
        if s.startswith(pfx):
            s = s[len(pfx) :]
            break
    # Crypto XYZUSDT / XYZUSD → XYZ-USD
    if s.endswith("USDT"):
        return s[:-4] + "-USD"
    if s.endswith("USD") and len(s) > 3 and not s[:-3].isdigit():
        return s[:-3] + "-USD"
    # Stock tickers pass through unchanged
    return s


def _compute_indicators_sync(yf_symbol: str) -> dict:
    """Blocking: fetch 60 days of daily data and compute Vol, RSI(14), MACD."""

    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(period="60d", interval="1d", auto_adjust=True)
    if df.empty or len(df) < 27:
        return {}

    closes = df["Close"].tolist()
    volumes = df["Volume"].tolist()

    # Volume: mean of last 14 days
    vol = int(sum(volumes[-14:]) / 14) if volumes else 0

    # RSI(14) — Wilder smoothing
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains[:14]) / 14
    avg_loss = sum(losses[:14]) / 14
    for i in range(14, len(gains)):
        avg_gain = (avg_gain * 13 + gains[i]) / 14
        avg_loss = (avg_loss * 13 + losses[i]) / 14
    rsi = 100.0 if avg_loss == 0 else round(100 - 100 / (1 + avg_gain / avg_loss), 2)

    # MACD(12, 26, 9)
    def ema(prices: list, period: int) -> list:
        k = 2 / (period + 1)
        result = [prices[0]]
        for p in prices[1:]:
            result.append(p * k + result[-1] * (1 - k))
        return result

    ema12 = ema(closes, 12)
    ema26 = ema(closes, 26)
    macd_line = [ema12[i] - ema26[i] for i in range(len(closes))]
    signal = ema(macd_line, 9)
    macd_val = round(macd_line[-1], 4)
    signal_val = round(signal[-1], 4)

    return {
        "volume": vol,
        "rsi": rsi,
        "macd": macd_val,
        "macd_signal": signal_val,
        "macd_hist": round(macd_val - signal_val, 4),
    }


@app.get("/market/indicators/{symbol}")
async def market_indicators(symbol: str):
    """
    Returns Vol (14d avg), RSI(14), and MACD(12,26,9) for a given symbol.
    symbol: any frontend ticker — e.g. BINANCE:BTCUSDT, NVDA, AAPL, BTC-USD
    """
    yf_symbol = _normalize_to_yf(symbol)
    cache_key = f"indicators:{yf_symbol}"
    cached = cache.get(cache_key)
    if cached is not None:
        return {"symbol": yf_symbol, "indicators": cached, "cached": True, "timestamp": _now_iso()}

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _compute_indicators_sync, yf_symbol)
    except Exception as e:
        log.error("indicators %s failed: %s", yf_symbol, e)
        raise HTTPException(502, f"yfinance failed: {e}") from e

    if not result:
        raise HTTPException(404, f"Insufficient data for {yf_symbol}")

    cache.set(cache_key, result, ttl=300)
    log.info(
        "indicators %s: RSI=%.1f MACD=%.4f", yf_symbol, result.get("rsi", 0), result.get("macd", 0)
    )
    return {"symbol": yf_symbol, "indicators": result, "cached": False, "timestamp": _now_iso()}


# ── News Fetcher ──────────────────────────────────────────────────────────────

# Finance/crypto domains — keeps general headlines on-topic
_FINANCE_DOMAINS = (
    "coindesk.com,cointelegraph.com,reuters.com,bloomberg.com,"
    "cnbc.com,marketwatch.com,wsj.com,ft.com,investing.com,"
    "cryptonews.com,decrypt.co,theblock.co,financialtimes.com"
)

# General market query used for the scrolling headline ticker
_GENERAL_QUERY = (
    'bitcoin OR ethereum OR cryptocurrency OR "stock market" OR '
    '"Federal Reserve" OR "interest rate" OR "S&P 500" OR NASDAQ'
)

# Maps common asset names/symbols to tighter search terms
_ASSET_QUERY_MAP: dict[str, str] = {
    "bitcoin": "Bitcoin BTC",
    "ethereum": "Ethereum ETH",
    "solana": "Solana SOL",
    "bnb": "Binance BNB",
    "ripple": "Ripple XRP",
    "dogecoin": "Dogecoin DOGE",
    "cardano": "Cardano ADA",
    "nvidia": "NVIDIA NVDA",
    "apple": "Apple AAPL",
    "microsoft": "Microsoft MSFT",
    "tesla": "Tesla TSLA",
    "amazon": "Amazon AMZN",
    "meta": "Meta Platforms META",
    "alphabet": "Alphabet Google GOOGL",
}


async def _fetch_news_live(query: str, page_size: int = 15) -> list[dict]:
    """Fetch articles from NewsAPI /v2/everything."""
    params: dict[str, Any] = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": settings.NEWSAPI_KEY,
        "domains": _FINANCE_DOMAINS,
    }
    r = await get_client().get(
        "https://newsapi.org/v2/everything",
        params=params,
    )
    r.raise_for_status()
    data = r.json()

    articles = []
    for item in data.get("articles", []):
        title = (item.get("title") or "").strip()
        if not title or title == "[Removed]":
            continue
        articles.append(
            {
                "title": title,
                "source": item.get("source", {}).get("name", ""),
                "url": item.get("url", ""),
                "published_at": item.get("publishedAt", ""),
                "summary": (item.get("description") or "")[:200].strip(),
            }
        )
    log.info("NewsAPI '%s': fetched %d articles", query[:40], len(articles))
    return articles


@app.get("/market/news")
async def market_news(q: str = ""):
    """
    Returns finance headlines from NewsAPI.

    - No  ?q  →  general market headlines (scrolling ticker)
    - ?q=bitcoin  →  asset-specific news for the intel drawer

    Cached per query string for NEWS_TTL seconds (default 300s).
    """
    if not settings.NEWSAPI_KEY:
        raise HTTPException(503, "NEWSAPI_KEY not configured")

    # Normalise: strip BINANCE: prefix, lower-case
    term = q.strip().lower().removeprefix("binance:").removesuffix("usdt")
    cache_key = f"news:{term or '__general__'}"

    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return {"articles": cached_data, "query": term, "cached": True, "timestamp": _now_iso()}

    # Resolve the best search query for this term
    if not term:
        search_query = _GENERAL_QUERY
        page_size = 20
    else:
        search_query = _ASSET_QUERY_MAP.get(term, q.strip())
        page_size = 8

    try:
        articles = await _fetch_news_live(search_query, page_size)
    except httpx.HTTPStatusError as e:
        log.error("NewsAPI HTTP error: %s", e)
        raise HTTPException(502, f"NewsAPI error: {e.response.status_code}") from e
    except Exception as e:
        log.error("NewsAPI fetch failed: %s", e)
        raise HTTPException(502, "NewsAPI unreachable") from e

    cache.set(cache_key, articles, ttl=settings.NEWS_TTL)
    return {"articles": articles, "query": term, "cached": False, "timestamp": _now_iso()}


# ── AI Chat (Ollama) ──────────────────────────────────────────────────────────


class _ChatMsg(BaseModel):
    role: str = Field(..., max_length=20)
    content: str = Field(..., max_length=20_000)


class ChatRequest(BaseModel):
    messages: list[_ChatMsg] = Field(..., min_length=1, max_length=50)


@app.get("/ai/intel/{symbol}", dependencies=[Depends(require_user), _AI_LIMIT])
async def ai_intel(symbol: str):
    """
    Returns structured AI analysis for an asset.
    Injects live price/change/mcap context and instructs Ollama to reply in JSON.
    Response: { consensus, confidence, vol_profile, report, catalysts[] }
    """
    # Resolve live asset data from cache (best-effort)
    sym_upper = symbol.upper()
    asset_data: dict | None = None
    for cache_key in ("crypto", "stocks"):
        pool = cache.get(cache_key)
        if pool:
            for a in pool:
                if (
                    a.get("sub", "").upper() == sym_upper
                    or a.get("symbol", "").upper() == sym_upper
                ):
                    asset_data = a
                    break
        if asset_data:
            break

    if asset_data:
        price = asset_data.get("price", 0)
        change_pct = asset_data.get("change_pct", 0)
        name = asset_data.get("name", symbol)
        mcap = asset_data.get("market_cap", 0)
        direction = "bullish" if change_pct >= 0 else "bearish"
        context_line = (
            f"{name} ({sym_upper}) is trading at ${price:,.2f}, "
            f"{'+' if change_pct >= 0 else ''}{change_pct:.2f}% in the last 24h "
            + (f"with a market cap of ${mcap / 1e9:.1f}B. " if mcap else ". ")
            + f"Sentiment direction: {direction}."
        )
    else:
        name = symbol
        context_line = f"Asset: {symbol}. No live price data available."

    prompt = (
        f"You are a professional market analyst. Analyze the following asset and respond ONLY with valid JSON — "
        f"no markdown, no explanation, no code fences.\n\n"
        f"Market data: {context_line}\n\n"
        f"Respond with exactly this JSON schema:\n"
        f'{{"consensus": "<Bullish|Bearish|Neutral> (<percent>%)", '
        f'"confidence": <integer 0-100>, '
        f'"vol_profile": "<short phrase describing volume/flow>", '
        f'"report": "<2-3 sentence market analysis>", '
        f'"catalysts": ["<tag1>", "<tag2>", "<tag3>"]}}'
    )

    try:
        raw = llm.strip_fences(await llm.chat([{"role": "user", "content": prompt}], timeout=30.0))

        import json as _json

        intel = _json.loads(raw)
        intel.setdefault("symbol", symbol)
        intel.setdefault("name", name)
        intel.setdefault("available", True)
        return intel
    except Exception as e:
        # Honest degraded response — we do NOT fabricate a buy/sell signal when the
        # model is offline or returns junk. The frontend renders an "Unavailable"
        # state from available:false; live price/headlines remain real.
        log.warning("ai_intel unavailable for %s: %s", symbol, e)
        return {
            "symbol": symbol,
            "name": name,
            "available": False,
            "consensus": None,
            "confidence": None,
            "vol_profile": None,
            "report": "AI analysis is temporarily unavailable.",
            "catalysts": [],
        }


@app.post("/ai/chat/stream", dependencies=[Depends(require_user), _AI_LIMIT])
async def ai_chat_stream(body: ChatRequest):
    """
    Streaming version of /ai/chat — returns Server-Sent Events.
    Each SSE data line is JSON: {"content": "<token>"}.
    Final line is: data: [DONE]
    """
    import json as _json

    from fastapi.responses import StreamingResponse

    messages = [{"role": m.role, "content": m.content} for m in body.messages]

    async def generate():
        try:
            async for token in llm.stream_chat(messages, timeout=60.0):
                yield f"data: {_json.dumps({'content': token})}\n\n"
            # The provider's own terminator is consumed by the token generator,
            # so the sentinel the client waits on is emitted here.
            yield "data: [DONE]\n\n"
        except llm.LLMError as exc:
            yield f"data: {_json.dumps({'error': str(exc)})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            log.error("Streaming chat error: %s", e)
            yield f"data: {_json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Backtesting (yfinance SMA crossover) ─────────────────────────────────────


class BacktestRequest(BaseModel):
    symbol: str = Field(..., pattern=r"^[A-Za-z0-9^.\-=]{1,15}$")  # e.g. "BTC-USD", "AAPL", "^GSPC"
    start: date  # "YYYY-MM-DD"
    end: date  # "YYYY-MM-DD"
    short_sma: int = Field(20, ge=2, le=400)
    long_sma: int = Field(50, ge=2, le=400)
    initial_capital: float = Field(100000.0, gt=0, le=1e9)


def _run_backtest_sync(
    symbol: str, start: str, end: str, short_sma: int, long_sma: int, initial_capital: float
) -> dict:
    import pandas as pd

    # Map friendly tickers
    _map = {"BTC": "BTC-USD", "ETH": "ETH-USD"}
    yf_sym = _map.get(symbol.upper(), symbol)

    df = yf.download(yf_sym, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {yf_sym} in range {start}–{end}")

    # yfinance returns MultiIndex columns (field, ticker) even for a single
    # symbol — flatten so df["Close"]/row["Close"] etc. are plain scalars.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].squeeze()
    df["sma_s"] = close.rolling(short_sma).mean()
    df["sma_l"] = close.rolling(long_sma).mean()
    df = df.dropna()

    # Signal: 1 = long, 0 = out
    df["signal"] = (df["sma_s"] > df["sma_l"]).astype(int)
    df["pos_chg"] = df["signal"].diff()

    capital = initial_capital
    shares = 0.0
    trades = []
    equity_curve = []

    for ts, row in df.iterrows():
        price = float(row["Close"])
        if row["pos_chg"] == 1 and capital > 0:  # buy
            shares = capital / price
            capital = 0.0
            trades.append(
                {
                    "date": str(ts.date()),
                    "action": "BUY",
                    "price": round(price, 4),
                    "shares": round(shares, 6),
                }
            )
        elif row["pos_chg"] == -1 and shares > 0:  # sell
            capital = shares * price
            pnl = (
                capital - initial_capital
                if not trades
                else capital - (trades[-1]["price"] * shares)
            )
            trades.append(
                {
                    "date": str(ts.date()),
                    "action": "SELL",
                    "price": round(price, 4),
                    "shares": round(shares, 6),
                    "pnl": round(pnl, 2),
                }
            )
            shares = 0.0

        total_value = capital + shares * price
        equity_curve.append({"t": int(ts.timestamp() * 1000), "v": round(total_value, 2)})

    # Final portfolio value
    last_price = float(df["Close"].iloc[-1])
    final_value = capital + shares * last_price
    total_return = (final_value - initial_capital) / initial_capital * 100

    # Daily returns for Sharpe
    eq_vals = [p["v"] for p in equity_curve]
    if len(eq_vals) > 1:
        daily_rets = [
            (eq_vals[i] - eq_vals[i - 1]) / eq_vals[i - 1] for i in range(1, len(eq_vals))
        ]
        avg_r = sum(daily_rets) / len(daily_rets)
        std_r = (sum((r - avg_r) ** 2 for r in daily_rets) / len(daily_rets)) ** 0.5
        sharpe = round((avg_r / std_r * (252**0.5)) if std_r > 0 else 0, 3)
        # Max drawdown
        peak = eq_vals[0]
        max_dd = 0.0
        for v in eq_vals:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100
            if dd > max_dd:
                max_dd = dd
    else:
        sharpe = 0
        max_dd = 0.0

    sell_trades = [t for t in trades if t["action"] == "SELL"]
    win_rate = round(
        sum(1 for t in sell_trades if t.get("pnl", 0) > 0) / max(len(sell_trades), 1) * 100, 1
    )

    return {
        "symbol": yf_sym,
        "start": start,
        "end": end,
        "short_sma": short_sma,
        "long_sma": long_sma,
        "initial_capital": initial_capital,
        "final_value": round(final_value, 2),
        "metrics": {
            "total_return": round(total_return, 2),
            "sharpe": sharpe,
            "max_drawdown": round(max_dd, 2),
            "win_rate": win_rate,
            "total_trades": len(trades),
        },
        "trades": trades[-20:],  # last 20 to keep payload small
        "equity_curve": equity_curve,
    }


@app.post("/backtest", dependencies=[Depends(require_user)])
async def run_backtest(body: BacktestRequest):
    """
    Run an SMA crossover backtest via yfinance.

    Body: { symbol, start, end, short_sma, long_sma, initial_capital }
    Returns: { metrics, equity_curve, trades }
    """
    if body.short_sma >= body.long_sma:
        raise HTTPException(400, "short_sma must be less than long_sma")
    if body.start >= body.end:
        raise HTTPException(400, "start must be before end")
    if (body.end - body.start).days > 3653:
        raise HTTPException(400, "Date range is limited to 10 years")
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            _run_backtest_sync,
            body.symbol,
            body.start.isoformat(),
            body.end.isoformat(),
            body.short_sma,
            body.long_sma,
            body.initial_capital,
        )
        return result
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
    except Exception as e:
        log.error("Backtest failed: %s", e)
        raise HTTPException(502, f"Backtest error: {e}") from e


@app.post("/ai/chat", dependencies=[Depends(require_user), _AI_LIMIT])
async def ai_chat(body: ChatRequest):
    """
    Forward a chat request to the configured LLM provider and return the reply.

    Provider is resolved in backend/llm.py: a hosted OpenAI-compatible endpoint
    when LLM_API_KEY is set, otherwise the local Ollama instance (OLLAMA_MODEL).
    """
    try:
        content = await llm.chat(
            [{"role": m.role, "content": m.content} for m in body.messages],
            timeout=30.0,
        )
        return {"content": content}
    except llm.LLMError as e:
        log.error("AI chat failed: %s", e)
        raise HTTPException(503, str(e)) from e
    except Exception as e:
        log.error("AI chat failed: %s", e)
        raise HTTPException(502, "AI unavailable") from e
