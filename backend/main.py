"""
Agentic AI Finance & Stock Prediction System — FastAPI entry point.

Routes live in backend/routes/ and the other *_api modules; see docs/API_REFERENCE.md.

Start
-----
    uvicorn backend.main:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from . import llm
from .auth import auth_router, ensure_auth_schema
from .cache import cache
from .config import settings
from .db import init_db
from .payments_api import payments_router
from .rag import chroma_stats, init_chroma
from .routes import ai, backtest, common, ingestion, market, predict
from .trading_api import trading_router
from .user_api import db_router

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
# httpx logs every request URL at INFO, and Finnhub/NewsAPI take the API key as a
# query parameter, so those lines would put the keys in the Render logs.
for _noisy in ("httpx", "httpcore"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
log = logging.getLogger("flux.api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await _startup()
    yield
    await _shutdown()


async def _startup() -> None:
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

    common.scheduler = build_scheduler(insight_job_fn=common._insight_job)
    common.scheduler.start()
    log.info("APScheduler started — %d jobs registered", len(common.scheduler.get_jobs()))


async def _shutdown() -> None:
    sched = common.scheduler
    if sched and sched.running:
        sched.shutdown(wait=False)
        log.info("APScheduler stopped")
    await common.close_client()
    from .ingestion import close_client as _close_ingest_client

    await _close_ingest_client()
    from .mysql_db import close_pool

    close_pool()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Agentic AI Finance & Stock Prediction System API",
    version="1.0.0",
    description="Market data, predictions and paper trading for the Agentic AI Finance & Stock Prediction System.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],  # DELETE: watchlist + alerts
    # Only what the frontend sends, plus the admin header for manual triggers.
    allow_headers=["Authorization", "Content-Type", "X-Admin-Token"],
    expose_headers=["Retry-After"],  # lets the frontend read 429 back-off
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
# Market data, predictions, ingestion, AI and backtest (see backend/routes/)
for _r in (market, predict, ingestion, ai, backtest):
    app.include_router(_r.router)


# HEAD as well as GET: uptime monitors default to HEAD because it is cheaper,
# and FastAPI — unlike plain Starlette — does not register it alongside GET, so
# a GET-only route answers 405 and the monitor reports the service as down.
@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    from .ingestion import get_status as ingestion_status

    sched_jobs = []
    if common.scheduler and common.scheduler.running:
        sched_jobs = [
            {
                "id": j.id,
                "next_run": j.next_run_time.isoformat() if j.next_run_time else None,
            }
            for j in common.scheduler.get_jobs()
        ]
    agent_ok, agent_model = await llm.health()

    return {
        "status": "ok",
        "cache": cache.stats(),
        "chroma": chroma_stats(),
        "scheduler": {
            "running": bool(common.scheduler and common.scheduler.running),
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
