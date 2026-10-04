"""State and helpers shared by the route modules."""

import logging
from datetime import datetime, timezone

import httpx
from fastapi import Depends

from ..cache import cache
from ..config import settings
from ..rag import embed_market_snapshots
from ..ratelimit import rate_limit

log = logging.getLogger("flux.api")

# One shared per-IP allowance for every route that spends LLM quota.
_AI_LIMIT = Depends(rate_limit("ai", 20, 60))

scheduler = None  # APScheduler instance, set by the lifespan in backend.main

# ── HTTP Client ───────────────────────────────────────────────────────────────
# Shared client — reuses TCP connections across requests
_http_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=12.0, follow_redirects=True)
    return _http_client


async def close_client() -> None:
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()


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
    from ..insights import run_insight_cycle

    all_assets = _cached_assets()
    if not all_assets:
        log.debug("Insight job skipped — cache not warm yet")
        return
    generated = await run_insight_cycle(all_assets, settings.INSIGHT_MAX_ASSETS)
    # Also embed the freshest market snapshots into ChromaDB
    embed_market_snapshots(all_assets)
    log.info("Insight cycle complete: %d insights generated", len(generated))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _wrap(assets: list[dict], source: str, cached: bool) -> dict:
    return {"assets": assets, "source": source, "cached": cached, "timestamp": _now_iso()}
