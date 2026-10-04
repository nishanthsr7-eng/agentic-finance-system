"""
FLUX Prediction — Alpaca Paper Forward-Test (Phase 8)
=====================================================
Backtests lie a little: real fills, spreads, and timing differ. The honest bridge before any
real capital is a PAPER account — route the live signal to Alpaca's paper endpoint and watch it
trade with realistic fills, risking nothing. This module turns `act=True` predictions into
paper orders sized by their fractional-Kelly weight.

SAFETY (deliberately conservative):
  • PAPER ONLY. The base URL is hard-pinned to the paper endpoint and we refuse to run unless
    settings.ALPACA_PAPER is True. There is no live-trading code path here at all.
  • DRY-RUN by default. `submit_orders(..., live_submit=False)` computes the orders but does NOT
    send them. You must pass live_submit=True explicitly to place paper orders.
  • No-op without keys (empty on this machine) → returns status='disabled'.
  • Keys are read from settings, sent only to Alpaca over https, never logged or returned.
  • Symbols validated; per-order notional is clamped to `MAX_NOTIONAL`.

Public API:
    await account()                              -> dict | None
    await submit_orders(preds, equity, live_submit=False) -> dict
"""

from __future__ import annotations

import logging
import re

import httpx

from ..config import settings

log = logging.getLogger("flux.prediction.paper")

_PAPER_BASE = "https://paper-api.alpaca.markets"  # hard-pinned: paper only, never live
_SYMBOL_RE = re.compile(r"^[A-Z0-9.\-]{1,12}$")
MAX_NOTIONAL = 10_000.0  # per-order clamp (paper safety rail)
DEFAULT_EQUITY = 100_000.0


def _enabled() -> bool:
    return bool(settings.ALPACA_API_KEY and settings.ALPACA_SECRET_KEY and settings.ALPACA_PAPER)


def _headers() -> dict:
    return {
        "APCA-API-KEY-ID": settings.ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": settings.ALPACA_SECRET_KEY,
    }


async def account() -> dict | None:
    """Paper account snapshot (equity, buying power). None if disabled/unreachable."""
    if not _enabled():
        return None
    try:
        async with httpx.AsyncClient(timeout=15.0, base_url=_PAPER_BASE) as cli:
            r = await cli.get("/v2/account", headers=_headers())
            r.raise_for_status()
            a = r.json()
        return {
            "equity": float(a.get("equity", 0)),
            "buying_power": float(a.get("buying_power", 0)),
            "status": a.get("status"),
        }
    except Exception as exc:
        log.warning("Alpaca account fetch failed: %s", exc)
        return None


def _plan_orders(preds: list[dict], equity: float) -> list[dict]:
    """Pure planner (no I/O): act=True signals → sized, validated, clamped paper orders."""
    orders = []
    for p in preds:
        if not p.get("act"):
            continue
        sym = str(p.get("symbol", "")).upper()
        if not _SYMBOL_RE.match(sym):
            continue
        kelly = float(p.get("kelly_frac") or 0.0)
        if kelly <= 0:
            continue
        notional = round(min(MAX_NOTIONAL, equity * kelly), 2)
        if notional < 1:
            continue
        orders.append(
            {
                "symbol": sym,
                "side": "buy" if p.get("direction") == "UP" else "sell",
                "notional": notional,
                "type": "market",
                "time_in_force": "day",
            }
        )
    return orders


async def submit_orders(
    preds: list[dict], equity: float | None = None, live_submit: bool = False
) -> dict:
    """
    Plan paper orders from predictions; submit them only if live_submit=True. Returns a summary
    including the planned orders (so it's useful as a dry-run preview).
    """
    if not _enabled():
        return {"status": "disabled", "reason": "no Alpaca paper keys", "orders": []}

    acct = await account()
    equity = equity or (acct["equity"] if acct else DEFAULT_EQUITY)
    orders = _plan_orders(preds, equity)
    if not live_submit:
        return {"status": "dry_run", "equity": equity, "planned": len(orders), "orders": orders}

    submitted, errors = [], []
    async with httpx.AsyncClient(timeout=20.0, base_url=_PAPER_BASE) as cli:
        for o in orders:
            try:
                r = await cli.post("/v2/orders", headers=_headers(), json=o)
                r.raise_for_status()
                submitted.append({"symbol": o["symbol"], "id": r.json().get("id")})
            except Exception as exc:
                errors.append({"symbol": o["symbol"], "error": str(exc)[:120]})
    log.info("Alpaca paper: submitted %d / %d orders", len(submitted), len(orders))
    return {"status": "submitted", "equity": equity, "submitted": submitted, "errors": errors}


if __name__ == "__main__":
    import asyncio

    async def _demo():
        sample = [
            {"symbol": "AAPL", "act": True, "direction": "UP", "kelly_frac": 0.05},
            {"symbol": "NVDA", "act": False, "direction": "UP", "kelly_frac": 0.10},
            {"symbol": "bad;x", "act": True, "direction": "UP", "kelly_frac": 0.05},
        ]
        # Planner is pure → testable even with no keys.
        print("planned orders (pure):", _plan_orders(sample, equity=100_000))
        print("submit_orders (no keys -> disabled):", await submit_orders(sample))

    asyncio.run(_demo())
