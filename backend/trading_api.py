"""
FLUX — Marketplace trading, watchlist & alerts API (/db/*)
=========================================================

The marketplace is a **paper-trading** surface: every account gets a virtual
USD buying-power wallet (seeded once at $100,000). Orders are spot buy/sell —
no leverage, no margin, no liquidation theatre. Positions are derived from the
immutable `trades` ledger so the wallet and the ledger can never disagree.

This is deliberately separate from `portfolio_holdings` / `/portfolio/value`,
which is the user's *real* INR portfolio (NSE tickers). Mixing the marketplace's
USD paper trades into that would corrupt the INR mark-to-market.

Tables (created by backend/migrations/003_trading_tables.sql)
-------------------------------------------------------
  trading_wallet   user_id PK, cash_usd, seeded_usd, updated_at
  watchlist        (user_id, symbol) unique, category, name
  market_alerts    user_id, symbol, direction(above|below), target_price,
                   active, triggered_at

Routes (all auth-scoped via require_user)
-----------------------------------------
  GET    /db/wallet                 — cash + mark-to-market positions + equity
  POST   /db/trades                 — execute a spot buy/sell (validated, filled
                                       at the server's live quote, not the client's)
  GET    /db/watchlist              — user's watched symbols
  POST   /db/watchlist              — add {symbol, category, name}
  DELETE /db/watchlist/{symbol}     — remove
  GET    /db/alerts                 — user's price alerts
  POST   /db/alerts                 — create {symbol, name, direction, target_price}
  POST   /db/alerts/{id}/triggered  — mark an alert as fired (client-detected)
  DELETE /db/alerts/{id}            — remove
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import anyio
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from . import mysql_db as M
from .auth import require_user
from .cache import cache

log = logging.getLogger("flux.trading")

trading_router = APIRouter(prefix="/db", tags=["trading"])

SEED_CASH_USD = 100_000.0
_QTY_EPS = 1e-9  # treat smaller residual positions as flat
# How far a client's price may sit on the wrong side of the live quote before
# the order is refused. Orders always fill at the server's quote; the client
# price only expresses the worst price the user will accept (a limit).
PRICE_TOLERANCE = 0.01


# ── Helpers ──────────────────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_wallet(user_id: int) -> dict:
    """Return the user's wallet, creating it (seeded) on first access."""
    row = M.query_one("SELECT * FROM trading_wallet WHERE user_id=%s", (user_id,))
    if row:
        return row
    with M.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO trading_wallet (user_id, cash_usd, seeded_usd) VALUES (%s, %s, %s)",
            (user_id, SEED_CASH_USD, SEED_CASH_USD),
        )
    return M.query_one("SELECT * FROM trading_wallet WHERE user_id=%s", (user_id,))


def _live_price(symbol: str) -> float | None:
    """Best-effort last price from the market cache (crypto + stocks pools).
    Matches on the marketplace `symbol` (e.g. BINANCE:BTCUSDT, AAPL) or the
    short `sub` ticker (BTC)."""
    s = (symbol or "").upper()
    for key in ("crypto", "stocks"):
        pool = cache.get(key)
        if not pool:
            continue
        for a in pool:
            if a.get("symbol", "").upper() == s or a.get("sub", "").upper() == s:
                try:
                    return float(a.get("price") or 0) or None
                except (TypeError, ValueError):
                    return None
    return None


def _refresh_quote_pool(asset_type: str) -> None:
    """Refill an expired crypto/stocks quote cache from upstream.

    Called from a sync route (FastAPI runs those in a worker thread), so the
    async fetchers run on the main event loop via anyio. Imported lazily to
    avoid an import cycle with the routers."""
    from .config import settings
    from .routes import market

    try:
        if asset_type == "crypto":
            if cache.get("crypto") is None:
                cache.set(
                    "crypto",
                    anyio.from_thread.run(market._fetch_crypto_live),
                    ttl=settings.CRYPTO_TTL,
                )
        elif cache.get("stocks") is None and settings.FINNHUB_API_KEY:
            cache.set(
                "stocks", anyio.from_thread.run(market._fetch_stocks_live), ttl=settings.STOCKS_TTL
            )
    except Exception as e:  # upstream down: caller treats it as "no price"
        log.warning("quote refresh for %s failed: %s", asset_type, e)


def _reference_price(symbol: str, asset_type: str) -> float | None:
    """Server-side execution price: the cached quote, refreshed once if stale."""
    price = _live_price(symbol)
    if price is None:
        _refresh_quote_pool(asset_type)
        price = _live_price(symbol)
    return price


def _check_client_price(side: str, client_price: float, market: float) -> None:
    """Refuse an order whose client price can't be met at the live quote.

    A BUY asking below market, or a SELL asking above it, by more than
    PRICE_TOLERANCE is the old exploit (buy at $0.01, sell at market)."""
    if side == "BUY" and client_price < market * (1 - PRICE_TOLERANCE):
        raise HTTPException(
            409,
            f"Price moved: live price is ${market:,.4f}, above your ${client_price:,.4f}. Refresh and retry.",
        )
    if side == "SELL" and client_price > market * (1 + PRICE_TOLERANCE):
        raise HTTPException(
            409,
            f"Price moved: live price is ${market:,.4f}, below your ${client_price:,.4f}. Refresh and retry.",
        )


def _held_quantity(cur, user_id: int, symbol: str) -> float:
    """Net quantity held, read on the caller's (locked) transaction."""
    cur.execute("SELECT symbol, side, quantity FROM trades WHERE user_id=%s", (user_id,))
    held = 0.0
    for r in cur.fetchall():
        if r["symbol"].upper() == symbol.upper():
            held += float(r["quantity"]) * (1 if r["side"] == "BUY" else -1)
    return held


def _positions(user_id: int) -> list[dict]:
    """Derive current positions from the trade ledger (average-cost basis)."""
    rows = M.query(
        "SELECT symbol, name, asset_type, side, quantity, price, amount "
        "FROM trades WHERE user_id=%s ORDER BY trade_date ASC",
        (user_id,),
    )
    acc: dict[str, dict] = {}
    for r in rows:
        sym = r["symbol"]
        p = acc.setdefault(
            sym,
            {
                "symbol": sym,
                "name": r.get("name") or sym,
                "asset_type": r.get("asset_type") or "stocks",
                "qty": 0.0,
                "cost": 0.0,
            },
        )
        qty = float(r["quantity"])
        price = float(r["price"])
        if r["side"] == "BUY":
            p["qty"] += qty
            p["cost"] += qty * price
        else:  # SELL — reduce qty and cost proportionally (avg-cost)
            if p["qty"] > _QTY_EPS:
                avg = p["cost"] / p["qty"]
                p["cost"] -= min(qty, p["qty"]) * avg
            p["qty"] -= qty
        if r.get("name"):
            p["name"] = r["name"]

    out = []
    for p in acc.values():
        if p["qty"] <= _QTY_EPS:
            continue
        avg_price = p["cost"] / p["qty"] if p["qty"] else 0
        last = _live_price(p["symbol"])
        value = (last or avg_price) * p["qty"]
        cost = p["cost"]
        out.append(
            {
                "symbol": p["symbol"],
                "name": p["name"],
                "asset_type": p["asset_type"],
                "quantity": round(p["qty"], 8),
                "avg_price": round(avg_price, 4),
                "last_price": round(last, 4) if last is not None else None,
                "value_usd": round(value, 2),
                "cost_usd": round(cost, 2),
                "pnl_usd": round(value - cost, 2),
                "pnl_pct": round((value - cost) / cost * 100, 2) if cost else 0,
                "priced": last is not None,
            }
        )
    out.sort(key=lambda x: x["value_usd"], reverse=True)
    return out


def _wallet_payload(user_id: int) -> dict:
    wallet = _ensure_wallet(user_id)
    cash = float(wallet["cash_usd"])
    positions = _positions(user_id)
    holdings_value = sum(p["value_usd"] for p in positions)
    cost_basis = sum(p["cost_usd"] for p in positions)
    return {
        "cash_usd": round(cash, 2),
        "seeded_usd": round(float(wallet["seeded_usd"]), 2),
        "holdings_value_usd": round(holdings_value, 2),
        "equity_usd": round(cash + holdings_value, 2),
        "unrealized_pnl_usd": round(holdings_value - cost_basis, 2),
        "positions": positions,
        "timestamp": _now().isoformat(),
    }


# ── Wallet ───────────────────────────────────────────────────────────────────


@trading_router.get("/wallet")
def get_wallet(user_id: int = Depends(require_user)) -> dict:
    """Virtual USD buying power + mark-to-market positions + total equity."""
    return _wallet_payload(user_id)


# ── Trades ───────────────────────────────────────────────────────────────────


class TradeRequest(BaseModel):
    symbol: str = Field(..., max_length=40)
    name: str | None = Field(None, max_length=160)
    side: str  # BUY | SELL
    price: float = Field(..., gt=0)  # client's quoted price (USD); a limit, not the fill
    quantity: float | None = Field(None, gt=0)
    amount: float | None = Field(None, gt=0)  # notional USD; used if quantity omitted
    asset_type: str = "stocks"  # stocks | crypto


@trading_router.post("/trades")
def create_trade(body: TradeRequest, user_id: int = Depends(require_user)) -> dict:
    """Execute a validated spot paper trade and update the wallet atomically.

    The fill price is the server's live quote; the client price is only
    checked against it. The wallet row is locked (SELECT ... FOR UPDATE) for
    the whole check-and-write, so parallel orders can't both spend the same
    cash or sell the same units."""
    side = body.side.upper()
    if side not in ("BUY", "SELL"):
        raise HTTPException(400, "side must be BUY or SELL")

    asset_type = "crypto" if body.asset_type.lower() == "crypto" else "stocks"
    price = _reference_price(body.symbol, asset_type)
    if price is None:
        raise HTTPException(
            503, f"Live price unavailable for {body.symbol}. Try again in a moment."
        )
    _check_client_price(side, float(body.price), price)

    # Resolve quantity from explicit qty or from a notional USD amount.
    if body.quantity is not None:
        qty = float(body.quantity)
    elif body.amount is not None:
        qty = float(body.amount) / price
    else:
        raise HTTPException(400, "Provide quantity or amount")
    if qty <= _QTY_EPS:
        raise HTTPException(400, "Order size too small")

    notional = round(qty * price, 2)
    _ensure_wallet(user_id)
    name = (body.name or body.symbol)[:160]

    with M.get_conn() as conn, conn.cursor() as cur:
        try:
            conn.begin()
            cur.execute(
                "SELECT cash_usd FROM trading_wallet WHERE user_id=%s FOR UPDATE",
                (user_id,),
            )
            cash = float(cur.fetchone()["cash_usd"])
            if side == "BUY":
                if notional > cash + 0.005:
                    raise HTTPException(
                        400,
                        f"Insufficient buying power: order ${notional:,.2f} exceeds "
                        f"available ${cash:,.2f}",
                    )
                new_cash = cash - notional
            else:  # SELL — can't sell more than held
                held_qty = _held_quantity(cur, user_id, body.symbol)
                if qty > held_qty + _QTY_EPS:
                    raise HTTPException(
                        400,
                        f"Insufficient position: trying to sell {qty:g} but only {max(held_qty, 0):g} held",
                    )
                new_cash = cash + notional

            cur.execute(
                "INSERT INTO trades (user_id, symbol, name, side, quantity, price, amount, "
                "currency, asset_type, trade_date) VALUES (%s,%s,%s,%s,%s,%s,%s,'USD',%s,%s)",
                (user_id, body.symbol, name, side, qty, price, notional, asset_type, _now()),
            )
            trade_id = cur.lastrowid
            cur.execute(
                "UPDATE trading_wallet SET cash_usd=%s, updated_at=%s WHERE user_id=%s",
                (new_cash, _now(), user_id),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    log.info(
        "paper trade #%d: user=%d %s %g %s @ %.4f", trade_id, user_id, side, qty, body.symbol, price
    )
    payload = _wallet_payload(user_id)
    position = next(
        (p for p in payload["positions"] if p["symbol"].upper() == body.symbol.upper()), None
    )
    return {
        "ok": True,
        "trade": {
            "id": trade_id,
            "symbol": body.symbol,
            "name": name,
            "side": side,
            "quantity": round(qty, 8),
            "price": round(price, 4),
            "amount": notional,
            "asset_type": asset_type,
            "trade_date": _now().isoformat(),
        },
        "position": position,
        "wallet": payload,
    }


# ── Watchlist ────────────────────────────────────────────────────────────────


class WatchRequest(BaseModel):
    symbol: str = Field(..., max_length=40)
    category: str | None = Field(None, max_length=20)
    name: str | None = Field(None, max_length=160)


@trading_router.get("/watchlist")
def get_watchlist(user_id: int = Depends(require_user)) -> dict:
    rows = M.query(
        "SELECT symbol, category, name FROM watchlist WHERE user_id=%s ORDER BY created_at",
        (user_id,),
    )
    return {"watchlist": rows, "count": len(rows)}


@trading_router.post("/watchlist")
def add_watchlist(body: WatchRequest, user_id: int = Depends(require_user)) -> dict:
    with M.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO watchlist (user_id, symbol, category, name) VALUES (%s,%s,%s,%s) "
            "ON DUPLICATE KEY UPDATE category=VALUES(category), name=VALUES(name)",
            (user_id, body.symbol, body.category, body.name),
        )
    return get_watchlist(user_id)


@trading_router.delete("/watchlist/{symbol:path}")
def remove_watchlist(symbol: str, user_id: int = Depends(require_user)) -> dict:
    with M.get_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM watchlist WHERE user_id=%s AND symbol=%s", (user_id, symbol))
    return get_watchlist(user_id)


# ── Alerts ───────────────────────────────────────────────────────────────────


class AlertRequest(BaseModel):
    symbol: str = Field(..., max_length=40)
    name: str | None = Field(None, max_length=160)
    direction: str = "above"  # above | below
    target_price: float = Field(..., gt=0)


@trading_router.get("/alerts")
def get_alerts(user_id: int = Depends(require_user)) -> dict:
    rows = M.query(
        "SELECT id, symbol, name, direction, target_price, active, created_at, triggered_at "
        "FROM market_alerts WHERE user_id=%s ORDER BY active DESC, created_at DESC",
        (user_id,),
    )
    return {"alerts": rows, "count": len(rows)}


@trading_router.post("/alerts")
def create_alert(body: AlertRequest, user_id: int = Depends(require_user)) -> dict:
    direction = body.direction.lower()
    if direction not in ("above", "below"):
        raise HTTPException(400, "direction must be 'above' or 'below'")
    with M.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO market_alerts (user_id, symbol, name, direction, target_price) "
            "VALUES (%s,%s,%s,%s,%s)",
            (user_id, body.symbol, body.name, direction, body.target_price),
        )
    return get_alerts(user_id)


@trading_router.post("/alerts/{alert_id}/triggered")
def mark_alert_triggered(alert_id: int, user_id: int = Depends(require_user)) -> dict:
    """Client detected the threshold crossing — record it and deactivate."""
    with M.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE market_alerts SET active=0, triggered_at=%s WHERE id=%s AND user_id=%s",
            (_now(), alert_id, user_id),
        )
    return get_alerts(user_id)


@trading_router.delete("/alerts/{alert_id}")
def delete_alert(alert_id: int, user_id: int = Depends(require_user)) -> dict:
    with M.get_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM market_alerts WHERE id=%s AND user_id=%s", (alert_id, user_id))
    return get_alerts(user_id)
