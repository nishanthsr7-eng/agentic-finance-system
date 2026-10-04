r"""
FLUX Prediction — Earnings-Window Gating (Phase 3.1)
====================================================
Earnings releases are the biggest single-name volatility events: a 5-day directional model has
no edge across an earnings gap (the move is driven by the surprise, not by the prior trend), and
the realized vol can dwarf the conformal band. So when a prediction's horizon straddles an
upcoming earnings date we DOWN-WEIGHT confidence and size — optionally abstain entirely.

Source: Finnhub `/calendar/earnings` (the FINNHUB_API_KEY is already configured). Crypto and
index symbols have no earnings → the gate is a clean no-op. If the endpoint is unavailable
(free-tier limit / network), we fail OPEN (no gate) and log, never crash the prediction.

Security:
  • Symbols are validated to ^[A-Z0-9.\-]{1,12}$ before going into the request (no injection).
  • The API key is read from settings, sent only to Finnhub over https, and never logged/returned.
  • Responses are size-bounded and parsed defensively.

Public API:
    await days_to_earnings(symbol, as_of=None)            -> int | None
    await earnings_gate(symbol, horizon, as_of, window)   -> dict {in_window, days, factor}
    apply_gate(prediction, gate)                          -> prediction (confidence/kelly scaled)
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta

import httpx

from ..config import settings

log = logging.getLogger("flux.prediction.earnings")

_SYMBOL_RE = re.compile(r"^[A-Z0-9.\-]{1,12}$")
_GATE_FACTOR = 0.5  # confidence/size multiplier when inside the earnings window
_CACHE: dict[str, tuple[float, list[str]]] = {}  # symbol -> (fetched_at, [earnings dates])
_CACHE_TTL = 6 * 3600  # earnings dates move slowly; cache 6h to spare the free tier


def _valid_symbol(symbol: str) -> bool:
    return bool(_SYMBOL_RE.match(symbol or ""))


async def _fetch_earnings_dates(symbol: str) -> list[str]:
    """Upcoming earnings dates (YYYY-MM-DD) for a symbol from Finnhub. [] on any failure."""
    import time as _t

    now = _t.time()
    if symbol in _CACHE and now - _CACHE[symbol][0] < _CACHE_TTL:
        return _CACHE[symbol][1]
    if not settings.FINNHUB_API_KEY or not _valid_symbol(symbol):
        return []
    frm = date.today().isoformat()
    to = (date.today() + timedelta(days=90)).isoformat()
    dates: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=12.0) as cli:
            r = await cli.get(
                "https://finnhub.io/api/v1/calendar/earnings",
                params={
                    "from": frm,
                    "to": to,
                    "symbol": symbol,
                    "token": settings.FINNHUB_API_KEY,
                },
            )
            r.raise_for_status()
            payload = r.json()
        for e in (payload.get("earningsCalendar") or [])[:20]:
            d = e.get("date")
            if d:
                dates.append(str(d))
    except Exception as exc:  # fail OPEN — no gate, never crash
        log.warning("earnings fetch for %s failed (gate disabled): %s", symbol, exc)
        return []
    dates = sorted(set(dates))
    _CACHE[symbol] = (now, dates)
    return dates


async def days_to_earnings(symbol: str, as_of: str | None = None) -> int | None:
    """Business-agnostic calendar days from `as_of` to the next earnings date (None if unknown)."""
    symbol = (symbol or "").upper()
    ref = datetime.fromisoformat(as_of).date() if as_of else date.today()
    future = [datetime.fromisoformat(d).date() for d in await _fetch_earnings_dates(symbol)]
    future = [d for d in future if d >= ref]
    return (min(future) - ref).days if future else None


async def earnings_gate(
    symbol: str, horizon: int = 5, as_of: str | None = None, window: int | None = None
) -> dict:
    """
    Decide whether a prediction's horizon straddles earnings.
        in_window : True if the next earnings falls within `window` days (default = horizon+1)
        factor    : confidence/size multiplier (1.0 normally, _GATE_FACTOR inside the window)
    """
    window = window if window is not None else horizon + 1
    dte = await days_to_earnings(symbol, as_of)
    in_window = dte is not None and 0 <= dte <= window
    return {
        "in_window": bool(in_window),
        "days_to_earnings": dte,
        "factor": _GATE_FACTOR if in_window else 1.0,
        "window": window,
    }


def apply_gate(prediction: dict, gate: dict) -> dict:
    """Scale a prediction's confidence + kelly by the earnings factor (in place-safe copy)."""
    if not gate.get("in_window"):
        return {
            **prediction,
            "earnings_soon": False,
            "days_to_earnings": gate.get("days_to_earnings"),
        }
    f = gate["factor"]
    p = dict(prediction)
    p["confidence"] = int(round(p.get("confidence", 0) * f))
    if p.get("kelly_frac") is not None:
        p["kelly_frac"] = round(p["kelly_frac"] * f, 4)
    p["earnings_soon"] = True
    p["days_to_earnings"] = gate.get("days_to_earnings")
    return p


if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    async def _demo():
        for sym in ("AAPL", "NVDA", "JPM", "BTC", "bad;symbol"):
            g = await earnings_gate(sym, horizon=5)
            print(
                f"  {sym:12} days_to_earnings={g['days_to_earnings']}  "
                f"in_window={g['in_window']}  factor={g['factor']}"
            )

    asyncio.run(_demo())
