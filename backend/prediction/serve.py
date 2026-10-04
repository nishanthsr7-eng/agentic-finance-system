"""
FLUX Prediction — Serving & Persistence (the data flywheel)
===========================================================
`predict.py` produces a calibrated prediction in memory; this module makes it *live*: it logs
every prediction, later resolves it against the realized price, and recomputes the displayed
confidence from real outcomes. That loop is what keeps "confidence 62%" honest over time — the
single property that sets this agent apart from demo predictors.

Flow:
  predict_and_log(sym)   → run the model, write a row to `predictions`, return it (+ db id)
  run_predictions()      → log all symbols, return a confidence-ranked leaderboard
  resolve_due()          → for predictions whose horizon has elapsed, compute the realized
                           h-day return from `ohlcv_history`, write `prediction_outcomes`
                           (incl. `in_band`: did the close land inside the 80% range)
  refresh_live_calibration() → bucket resolved outcomes by stated confidence → calibration_buckets

Honesty note: the live outcome is the realized return over the prediction horizon (target_date
vs last_close). We don't replay the intra-horizon triple-barrier path live — the fixed-horizon
return is a simpler, fully auditable correctness measure for calibration.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from ..db import (
    get_due_predictions,
    get_history,
    get_resolved_outcomes,
    insert_outcome,
    insert_prediction,
    upsert_calibration,
)
from .predict import predict, predict_all

log = logging.getLogger("flux.prediction.serve")

COST = 0.0005  # 5 bps round-trip cost + slippage (matches backtest.py)
MODEL_TAG = "xgb_primary+meta+sentiment+conformal"


def _to_row(p: dict) -> dict:
    """Map a predict() dict to the `predictions` table schema."""
    return {
        "symbol": p["symbol"],
        "model": MODEL_TAG,
        "horizon_days": p["horizon_days"],
        "direction": p["direction"],
        "prob_up": p["prob_up"],
        "meta_prob": p["meta_prob"],
        "act": int(bool(p["act"])),
        "confidence": p["confidence"],
        "kelly_frac": p["kelly_frac"],
        "sentiment": p.get("sentiment"),
        "last_close": p["last_close"],
        "pred_return": p.get("pred_return"),
        "pred_price": p.get("pred_price"),
        "conf_low": p.get("conf_low"),
        "conf_high": p.get("conf_high"),
        "conf_low_90": p.get("conf_low_90"),
        "conf_high_90": p.get("conf_high_90"),
        "regime": p.get("regime"),
        "generated_at": int(time.time() * 1000),
        "target_date": p["target_date"],
        "iv_atm": p.get("iv_atm"),
        "iv_skew": p.get("iv_skew"),
    }


async def predict_and_log(symbol: str, **kw) -> dict | None:
    """Predict for one symbol and persist it. Returns the prediction dict with its db `id`."""
    p = await predict(symbol, **kw)
    if not p:
        return None
    pid = await insert_prediction(_to_row(p))
    return {**p, "id": pid}


async def run_predictions(symbols: list[str] | None = None, **kw) -> list[dict]:
    """Predict + log every symbol; return a confidence-ranked leaderboard."""
    preds = await predict_all(symbols) if not kw else None
    out = []
    if preds is not None:
        for p in preds:  # predict_all already ran the models
            try:
                pid = await insert_prediction(_to_row(p))
                out.append({**p, "id": pid})
            except Exception as exc:
                log.warning("log %s failed: %s", p.get("symbol"), exc)
    else:
        from ..db import history_summary
        from .train import EXCLUDE

        syms = symbols or [
            r["symbol"] for r in await history_summary() if r["symbol"] not in EXCLUDE
        ]
        for s in syms:
            r = await predict_and_log(s, **kw)
            if r:
                out.append(r)
    out.sort(key=lambda r: r["confidence"], reverse=True)
    log.info("Logged %d predictions", len(out))
    return out


async def _close_on_or_after(symbol: str, date: str) -> float | None:
    """First adjusted close on/after `date` from history (None if not yet available)."""
    rows = await get_history(symbol, date)
    if not rows:
        return None
    px = rows[0].get("adj_close") or rows[0].get("close")
    return float(px) if px is not None else None


async def resolve_due(as_of: str | None = None) -> int:
    """
    Resolve every prediction whose target_date has passed. Realized h-day return = close at
    target_date / last_close − 1. Records direction-correctness + cost-aware P&L. Returns count.
    """
    as_of = as_of or str(pd.Timestamp.utcnow().date())
    due = await get_due_predictions(as_of)
    resolved = 0
    for p in due:
        future_close = await _close_on_or_after(p["symbol"], p["target_date"])
        if future_close is None or not p.get("last_close"):
            continue  # bar not in history yet → leave pending
        actual = future_close / float(p["last_close"]) - 1.0
        correct = int((p["direction"] == "UP") == (actual > 0))
        side = 1 if p["direction"] == "UP" else -1
        pnl = (side * actual - COST) if p.get("act") else 0.0
        lo, hi = p.get("conf_low"), p.get("conf_high")
        in_band = (
            int(float(lo) <= future_close <= float(hi))
            if lo is not None and hi is not None
            else None
        )
        await insert_outcome(
            {
                "prediction_id": p["id"],
                "actual_return": round(actual, 6),
                "correct": correct,
                "pnl_after_costs": round(pnl, 6),
                "in_band": in_band,
                "resolved_at": int(time.time() * 1000),
            }
        )
        resolved += 1
    if resolved:
        log.info("Resolved %d predictions (as_of %s)", resolved, as_of)
        await refresh_live_calibration()
    return resolved


async def refresh_live_calibration() -> int:
    """Bucket resolved outcomes by stated confidence decile → calibration_buckets(model='live')."""
    rows = await get_resolved_outcomes()
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    df["bucket"] = (df["confidence"].clip(0, 99) // 10).astype(int)
    ts = int(time.time() * 1000)
    out = []
    for b, g in df.groupby("bucket"):
        out.append(
            {
                "model": "live",
                "bucket": int(b),
                "stated_conf": (b + 0.5) / 10,
                "realized_hit": float(g["correct"].mean()),
                "n": int(len(g)),
                "updated_at": ts,
            }
        )
    await upsert_calibration(out)
    log.info("Refreshed %d live calibration buckets from %d outcomes", len(out), len(df))
    return len(out)


async def daily_prediction_cycle(verify_top_k: int = 5) -> dict:
    """
    Scheduler entrypoint (the data flywheel turn): resolve matured predictions → refresh live
    calibration, log a fresh batch, then RED-TEAM the top-k portfolio names through the Layer-3
    verifier (Phase 8). The verifier can only downgrade/veto and persists an auditable rationale;
    it degrades gracefully if the LLM is unavailable, so the cycle never hard-depends on it.
    """
    from .regime import invalidate_regime_cache

    invalidate_regime_cache()  # refit the HMM once for this cycle, then
    n_resolved = await resolve_due()  # reuse the cached regime across all symbols
    preds = await run_predictions()

    verified = 0
    if verify_top_k:
        try:
            from .agent import verify_portfolio

            verdicts = await verify_portfolio(preds, top_k=verify_top_k, persist=True)
            verified = sum(
                1 for v in verdicts if v.get("verifier") not in ("unavailable", "parse_error")
            )
        except Exception as exc:  # the LLM layer must never break the flywheel
            log.warning("portfolio verify skipped: %s", exc)
    return {"resolved": n_resolved, "logged": len(preds), "verified": verified}


if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    async def _demo():
        from backend.db import init_db

        await init_db()
        # Log a small batch, then resolve anything already mature (backfilled-history symbols
        # will have the target bar, so this exercises the full loop offline).
        preds = await run_predictions(
            ["AAPL", "NVDA", "BTC"], with_sentiment=False, with_regime=False
        )
        for p in preds:
            print(
                f"  logged #{p['id']} {p['symbol']:5} {p['direction']:4} conf={p['confidence']} "
                f"band=[{p['conf_low']}, {p['conf_high']}]"
            )
        n = await resolve_due()
        print(f"resolved {n} matured predictions")

    asyncio.run(_demo())
