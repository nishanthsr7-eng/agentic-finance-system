"""Prediction agent: calibrated direction, conformal band, regime, verifier."""

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from ..auth import is_admin, require_admin, require_user
from ..db import (
    get_calibration_buckets,
    get_latest_predictions,
    get_prediction_history,
)
from .common import _AI_LIMIT, _now_iso, log

router = APIRouter()


# ── Prediction agent (Layer 2: calibrated direction + conformal band + regime) ────
# Symbols dropped from the product. Old predictions for them may still sit in the
# database; they are kept there but never listed.
RETIRED_SYMBOLS = {"NIFTY"}


def _listed(rows):
    return [r for r in rows if str(r.get("symbol", "")).upper() not in RETIRED_SYMBOLS]


@router.get("/predict/leaderboard")
async def predict_leaderboard(limit: int = Query(50, ge=1, le=200)):
    """Latest stored prediction per symbol, ranked by calibrated confidence."""
    rows = _listed(await get_latest_predictions(None, limit))
    return {"leaderboard": rows, "count": len(rows), "timestamp": _now_iso()}


@router.get("/predict/calibration")
async def predict_calibration(model: str = "live"):
    """Realized hit-rate per confidence bucket — the reliability curve behind the gauge."""
    rows = await get_calibration_buckets(model)
    return {"model": model, "buckets": rows, "timestamp": _now_iso()}


@router.post("/predict/run", dependencies=[Depends(require_admin)])
async def predict_run(symbol: str | None = Query(None)):
    """Generate + persist predictions now (one symbol or all). Also resolves matured ones."""
    from ..prediction.serve import resolve_due, run_predictions

    resolved = await resolve_due()
    preds = await run_predictions([symbol.upper()] if symbol else None)
    return {
        "resolved": resolved,
        "logged": len(preds),
        "predictions": preds,
        "timestamp": _now_iso(),
    }


@router.get("/predict/verdicts")
async def predict_verdicts(limit: int = Query(50, ge=1, le=200)):
    """Latest verifier verdict per symbol — feeds leaderboard VETO/downgrade badges."""
    from ..db import get_latest_verdicts

    rows = _listed(await get_latest_verdicts(limit))
    return {
        "verdicts": [_parse_verdict(r) for r in rows],
        "count": len(rows),
        "timestamp": _now_iso(),
    }


@router.get("/predict/{symbol}")
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
        from ..prediction.serve import predict_and_log

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


@router.get("/predict/{symbol}/history")
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
        from ..prediction.point_eval import point_forecast_has_skill

        meta = json.loads(
            (Path(__file__).parent.parent / "prediction" / "models" / "model_meta.json").read_text()
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
            (Path(__file__).parent.parent / "prediction" / "models" / "model_meta.json").read_text()
        )
        return frozenset(meta.get("fd_orders", {}))
    except Exception as exc:
        log.warning("model_meta.json unreadable, forecast recompute disabled: %s", exc)
        return frozenset()


@router.get("/predict/{symbol}/forecast")
async def predict_symbol_forecast(symbol: str, lookback: int = Query(60, ge=10, le=400)):
    """
    Chart-ready forecast bundle for the advisor: the recent daily CLOSE series (the actual-price
    line) plus the latest calibrated prediction (the projected point + conformal band), sourced
    from the SAME `ohlcv_history` series the model anchors on, so the actual line and the
    prediction point are guaranteed consistent. Falls back to a fresh prediction if none stored.
    """
    from ..db import get_history

    symbol = symbol.upper()

    rows = await get_history(symbol)  # full series, oldest→newest
    if not rows:
        raise HTTPException(status_code=404, detail=f"No price history for {symbol}")
    tail = rows[-lookback:]
    series = []
    for r in tail:
        close = r.get("adj_close") or r.get("close")
        if close is None:
            continue
        point = {"date": r["date"], "close": float(close)}
        # Daily open/high/low on the same adjusted scale as `close`, so the
        # advisor can draw candles for symbols without a live candle feed.
        raw = r.get("close")
        if raw and all(r.get(k) is not None for k in ("open", "high", "low")):
            k = float(close) / float(raw)
            point.update(
                {
                    "open": float(r["open"]) * k,
                    "high": float(r["high"]) * k,
                    "low": float(r["low"]) * k,
                }
            )
        series.append(point)

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
                from ..prediction.serve import predict_and_log

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


@router.post("/predict/{symbol}/verify", dependencies=[Depends(require_user), _AI_LIMIT])
async def predict_symbol_verify(symbol: str):
    """
    Run the LLM verifier (Layer 3) over the calibrated signal: it explains the call and may
    DOWNGRADE or VETO it against fresh news/RAG — never raise confidence above the calibrated
    number. Degrades gracefully (verifier='unavailable') if Ollama is down.
    """
    from ..prediction.agent import verify_prediction

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


@router.get("/predict/{symbol}/verdict")
async def predict_symbol_verdict(symbol: str):
    """
    Latest persisted verifier verdict for a symbol (logged during the daily prediction cycle),
    parsed into structured fields. `verdict: null` if the verifier hasn't run for this symbol yet.
    """
    from ..db import get_latest_verdict

    row = await get_latest_verdict(symbol.upper())
    return {
        "symbol": symbol.upper(),
        "verdict": _parse_verdict(row) if row else None,
        "timestamp": _now_iso(),
    }
