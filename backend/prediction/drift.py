"""
FLUX Prediction — Drift Detection & Triggered Retrain (Phase 9)
==============================================================
A calibrated model silently decays: the market regime shifts, the calibration map goes stale,
and "62%" quietly stops meaning 62%. This module watches the LIVE resolved outcomes and triggers
a retrain only when the model has measurably drifted — so we retrain on evidence, not on a blind
fixed schedule (which either wastes compute or reacts too late).

Drift signals (computed from prediction_outcomes ⨝ predictions):
  • live directional accuracy on resolved trades drops below `ACC_FLOOR`, or
  • live calibration error (ECE between stated confidence and realized hit-rate) exceeds `ECE_CEIL`.
Both require at least `MIN_N` resolved outcomes, so we never retrain on noise.

A retrain is also rate-limited (`MIN_RETRAIN_AGE_DAYS`) so a bad streak can't thrash the trainer.

Public API:
    await compute_live_metrics(min_n)  -> {n, accuracy, ece, status}
    await check_drift()                -> {drift, reasons, metrics, last_trained_days}
    await maybe_retrain(force=False)   -> {retrained, reason, ...}   # scheduler entrypoint
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("flux.prediction.drift")

MODELS_DIR = Path(__file__).parent / "models"
MIN_N = 200  # need this many resolved trades before trusting live metrics
ACC_FLOOR = 0.48  # below ~coin-flip-minus → something broke
ECE_CEIL = 0.10  # calibration error ceiling before we consider it drifted
MIN_RETRAIN_AGE_DAYS = 3  # rate-limit: don't retrain more often than this


def _ece(conf01: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf01 >= lo) & (conf01 < hi)
        if m.sum():
            e += abs(conf01[m].mean() - correct[m].mean()) * m.sum() / len(conf01)
    return float(e)


async def compute_live_metrics(min_n: int = MIN_N) -> dict:
    """Live accuracy + ECE from resolved outcomes. status='insufficient' until min_n resolved."""
    from ..db import get_resolved_outcomes

    rows = await get_resolved_outcomes()
    n = len(rows)
    if n < min_n:
        return {"n": n, "accuracy": None, "ece": None, "status": "insufficient", "needed": min_n}
    conf = np.array([(r.get("confidence") or 0) / 100.0 for r in rows], dtype=float)
    correct = np.array([int(r.get("correct") or 0) for r in rows], dtype=float)
    return {"n": n, "accuracy": float(correct.mean()), "ece": _ece(conf, correct), "status": "ok"}


def _last_trained_days() -> float | None:
    meta = MODELS_DIR / "model_meta.json"
    if not meta.exists():
        return None
    try:
        ts = json.loads(meta.read_text()).get("trained_at")
        return (time.time() * 1000 - ts) / 86_400_000 if ts else None
    except Exception:
        return None


async def check_drift() -> dict:
    """Evaluate drift signals. Returns {drift, reasons, metrics, last_trained_days}."""
    m = await compute_live_metrics()
    reasons = []
    if m["status"] == "ok":
        if m["accuracy"] < ACC_FLOOR:
            reasons.append(f"accuracy {m['accuracy']:.3f} < floor {ACC_FLOOR}")
        if m["ece"] > ECE_CEIL:
            reasons.append(f"ECE {m['ece']:.3f} > ceil {ECE_CEIL}")
    return {
        "drift": bool(reasons),
        "reasons": reasons,
        "metrics": m,
        "last_trained_days": _last_trained_days(),
    }


async def maybe_retrain(force: bool = False) -> dict:
    """
    Retrain iff drift is detected (or force=True) AND the model isn't too fresh. Returns a
    summary; the actual retrain reuses train.main() so all artifacts are regenerated consistently.
    """
    d = await check_drift()
    age = d["last_trained_days"]
    if not force and not d["drift"]:
        return {"retrained": False, "reason": "no drift", **d}
    if not force and age is not None and age < MIN_RETRAIN_AGE_DAYS:
        return {
            "retrained": False,
            "reason": f"rate-limited (model {age:.1f}d < {MIN_RETRAIN_AGE_DAYS}d)",
            **d,
        }

    log.warning("Drift retrain triggered: reasons=%s force=%s", d["reasons"], force)
    try:
        from .train import main as train_main

        await train_main()  # regenerates all artifacts
        return {"retrained": True, "reason": "forced" if force else "; ".join(d["reasons"]), **d}
    except Exception as exc:
        log.error("Drift retrain failed: %s", exc)
        return {"retrained": False, "reason": f"retrain error: {exc}", **d}


if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path as _P

    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))

    async def _demo():
        from backend.db import init_db

        await init_db()
        print("live metrics :", await compute_live_metrics())
        print("drift check  :", await check_drift())
        print("(retrain is gated; run maybe_retrain(force=True) to force.)")

    asyncio.run(_demo())
