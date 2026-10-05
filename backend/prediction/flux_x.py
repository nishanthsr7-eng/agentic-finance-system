"""
FLUX-X — The Advanced Algorithm (serving capstone)
==============================================================
Every phase (0–9) built one organ of FLUX-X; this module is the spinal cord that runs the
full per-rebalance loop end to end, in ten steps:

    1. FEATURES ─┐
    2. REGIME    │  per-symbol → predict.predict()  (Layers 1–2: features, regime, base
    3. BASE      │                                    learners, meta-label, stack, calibrate,
    4. META      │                                    conformal band, edge/act gate, Kelly size)
    5. STACK     │
    6. CALIBRATE │
    7. EDGE/SIZE ─┘
    8. CROSS-SECTION   ← **this module** (construct_book): rank the whole live universe, take the
                         long-only top decile, regime-gate + causal vol-target the book → weights
    9. VERIFY          → agent.verify_portfolio red-teams the book's top-k (veto/downgrade only)
   10. EXECUTE         → paper.submit_orders (Alpaca paper, dry-run) ; serve.resolve_due +
                         refresh_live_calibration close the flywheel ; drift.maybe_retrain self-heals

Steps 1–7 and 9–10's plumbing already exist as standalone organs. The MISSING capstone was step 8
*at serving time*: ``portfolio.py`` proved the cross-sectional construction in backtest (GATE-7) but
nothing turned **today's** leaderboard into **today's** target book. ``construct_book`` is that live
step; ``run_flux_x`` wires all ten steps into one auditable cycle.

Honesty contract (inherited, enforced in code):
  • The book ships the GATE-7 *champion*, not the spec's theoretical ideal. Phase-7 evidence
    (see ``portfolio.py`` / memory) locked: LONG-ONLY, top-decile, ranked by **edge**, EQUAL-weight,
    causal **vol-target** (30% annual) + **regime** gate as the drawdown levers. The neutralized
    long/short and the inverse-vol / edge×meta tilts FAILED their gate in this long-biased universe,
    so they stay off by default (still selectable as params for research, never the default).
  • Vol-target leverage is CAUSAL: it uses only the selected names' returns strictly before today.
  • The Layer-3 verifier is a one-way valve: it may shrink or drop a position, never enlarge one.
    Book de-risking from a veto/downgrade is NOT renormalized away — gross exposure honestly falls.

Public API:
    await construct_book(predictions, equity=..., ...)  -> dict  (live step 8)
    await run_flux_x(equity=None, verify_top_k=5, ...)  -> dict  (the full loop)
"""

from __future__ import annotations

import logging

import numpy as np

from .regime import REGIME_SCALE

log = logging.getLogger("flux.prediction.flux_x")

# GATE-7 champion defaults (the deployable book). See portfolio.py for the evidence trail.
FRAC = 0.10  # long-only top decile
VOL_TARGET = 0.30  # annualized vol target for the book (causal lever)
VOL_WINDOW = 20  # trailing trading days for the causal basket-vol estimate
MAX_LEVERAGE = 3.0  # cap on the vol-target leverage
TRADING_DAYS = 252
DEFAULT_EQUITY = 100_000.0


def _edge(p: dict) -> float:
    return float(p.get("prob_up", 0.5)) - 0.5


def _conviction(p: dict) -> float:
    """Signed edge × meta-prob — the cross-sectional ranking score (step 8)."""
    meta = p.get("meta_prob")
    return _edge(p) * float(meta if meta is not None else 0.5)


async def _basket_daily_vol(symbols: list[str], window: int = VOL_WINDOW) -> float | None:
    """
    CAUSAL trailing daily vol of an equal-weight basket of `symbols`, using only bars ≤ today.

    Equal-weight the names' daily log returns, then take the std of the last `window` basket
    returns. Returns None if there isn't enough shared history (caller falls back to 1× leverage).
    """
    import pandas as pd

    from ..db import get_history

    rets = {}
    for sym in symbols:
        rows = await get_history(sym)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["date"])
        px = (df["adj_close"] if "adj_close" in df else df["close"]).astype(float)
        rets[sym] = np.log(px / px.shift())
    if not rets:
        return None
    basket = pd.DataFrame(rets).sort_index().mean(axis=1, skipna=True).dropna()
    if len(basket) < max(5, window // 2):
        return None
    sd = float(basket.tail(window).std(ddof=1))
    return sd if np.isfinite(sd) and sd > 0 else None


def _vol_target_leverage(
    basket_vol: float | None, *, vol_target: float = VOL_TARGET, max_leverage: float = MAX_LEVERAGE
) -> float:
    """clip(target_daily / trailing_basket_vol, 0, max_leverage). 1× when no estimate (warm-up)."""
    if not vol_target or basket_vol is None or basket_vol <= 0:
        return 1.0
    target_daily = vol_target / np.sqrt(TRADING_DAYS)
    return float(min(max_leverage, max(0.0, target_daily / basket_vol)))


async def construct_book(
    predictions: list[dict],
    *,
    equity: float = DEFAULT_EQUITY,
    frac: float = FRAC,
    regime: str | None = None,
    rank: str = "edge",
    min_edge: float = 0.0,
    use_regime_gate: bool = True,
    vol_target: float | None = VOL_TARGET,
    max_leverage: float = MAX_LEVERAGE,
    vol_window: int = VOL_WINDOW,
    kelly: bool = False,
    inv_vol: bool = False,
) -> dict:
    """
    Step 8 (live): turn today's per-symbol leaderboard into today's target book.

    Reproduces the GATE-7 champion by default — long-only, top-`frac` by `edge`, equal-weight,
    regime-gated, causal vol-targeted. Pure ranking/sizing on the passed predictions; the only I/O
    is the causal basket-vol lookup for the vol-target leverage (skipped if ``vol_target`` is None).

        equity          : account equity the notional weights are applied to
        frac            : long-leg fraction of the universe (0.10 = top decile, the champion)
        regime          : market regime label for the size gate; defaults to predictions[0]['regime']
        rank            : 'edge' (champion) or 'score' (spec's edge×meta conviction)
        min_edge        : only long names with edge strictly above this (no longing a bearish name)
        use_regime_gate : scale gross exposure by REGIME_SCALE[regime] (stand down in risk_off)
        vol_target      : annualized book vol target (None = off) — the causal drawdown lever
        kelly / inv_vol : optional conviction / risk-parity leg tilts (OFF — failed GATE-7 at decile)

    Returns a dict: {regime, n_universe, k, gross_exposure, leverage, gross_after_levers, book[...]}
    where each book row carries {symbol, direction, edge, meta_prob, weight, target_frac, notional}.
    """
    valid = [p for p in predictions if p and p.get("prob_up") is not None]
    n = len(valid)
    if regime is None:
        regime = valid[0].get("regime", "trend") if valid else "trend"

    # Long-only: rank the universe, keep only genuinely-bullish names, take the top fraction.
    key = _conviction if rank == "score" else _edge
    bullish = [p for p in valid if _edge(p) > min_edge]
    ranked = sorted(bullish, key=key, reverse=True)
    k = max(1, int(round(frac * n))) if n else 0
    longs = ranked[:k]

    # ── Leg weights (equal-weight champion; optional conviction / risk-parity tilts) ──
    book: list[dict] = []
    if longs:
        if kelly or inv_vol:
            from .portfolio import leg_weights

            meta = np.array([float(p.get("meta_prob") or 0.5) for p in longs])
            vols = np.array([float(p.get("trail_vol") or np.nan) for p in longs])
            w = leg_weights(meta, vols, kelly=kelly, inv_vol=inv_vol)
        else:
            w = np.full(len(longs), 1.0 / len(longs))  # equal weight — the locked champion

        gross = REGIME_SCALE.get(regime, 1.0) if use_regime_gate else 1.0
        basket_vol = (
            await _basket_daily_vol([p["symbol"] for p in longs], vol_window)
            if vol_target
            else None
        )
        lev = _vol_target_leverage(basket_vol, vol_target=vol_target, max_leverage=max_leverage)

        for p, wi in zip(longs, w):
            tgt = gross * lev * float(wi)  # fraction of equity in this name
            book.append(
                {
                    "symbol": p["symbol"],
                    "direction": "UP",
                    "edge": round(_edge(p), 4),
                    "meta_prob": p.get("meta_prob"),
                    "act": bool(p.get("act")),
                    "weight": round(float(wi), 4),
                    "target_frac": round(tgt, 4),
                    "notional": round(equity * tgt, 2),
                }
            )
    else:
        gross, lev = (REGIME_SCALE.get(regime, 1.0) if use_regime_gate else 1.0), 1.0

    return {
        "regime": regime,
        "n_universe": n,
        "k": len(book),
        "gross_exposure": round(gross, 4),
        "leverage": round(lev, 4),
        "gross_after_levers": round(sum(b["target_frac"] for b in book), 4),
        "rank_by": rank,
        "vol_target": vol_target,
        "book": book,
    }


def apply_verdicts(book: dict, verdicts: list[dict]) -> dict:
    """
    Fold the Layer-3 verifier's one-way decisions into the book (step 9 → book).

    A veto DROPS the name (notional→0); a downgrade SHRINKS its notional by
    final_confidence/model_confidence. Neither is ever renormalized away — the book's gross
    exposure honestly falls when the verifier de-risks, which is the whole point of the safety
    valve. Names with no verdict (outside the reviewed top-k) are left untouched.
    """
    by_sym = {v.get("symbol"): v for v in verdicts}
    kept, n_veto, n_down = [], 0, 0
    for b in book["book"]:
        v = by_sym.get(b["symbol"])
        if v is None:
            kept.append(b)
            continue
        if v.get("veto"):
            n_veto += 1
            continue  # dropped from the book entirely
        mc = int(v.get("model_confidence", v.get("confidence", 0)) or 0)
        fc = int(v.get("final_confidence", mc) or 0)
        if mc > 0 and fc < mc:  # downgrade → shrink size (never grow)
            scale = fc / mc
            b = {
                **b,
                "target_frac": round(b["target_frac"] * scale, 4),
                "notional": round(b["notional"] * scale, 2),
                "downgraded_to": fc,
            }
            n_down += 1
        kept.append(b)
    out = {
        **book,
        "book": kept,
        "vetoed": n_veto,
        "downgraded": n_down,
        "gross_after_levers": round(sum(b["target_frac"] for b in kept), 4),
    }
    return out


async def run_flux_x(
    equity: float | None = None, verify_top_k: int = 5, live_submit: bool = False, **book_kw
) -> dict:
    """
    The full FLUX-X loop, in order — the production rebalance turn.

      regime refresh → resolve matured (flywheel) → predict+log the universe (steps 1–7) →
      construct the cross-sectional book (step 8) → red-team its top-k (step 9, veto/downgrade
      only) → paper dry-run the surviving orders (step 10).

    Every outward/heavy step degrades gracefully: a missing LLM, missing Alpaca keys, or a thin
    universe never crashes the cycle — they just shrink the loop. Returns a structured report.
    """
    from .regime import current_regime, invalidate_regime_cache
    from .serve import resolve_due, run_predictions

    invalidate_regime_cache()  # one HMM fit shared across the cycle
    n_resolved = await resolve_due()
    preds = await run_predictions()

    regime = "trend"
    try:
        regime = (await current_regime())["regime"]
    except Exception as exc:
        log.warning("regime unavailable, defaulting trend: %s", exc)

    if equity is None:
        try:
            from .paper import DEFAULT_EQUITY as PAPER_EQUITY
            from .paper import account

            acct = await account()
            equity = acct["equity"] if acct else PAPER_EQUITY
        except Exception:
            equity = DEFAULT_EQUITY

    book = await construct_book(preds, equity=equity, regime=regime, **book_kw)

    # ── Step 9: red-team the book's names (the top-k by conviction are exactly the funded ones) ──
    verdicts: list[dict] = []
    if verify_top_k:
        try:
            from .agent import verify_portfolio

            verdicts = await verify_portfolio(preds, top_k=verify_top_k, persist=True)
        except Exception as exc:  # the LLM layer must never break the loop
            log.warning("portfolio verify skipped: %s", exc)
    book = apply_verdicts(book, verdicts)

    # ── Step 10: paper execution (dry-run by default) of the surviving, sized book ──
    paper = {"status": "skipped"}
    try:
        from .paper import submit_orders

        orders_in = [
            {
                "symbol": b["symbol"],
                "act": True,
                "direction": b["direction"],
                "kelly_frac": b["target_frac"],
            }
            for b in book["book"]
            if b["target_frac"] > 0
        ]
        paper = await submit_orders(orders_in, equity=equity, live_submit=live_submit)
    except Exception as exc:
        log.warning("paper submit skipped: %s", exc)
        paper = {"status": "error", "reason": str(exc)[:120]}

    report = {
        "resolved": n_resolved,
        "logged": len(preds),
        "regime": regime,
        "equity": equity,
        "book_size": len(book["book"]),
        "gross_exposure": book.get("gross_after_levers"),
        "leverage": book.get("leverage"),
        "vetoed": book.get("vetoed", 0),
        "downgraded": book.get("downgraded", 0),
        "verified": sum(
            1 for v in verdicts if v.get("verifier") not in ("unavailable", "parse_error")
        ),
        "paper_status": paper.get("status"),
        "book": book["book"],
    }
    log.info(
        "FLUX-X cycle: logged %d | book %d names | gross %.2f | regime %s | paper %s",
        report["logged"],
        report["book_size"],
        report["gross_exposure"] or 0.0,
        regime,
        report["paper_status"],
    )
    return report


if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    async def _demo():
        from backend.db import init_db

        await init_db()
        rep = await run_flux_x(verify_top_k=0)  # skip the LLM for a fast offline demo
        print(
            f"FLUX-X: logged {rep['logged']} | regime {rep['regime']} | "
            f"book {rep['book_size']} names | gross {rep['gross_exposure']} | "
            f"lev {rep['leverage']} | paper {rep['paper_status']}"
        )
        for b in rep["book"]:
            print(
                f"  {b['symbol']:6} edge {b['edge']:+.3f} meta {b['meta_prob']} "
                f"w {b['weight']:.2f} → {b['target_frac']:.3f} (${b['notional']:,.0f})"
            )

    asyncio.run(_demo())
