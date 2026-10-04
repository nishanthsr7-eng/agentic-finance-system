"""
FLUX-X capstone tests — live cross-sectional book construction (Workflow §4, step 8).

Run directly:    python backend/prediction/test_flux_x.py
Or with pytest:  pytest backend/prediction/test_flux_x.py -q

These exercise the PURE selection / sizing / verifier-folding math of the live book (the only
I/O in construct_book — the causal basket-vol lookup — is disabled here by passing
vol_target=None, so the tests are DB-free and run in milliseconds). The decisive properties:
  • long-only top-decile selection ranks by edge and never longs a bearish name,
  • the regime gate scales gross exposure down in stress (and is the only size lever when
    vol-target is off),
  • the vol-target leverage is the causal clip(target/trailing_vol) formula,
  • the verifier is a ONE-WAY valve: a veto drops the name, a downgrade shrinks it, neither is
    renormalized away (gross exposure honestly falls).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.prediction.flux_x import (
    TRADING_DAYS,
    VOL_TARGET,
    _conviction,
    _edge,
    _vol_target_leverage,
    apply_verdicts,  # noqa: E402
    construct_book,
)
from backend.prediction.regime import REGIME_SCALE  # noqa: E402


def _pred(sym, prob_up, meta=0.6, regime="trend"):
    return {
        "symbol": sym,
        "prob_up": prob_up,
        "meta_prob": meta,
        "regime": regime,
        "act": meta >= 0.60,
        "direction": "UP" if prob_up >= 0.5 else "DOWN",
    }


def _run(coro):
    return asyncio.run(coro)  # fresh loop per call (no DB I/O here → cheap)


# ── 1. Long-only top-decile selection ranks by edge, skips bearish names ───────
def test_top_decile_long_only() -> None:
    # 10 names: 6 bullish (prob>0.5), 4 bearish. Top decile of 10 = 1 name → the most bullish.
    preds = [_pred(f"B{i}", 0.50 + 0.01 * i) for i in range(1, 7)] + [
        _pred(f"S{i}", 0.50 - 0.01 * i) for i in range(1, 5)
    ]
    book = _run(construct_book(preds, frac=0.10, vol_target=None))
    assert book["n_universe"] == 10, "universe count wrong"
    assert book["k"] == 1, f"top decile of 10 should be 1 name, got {book['k']}"
    assert book["book"][0]["symbol"] == "B6", "must pick the highest-edge bullish name"
    # No bearish name can ever enter a long-only book, even if the fraction is large.
    wide = _run(construct_book(preds, frac=1.0, vol_target=None))
    assert all(b["edge"] > 0 for b in wide["book"]), "long-only book longed a bearish name"
    assert len(wide["book"]) == 6, "only the 6 bullish names are eligible to be long"
    print(
        f"  [1] top-decile long-only   OK   (picked {book['book'][0]['symbol']}, "
        f"wide book = {len(wide['book'])} bullish)"
    )


# ── 2. Equal weight is a valid simplex (the GATE-7 champion weighting) ─────────
def test_equal_weight_simplex() -> None:
    preds = [_pred(f"B{i}", 0.50 + 0.02 * i) for i in range(1, 6)]  # 5 bullish names
    book = _run(construct_book(preds, frac=1.0, vol_target=None, use_regime_gate=False))
    w = np.array([b["weight"] for b in book["book"]])
    assert abs(w.sum() - 1.0) < 1e-9, "weights must sum to 1"
    assert np.allclose(w, 0.2), "champion weighting is equal weight"
    # With no regime gate and vol-target off, gross exposure is exactly 1× (fully invested).
    assert abs(book["gross_after_levers"] - 1.0) < 1e-9, "gross should be 1x with no levers"
    print("  [2] equal-weight simplex   OK   (sum=1, gross=1x with levers off)")


# ── 3. Regime gate scales gross exposure down in stress ───────────────────────
def test_regime_gate_scales_exposure() -> None:
    preds = [_pred(f"B{i}", 0.55, regime="risk_off") for i in range(1, 6)]
    book = _run(construct_book(preds, frac=1.0, vol_target=None, regime="risk_off"))
    # gross = REGIME_SCALE['risk_off'] = 0.25 → the whole book is delevered to a quarter.
    assert abs(book["gross_exposure"] - REGIME_SCALE["risk_off"]) < 1e-9, "regime gate not applied"
    assert abs(book["gross_after_levers"] - 0.25) < 1e-9, "risk_off should cut gross to 0.25x"
    trend = _run(
        construct_book([_pred(f"B{i}", 0.55) for i in range(1, 6)], frac=1.0, vol_target=None)
    )
    assert trend["gross_after_levers"] > book["gross_after_levers"], (
        "trend must risk more than risk_off"
    )
    print(
        f"  [3] regime gate            OK   (risk_off gross {book['gross_after_levers']} "
        f"< trend {trend['gross_after_levers']})"
    )


# ── 4. Vol-target leverage is the causal clip(target/trailing_vol) formula ─────
def test_vol_target_leverage_formula() -> None:
    target_daily = VOL_TARGET / np.sqrt(TRADING_DAYS)
    # Calm tape (vol below target) → lever UP; stormy tape (vol above target) → lever DOWN.
    calm = _vol_target_leverage(target_daily / 2.0)
    storm = _vol_target_leverage(target_daily * 3.0)
    assert abs(calm - 2.0) < 1e-9, "leverage should be target/vol = 2x in calm tape"
    assert abs(storm - (1 / 3.0)) < 1e-6, "leverage should delever to 1/3 in a storm"
    # Cap and warm-up guards.
    assert _vol_target_leverage(1e-9) == 3.0, "leverage must be capped at MAX_LEVERAGE"
    assert _vol_target_leverage(None) == 1.0, "no estimate → 1x (warm-up)"
    assert _vol_target_leverage(0.02, vol_target=None) == 1.0, "vol-target off → 1x"
    print("  [4] vol-target leverage    OK   (causal clip(target/vol), capped, warm-up=1x)")


# ── 5. Verifier veto DROPS a name; gross exposure honestly falls ──────────────
def test_verdict_veto_drops_name() -> None:
    preds = [_pred(f"B{i}", 0.50 + 0.02 * i) for i in range(1, 6)]
    book = _run(construct_book(preds, frac=1.0, vol_target=None, use_regime_gate=False))
    gross0 = book["gross_after_levers"]
    veto_sym = book["book"][0]["symbol"]
    out = apply_verdicts(
        book, [{"symbol": veto_sym, "veto": True, "model_confidence": 70, "final_confidence": 35}]
    )
    assert all(b["symbol"] != veto_sym for b in out["book"]), "vetoed name still in book"
    assert out["vetoed"] == 1 and len(out["book"]) == len(book["book"]) - 1, "veto count wrong"
    # The freed weight is NOT redistributed — gross drops by the vetoed name's slice.
    assert out["gross_after_levers"] < gross0 - 1e-9, "veto must lower gross (no renormalization)"
    print(
        f"  [5] verifier veto          OK   (dropped {veto_sym}, gross {gross0:.2f}->"
        f"{out['gross_after_levers']:.2f}, not renormalized)"
    )


# ── 6. Verifier downgrade SHRINKS a position proportionally (never grows) ──────
def test_verdict_downgrade_shrinks() -> None:
    preds = [_pred(f"B{i}", 0.50 + 0.02 * i) for i in range(1, 6)]
    book = _run(construct_book(preds, frac=1.0, vol_target=None, use_regime_gate=False))
    sym = book["book"][0]["symbol"]
    frac0 = book["book"][0]["target_frac"]
    out = apply_verdicts(
        book, [{"symbol": sym, "veto": False, "model_confidence": 80, "final_confidence": 40}]
    )  # halved
    row = next(b for b in out["book"] if b["symbol"] == sym)
    assert abs(row["target_frac"] - frac0 * 0.5) < 1e-9, "downgrade should halve the size"
    assert row["downgraded_to"] == 40 and out["downgraded"] == 1, "downgrade not recorded"
    # A verdict that tries to RAISE confidence can never enlarge the position (one-way valve).
    up = apply_verdicts(
        book, [{"symbol": sym, "veto": False, "model_confidence": 80, "final_confidence": 99}]
    )
    assert abs(next(b for b in up["book"] if b["symbol"] == sym)["target_frac"] - frac0) < 1e-9, (
        "verifier raised a position — one-way valve violated"
    )
    print("  [6] verifier downgrade     OK   (shrinks proportionally, never enlarges)")


# ── 7. Helper invariants: edge, conviction, score-ranking ─────────────────────
def test_helpers_and_score_rank() -> None:
    assert abs(_edge({"prob_up": 0.62}) - 0.12) < 1e-12, "edge = prob_up - 0.5"
    # conviction = edge × meta → a high-edge low-meta name can rank below a lower-edge high-meta one.
    a = {"prob_up": 0.60, "meta_prob": 0.55}  # edge .10 × .55 = .055
    b = {"prob_up": 0.56, "meta_prob": 0.90}  # edge .06 × .90 = .054
    assert _conviction(a) > _conviction(b), "conviction = edge×meta"
    # rank='score' reorders vs rank='edge': here edge picks A, but they're close on conviction.
    preds = [_pred("A", 0.60, meta=0.55), _pred("B", 0.59, meta=0.95)]
    by_edge = _run(construct_book(preds, frac=0.5, vol_target=None))["book"][0]["symbol"]
    by_score = _run(construct_book(preds, frac=0.5, rank="score", vol_target=None))["book"][0][
        "symbol"
    ]
    assert by_edge == "A", "edge ranking should pick the higher-edge name"
    assert by_score == "B", "score ranking should pick the higher edge×meta name"
    print(f"  [7] helpers + score-rank   OK   (edge->{by_edge}, score->{by_score})")


def _run_all() -> None:
    print("Running FLUX-X capstone tests (live cross-sectional book, step 8):")
    test_top_decile_long_only()
    test_equal_weight_simplex()
    test_regime_gate_scales_exposure()
    test_vol_target_leverage_formula()
    test_verdict_veto_drops_name()
    test_verdict_downgrade_shrinks()
    test_helpers_and_score_rank()
    print("All tests passed.")


if __name__ == "__main__":
    _run_all()
