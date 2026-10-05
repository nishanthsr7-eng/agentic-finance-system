"""
GATE-6 — Regime-conditional ensemble (mixture-of-experts): does the Phase-6 stack finally EARN its
place by beating the best single base learner on the leak-free OOF?

This is the gate the stacking ensemble has failed until now. Phase 6 gives it two things it lacked:
  1. a DECORRELATED 2nd base learner (ElasticNet logistic, `p2_base`) — a linear view that makes
     different errors from the XGBoost primary, so the blend has something to gain from;
  2. the HMM REGIME used three ways — as stack FEATURES (the posteriors), as PER-REGIME EXPERT
     WEIGHTS (a logistic per trend/chop/risk_off, softly mixed by the live posterior), and as the
     downstream size gate (unchanged).

Method (honest, time-ordered, leak-free):
  • Assemble the OOF base signals with the SAME purged walk-forward splitter used in training, so
    primary_cal / p2_base / meta_prob / mag_oof line up row-for-row with no peek, and attach the
    causal HMM regime posteriors per event date.
  • Fit every expert on the EARLY 70% of rows, score on the LATER 30% only.

GATE-6: held-out stack AUC > the BEST single base learner's held-out AUC (max of primary_cal,
p2_base). PASS -> the stack ships ON (self-gate in predict.py clears against the persisted report);
a tie/decline -> it stays self-gated OFF (honesty contract), exactly as the flat stack did before.

    python scripts/experiments/gate6_regime_ensemble_eval.py
Heavy: builds the ~138k-event panel and trains several purged-OOF models (a few minutes).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


async def _main() -> int:
    from backend.prediction.ensemble import (
        BASE_SIGNALS,
        GATE_MARGIN,
        REGIME_PROBS,
        REGIMES,
        _assemble_oof,
        train_regime_stack,
    )

    print("=" * 78)
    print("  GATE-6 — regime-conditional mixture-of-experts stack")
    print("=" * 78)
    print(
        "\nAssembling leak-free OOF base signals (primary + base2 + meta + magnitude + regime)..."
    )
    feats, y, _ctx = await _assemble_oof()
    mix = {r: int((feats["regime"] == r).sum()) for r in REGIMES}
    print(
        f"  rows: {len(feats):,}  | base signals {BASE_SIGNALS} (regime {REGIME_PROBS} = gate only)"
    )
    print(f"  regime mix: {mix}")

    stacker, rep = train_regime_stack(feats, y)

    print("\n" + "-" * 78)
    print(f"  held-out eval rows: {rep['n_eval']:,}  (LATER 30%, time-ordered)")
    print(
        f"  base learners : primary {rep['primary_auc']:.4f}  |  base2(ElasticNet) {rep['p2_auc']:.4f}"
        f"  (corr {rep['p2_corr_primary']:.3f})"
    )
    print(f"  best single base learner: {rep['best_base_name']} = {rep['best_base_auc']:.4f}")
    print(
        f"  shipped stack AUC       : {rep['stack_auc']:.4f}   "
        f"(unshrunk {rep['stack_unshrunk_auc']:.4f}, blend->primary {rep['primary_blend']:.2f}; "
        f"lift vs best base {rep['lift_vs_best_base']:+.4f})"
    )
    print("  regime ablation (held-out AUC):")
    print(f"      global stack           : {rep['global_stack_auc']:.4f}")
    print(
        f"      + per-regime experts   : {rep['moe_stack_auc']:.4f}   -> use_experts={rep['use_experts']}"
    )
    print(
        f"      + regime-as-feature    : {rep['regime_feature_stack_auc']:.4f}   (usage (a), not shipped)"
    )
    print(f"  ACC : stack {rep['stack_acc']:.4f}  vs primary {rep['primary_acc']:.4f}")
    print(
        f"  ECE : stack {rep['stack_ece']:.4f}  vs primary {rep['primary_ece']:.4f}  (lower better)"
    )

    print("\n  per-regime held-out AUC (stack vs primary):")
    print(f"    {'regime':10}{'n':>8}{'stack':>10}{'primary':>10}{'delta':>10}")
    for r in REGIMES:
        pr = rep["per_regime"][r]
        if pr["stack_auc"] is not None:
            print(
                f"    {r:10}{pr['n']:>8,}{pr['stack_auc']:>10.4f}{pr['primary_auc']:>10.4f}"
                f"{pr['stack_auc'] - pr['primary_auc']:>+10.4f}"
            )
        else:
            print(f"    {r:10}{pr['n']:>8,}{'(too few)':>10}")

    print("\n" + "=" * 78)
    if rep["gate6_pass"]:
        verdict = (
            "PASS — the regime stack beats the best single base learner. It ships ON; "
            "predict.py's self-gate clears against the persisted report."
        )
    elif rep["lift_vs_best_base"] >= -GATE_MARGIN:
        verdict = (
            "TIE within noise — no lift over the best base learner; the stack stays "
            "self-gated OFF (honesty contract). The base signals are still too correlated."
        )
    else:
        verdict = "FAIL — stack underperforms the best base learner; stays self-gated OFF."
    print(f"  GATE-6 (stack AUC > best base learner, margin {GATE_MARGIN:g}): {verdict}")
    print("=" * 78)
    return 0 if rep["gate6_pass"] else 0  # gate result is reported, not a hard process failure


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
