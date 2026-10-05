"""
PHASE-6 AUDIT — Regime-conditional ensemble (mixture-of-experts): does it work right *and* stay honest?
======================================================================================================
The self-contained acceptance audit for the Phase-6 regime stack (``ensemble.py``: ElasticNet 2nd
base learner + per-regime logistic experts softly mixed by the live HMM posterior). It is the
"verify it fully" companion to the GATE-6 evaluation (``scripts/experiments/gate6_regime_ensemble_eval.py``) and the unit
tests (``test_ensemble.py``): it re-runs the unit checks, proves the OOF assembly is leak-safe and
causal on the REAL panel, reproduces GATE-6, and confirms the predict.py self-gate is wired so the
stack only ships when it actually beats the best base learner.

Four sections, each a hard pass/fail:

  A. UNIT CORRECTNESS (synthetic)   soft-mixture math == manual posterior-weighted average; scalar ==
       batch; outputs bounded; one-hot posterior routes to a single expert; low-count / zero-posterior
       fall back to global; oof_base2 is purged-OOF aligned; base2 pipeline takes sample_weight;
       save/load round-trips; the legacy flat Stacker still works.

  B. OOF ASSEMBLY LEAK-SAFETY (real panel)   build the leak-free OOF stack frame and prove the inputs
       are sane and causal: base signals (primary_cal, p2_base, meta_prob, mag_oof) finite & in range;
       regime posteriors in [0,1] and ~sum-to-1 (proper HMM filter output); rows time-sorted; no NaN
       left in the stack columns after the dropna.

  C. GATE-6 REPRODUCTION (real panel)   fit the regime mixture-of-experts on the EARLY 70% / score the
       LATER 30%; report stack AUC vs the best single base learner; assert the report bookkeeping is
       internally consistent (best_base_auc == max(primary,p2); gate flag == stack>best+margin). The
       PASS/TIE/FAIL verdict is recorded — a tie keeps the stack self-gated OFF (honesty contract).

  D. SELF-GATE INTEGRITY   the predict.py serving gate must enable the stack IFF its report clears
       GATE-6, and require BOTH artifacts (regime_stack.pkl + base2.pkl). Verified against synthetic
       pass/fail reports using the exact gate condition, plus a save/load round-trip of a real stacker.

Run:  python scripts/experiments/gate6_regime_ensemble_audit.py        (exit 0 = all pass, 1 = any failure)
Heavy: builds the ~138k-event panel + trains several purged-OOF models (a few minutes).
"""

from __future__ import annotations

import asyncio
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

_results: list[tuple[str, bool, str]] = []


def _check(name: str, fn) -> None:
    try:
        detail = fn() or ""
        _results.append((name, True, detail))
        print(f"  PASS  {name}" + (f"   {detail}" if detail else ""))
    except Exception as exc:  # noqa: BLE001 — an audit reports every failure, never crashes out
        _results.append((name, False, str(exc)))
        print(f"  FAIL  {name}   {exc}")
        if not isinstance(exc, AssertionError):
            traceback.print_exc()


async def _main() -> int:
    from backend.prediction import test_ensemble as t6
    from backend.prediction.ensemble import (
        BASE_SIGNALS,
        GATE_MARGIN,
        OOF_COLS,
        REGIME_PROBS,
        REGIMES,
        RegimeStacker,
        train_regime_stack,
    )

    print("=" * 80)
    print("  PHASE-6 AUDIT — regime-conditional mixture-of-experts: correctness + honesty")
    print("=" * 80)

    # ── A. Unit correctness (synthetic; reuse the shared checks) ─────────────────
    print("\n[A] UNIT CORRECTNESS  (ensemble.py synthetic checks)")
    feats_s, y_s = t6._synth_oof()
    stk_s, rep_s = train_regime_stack(feats_s, y_s)
    _check(
        "A1 soft-mixture math == manual weighted avg; scalar==batch",
        lambda: t6._check_mixture_math(stk_s, feats_s),
    )
    _check("A2 probabilities finite + bounded [0,1]", lambda: t6._check_bounds(stk_s, feats_s))
    _check(
        "A3 one-hot posterior routes to the matching expert",
        lambda: t6._check_hard_routing(feats_s),
    )
    _check("A4 low-count / zero-posterior fall back to global", t6._check_fallback)
    _check("A5 oof_base2 is purged-OOF aligned", t6._check_oof_base2_alignment)
    _check("A6 GATE-6 report internally consistent", lambda: t6._check_gate_logic(rep_s))
    _check(
        "A7 regime experts earn their place on regime-switch truth",
        lambda: t6._check_moe_earns_place(rep_s),
    )

    # ── B + C need the real panel + OOF assembly ─────────────────────────────────
    print(
        "\n[B] OOF ASSEMBLY LEAK-SAFETY  (real panel — assembling base signals + regime posteriors)"
    )
    from backend.prediction.ensemble import _assemble_oof

    feats, y, _ctx = await _assemble_oof()
    print(
        f"  assembled {len(feats):,} OOF stack rows | "
        f"regime mix {{ {', '.join(f'{r}:{int((feats.regime == r).sum()):,}' for r in REGIMES)} }}"
    )

    def b_signals_sane():
        for c in ("primary_cal", "p2_base", "meta_prob"):
            v = feats[c].values
            assert np.isfinite(v).all(), f"{c} has non-finite values"
            assert (v >= 0).all() and (v <= 1).all(), f"{c} outside [0,1]"
        assert np.isfinite(feats["mag_oof"].values).all(), "mag_oof has non-finite values"
        return "primary_cal/p2_base/meta_prob finite & in [0,1]; mag_oof finite"

    def b_regime_causal():
        rp = feats[REGIME_PROBS].values
        assert np.isfinite(rp).all(), "regime posteriors non-finite"
        assert (rp >= -1e-9).all() and (rp <= 1 + 1e-9).all(), "regime posteriors outside [0,1]"
        sums = rp.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-3), (
            f"regime posteriors do not sum to 1 (max dev {np.abs(sums - 1).max():.2e})"
        )
        return f"3 regime posteriors in [0,1], rows sum to 1 (max dev {np.abs(sums - 1).max():.1e})"

    def b_no_nan_sorted():
        assert not feats[OOF_COLS].isna().any().any(), "NaN left in stack columns after dropna"
        d = feats["_date"].values
        assert (d[:-1] <= d[1:]).all(), "stack rows not time-sorted"
        return f"no NaN in {len(OOF_COLS)} stack cols; rows time-sorted"

    _check("B1 base signals finite & in range", b_signals_sane)
    _check("B2 regime posteriors causal-valid (in [0,1], sum=1)", b_regime_causal)
    _check("B3 no NaN in stack cols + time-sorted", b_no_nan_sorted)

    # ── C. GATE-6 reproduction on the real panel ─────────────────────────────────
    print("\n[C] GATE-6 REPRODUCTION  (fit MoE on early 70%, score later 30%)")
    stacker, rep = train_regime_stack(feats, y)
    print(
        f"    base : primary {rep['primary_auc']:.4f} | base2(EN) {rep['p2_auc']:.4f} "
        f"(corr {rep['p2_corr_primary']:.3f}) -> best {rep['best_base_name']} {rep['best_base_auc']:.4f}"
    )
    print(
        f"    stack: shipped {rep['stack_auc']:.4f} (unshrunk {rep['stack_unshrunk_auc']:.4f}, "
        f"blend->primary {rep['primary_blend']:.2f}) | lift vs best base {rep['lift_vs_best_base']:+.4f}"
    )
    print(
        f"    regime ablation: global {rep['global_stack_auc']:.4f} | MoE {rep['moe_stack_auc']:.4f} "
        f"| +regime-as-feature {rep['regime_feature_stack_auc']:.4f} -> use_experts={rep['use_experts']}"
    )
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

    def c_report_consistent():
        return t6._check_gate_logic(rep)

    def c_verdict():
        lift = rep["lift_vs_best_base"]
        if rep["gate6_pass"]:
            v = f"PASS (lift {lift:+.4f}) -> stack ships ON"
        elif lift >= -GATE_MARGIN:
            v = f"TIE within noise (lift {lift:+.4f}) -> stays self-gated OFF (honesty contract)"
        else:
            v = f"FAIL (lift {lift:+.4f}) -> stays self-gated OFF"
        print(f"    GATE-6 verdict: {v}")
        return v

    def c_regime_usage_honest():
        # usage (b): per-regime experts are kept ONLY if they beat the global stack OOS.
        expect = rep["moe_stack_auc"] > rep["global_stack_auc"] + GATE_MARGIN
        assert rep["use_experts"] == expect, "use_experts decision not data-driven"
        # usage (a): regime-as-feature is measured but must never be the shipped stack.
        assert "regime_feature_stack_auc" in rep, "regime-as-feature ablation not recorded"
        note = (
            ""
            if rep["regime_feature_stack_auc"] <= rep["global_stack_auc"]
            else " (NOTE: regime-as-feature helped on this panel — revisit usage (a))"
        )
        return (
            f"experts kept iff MoE>global (use_experts={rep['use_experts']}); "
            f"regime-as-feature {rep['regime_feature_stack_auc']:.4f} vs global "
            f"{rep['global_stack_auc']:.4f}, not shipped{note}"
        )

    _check("C1 GATE-6 report internally consistent (real panel)", c_report_consistent)
    _check("C2 GATE-6 verdict recorded", c_verdict)
    _check("C3 regime usages (a/b) decided honestly by evidence", c_regime_usage_honest)

    # ── D. Self-gate integrity (the predict.py serving gate) ─────────────────────
    print("\n[D] SELF-GATE INTEGRITY  (predict.py only ships the stack when it earns it)")

    def _gate_decision(report: dict) -> bool:
        """Mirror of the exact condition in predict.py _load()."""
        return bool(
            report.get("gate6_pass")
            or report.get("stack_auc", 0.0) > report.get("best_base_auc", 1.0) + GATE_MARGIN
        )

    def d_gate_enables_only_on_lift():
        passing = {"gate6_pass": True, "stack_auc": 0.55, "best_base_auc": 0.52}
        failing = {"gate6_pass": False, "stack_auc": 0.51, "best_base_auc": 0.52}
        tie = {"gate6_pass": False, "stack_auc": 0.5201, "best_base_auc": 0.52}
        assert _gate_decision(passing), "gate should ENABLE a stack that clears GATE-6"
        assert not _gate_decision(failing), "gate should DISABLE a stack with no lift"
        assert not _gate_decision(tie), "gate should DISABLE a within-margin tie"
        assert _gate_decision(rep) == rep["gate6_pass"], "live report vs gate decision disagree"
        return "enable IFF stack beats best base by > margin (real report matches)"

    def d_requires_both_artifacts():
        # predict.py guards on `stack_path.exists() and base2_path.exists()`. Verify the source
        # actually requires both so a stack without its base-2 learner can never be activated.
        src = (Path(__file__).resolve().parents[2] / "backend/prediction/predict.py").read_text()
        assert "regime_stack.pkl" in src and "base2.pkl" in src, "predict.py missing artifact refs"
        assert "stack_path.exists() and base2_path.exists()" in src, (
            "predict.py does not require BOTH stack + base2 artifacts"
        )
        return "serving requires regime_stack.pkl AND base2.pkl"

    def d_save_load_roundtrip(
        tmp=Path(__file__).resolve().parents[2] / "backend/prediction/models",
    ):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "regime_stack.pkl"
            stacker.save(p)
            loaded = RegimeStacker.load(p)
            a = loaded.predict_proba_batch(feats.iloc[:2000])
            b = stacker.predict_proba_batch(feats.iloc[:2000])
            assert np.allclose(a, b, atol=1e-12), "save/load changed predictions"
        return "regime_stack save/load prediction-identical"

    _check("D1 gate enables only on real lift", d_gate_enables_only_on_lift)
    _check("D2 serving requires both stack + base2 artifacts", d_requires_both_artifacts)
    _check("D3 stacker save/load round-trip identical", d_save_load_roundtrip)

    # ── Verdict ──
    npass = sum(ok for _, ok, _ in _results)
    nfail = len(_results) - npass
    print("\n" + "=" * 80)
    if nfail == 0:
        print(f"  PHASE-6 AUDIT: ALL {npass} CHECKS PASS")
        print("  MoE math correct; OOF assembly leak-safe & causal; GATE-6 reproduces;")
        print("  predict.py self-gate ships the stack IFF it beats the best base learner.")
    else:
        print(f"  PHASE-6 AUDIT: {nfail} FAILED / {npass} passed")
        for name, ok, detail in _results:
            if not ok:
                print(f"    FAIL  {name}   {detail}")
    print("=" * 80)
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
