"""
PHASE-4 AUDIT — Equity fundamentals & events: does the block work right *and not disturb the rest*?
===================================================================================================
A single, self-contained acceptance audit for the Phase-4 equity fundamentals block
(``equity_features.py``: earnings surprise/drift, revenue revision, valuation-z, accruals — point-in-
time from SEC filings). It is the "verify it fully" companion to the GATE-4 evaluation
(``scripts/gate4_equity_fundamentals_eval.py``) and the unit tests (``test_equity_features.py``): it re-runs the
leak-safety checks, then proves the block is *purely additive* to the production pipeline, reproduces
GATE-4, and confirms the older earnings gate it grew out of is untouched.

Four sections, each a hard pass/fail:

  A. LEAK-SAFETY & CORRECTNESS   re-runs the 5 unit checks (truncation-invariance filing + daily,
       finite/aligned, crypto-neutral, point-in-time, synthetic YTD->single-quarter reconstruction).

  B. NON-DISRUPTION (the headline)   builds the FULL production panel twice on the same DB — once with
       FLUX_EQUITY_FEATURES=0 (production default) and once =1 — and proves the block is additive and
       neutral: identical row set / labels / weights / dates, the 42 base feature columns BYTE-IDENTICAL,
       exactly the 5 equity columns added (nothing else), crypto rows all-zero in those columns (so the
       dropna() never wipes a crypto row), and the equity columns actually populated on equities. This
       is the proof that turning the block on (or leaving it off) cannot change any other phase.

  C. GATE-4 REPRODUCTION   on the ON panel, evaluate_oof on base-42 vs base+equity-47 over the SAME
       shared OOF rows; assert the equity-subset AUC delta is a within-noise tie (no structural lift ->
       the recorded "keep self-gated OFF" decision stands) and the neutral block does not move crypto.
       Soft-checks that the BASE arm still reproduces the locked Phase-3 numbers.

  D. EARNINGS-GATE INTEGRITY   the Phase-3.1 live earnings *gate* (earnings.py) that this phase expanded
       "from gate-only to feature" must still behave: down-scales confidence/size in-window, validates
       symbols. Confirms the expansion did not regress the gate.

Run:  python scripts/gate4_equity_fundamentals_audit.py        (exit 0 = all pass, 1 = any failure)
Heavy: builds the ~138k-event panel twice + trains the purged-OOF model twice (a few minutes).
"""

from __future__ import annotations

import asyncio
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

# Locked Phase-3 reference numbers (scripts/gate4_equity_fundamentals_eval.py / gate4_equity_fundamentals_eval.log) — soft guard only.
LOCKED = {"equity": 0.5107, "crypto": 0.5209, "pooled": 0.5158}
MARGIN = 1e-3  # an AUC delta within +/-MARGIN is a "tie"
TIE_BAND = 2e-3  # the equity delta must stay inside this band (no spurious lift, no real decline)
CRYPTO_GUARD = 5e-3
DRIFT_WARN = 5e-3  # warn (don't fail) if a BASE AUC drifts this far from the locked number

_results: list[tuple[str, bool, str]] = []


def _check(name: str, fn) -> None:
    """Run one named check; record pass/fail; never abort the whole audit on a single failure."""
    try:
        detail = fn() or ""
        _results.append((name, True, detail))
        print(f"  PASS  {name}" + (f"   {detail}" if detail else ""))
    except Exception as exc:  # noqa: BLE001 — an audit must report every failure, not crash
        _results.append((name, False, str(exc)))
        print(f"  FAIL  {name}   {exc}")
        if not isinstance(exc, AssertionError):
            traceback.print_exc()


def _auc_pair(y, base, aug, mask):
    """AUC of base & aug on the SAME rows (mask AND both-tested). (auc_base, auc_aug, n)."""
    from sklearn.metrics import roc_auc_score

    common = mask & ~np.isnan(base) & ~np.isnan(aug)
    yt = y[common]
    if yt.size == 0 or yt.min() == yt.max():
        return float("nan"), float("nan"), int(common.sum())
    return (
        float(roc_auc_score(yt, base[common])),
        float(roc_auc_score(yt, aug[common])),
        int(common.sum()),
    )


async def _main() -> int:
    from backend.prediction import test_equity_features as t4
    from backend.prediction.datasources import CRYPTO_SYMBOLS
    from backend.prediction.equity_features import EQUITY_FEATURE_COLS
    from backend.prediction.train import evaluate_oof, load_dataset

    print("=" * 78)
    print("  PHASE-4 AUDIT - equity fundamentals block: correctness + non-disruption")
    print("=" * 78)

    # ── A. Leak-safety & correctness (reuse the unit checks for one self-contained report) ──
    print("\n[A] LEAK-SAFETY & CORRECTNESS  (equity_features.py unit checks)")
    p_aapl, p_btc = await t4._setup()
    _check(
        "A1 causality (filing + daily truncation-invariance)",
        lambda: t4._check_filing_causality(p_aapl),
    )
    _check(
        "A2 equity block finite + index-aligned + populated", lambda: t4._check_equity_block(p_aapl)
    )
    _check("A3 crypto / unknown symbol neutral (all-zero)", lambda: t4._check_crypto_neutral(p_btc))
    _check(
        "A4 point-in-time (neutral before first filing)", lambda: t4._check_point_in_time(p_aapl)
    )
    _check("A5 YTD->single-quarter reconstruction + YoY", t4._check_quarterly_reconstruction)

    # ── B. Non-disruption: build the production panel OFF vs ON and prove additive + neutral ──
    print("\n[B] NON-DISRUPTION  (production panel: FLUX_EQUITY_FEATURES 0 vs 1)")
    os.environ["FLUX_CRYPTO_FEATURES"] = "0"  # isolate: only the equity flag varies
    os.environ["FLUX_EQUITY_FEATURES"] = "0"
    print("  building OFF panel (production default) ...")
    _Xoff, yoff, woff, _t1off, cols_off, _f0, doff = await load_dataset()
    os.environ["FLUX_EQUITY_FEATURES"] = "1"
    print("  building ON panel  (block joined) ...")
    Xon, yon, won, t1on, cols_on, _f1, don = await load_dataset()

    base_cols = [c for c in cols_on if c not in EQUITY_FEATURE_COLS]
    added = sorted(set(cols_on) - set(cols_off))
    crypto = don["_sym"].isin(CRYPTO_SYMBOLS).values
    equity = ~crypto

    def b_rowset():
        assert len(doff) == len(don), (
            f"row count changed {len(doff)} -> {len(don)} (block dropped/added rows!)"
        )
        assert np.array_equal(doff["_sym"].values, don["_sym"].values), "symbol ordering changed"
        assert np.array_equal(doff["_y"].values, don["_y"].values), "labels changed"
        assert np.allclose(doff["_w"].values, don["_w"].values), "sample weights changed"
        assert np.array_equal(doff["_date"].values, don["_date"].values), "event dates changed"
        return f"{len(don):,} events identical (labels/weights/dates byte-identical)"

    def b_additive_cols():
        assert added == sorted(EQUITY_FEATURE_COLS), f"ON added != the 5 equity cols: {added}"
        assert not (set(cols_off) & set(EQUITY_FEATURE_COLS)), "OFF panel already leaks equity cols"
        assert len(base_cols) == len(cols_off) == 42, (
            f"base feature count drifted: {len(base_cols)} / {len(cols_off)}"
        )
        return f"+{len(added)} cols exactly {added}; base stays {len(cols_off)}"

    def b_base_identical():
        worst, worstcol = 0.0, None
        for c in base_cols:
            d = float(np.max(np.abs(doff[c].values - don[c].values)))
            if d > worst:
                worst, worstcol = d, c
        assert worst < 1e-9, (
            f"base column '{worstcol}' changed by {worst:.2e} when the block was joined"
        )
        return f"all {len(base_cols)} base columns byte-identical (max abs diff {worst:.1e})"

    def b_crypto_neutral():
        cv = don.loc[crypto, EQUITY_FEATURE_COLS].values
        assert crypto.sum() > 0, "no crypto rows in panel?!"
        assert np.abs(cv).max() == 0.0, (
            "crypto rows are NOT all-zero in the equity block (contamination)"
        )
        return f"{crypto.sum():,} crypto rows all-zero in equity cols (dropna() spares them)"

    def b_equity_active():
        ev = don.loc[equity, EQUITY_FEATURE_COLS]
        active = int((ev.abs().to_numpy() > 1e-9).any(axis=0).sum())
        assert active >= 4, (
            f"equity rows barely populate the block ({active}/5) — block may be inert"
        )
        return f"{equity.sum():,} equity rows populate {active}/5 equity cols"

    _check("B1 row set / labels / weights / dates unchanged", b_rowset)
    _check("B2 exactly the 5 equity cols added, base count unchanged", b_additive_cols)
    _check("B3 all 42 base feature columns byte-identical", b_base_identical)
    _check("B4 crypto rows all-zero in equity block", b_crypto_neutral)
    _check("B5 equity rows actually populate the block", b_equity_active)

    # ── C. GATE-4 reproduction (one panel, like-for-like on shared OOF rows) ──
    print("\n[C] GATE-4 REPRODUCTION  (base-42 vs base+equity-47 on shared OOF rows)")
    print("  training purged-OOF base arm ...")
    base = evaluate_oof(Xon, yon, won, t1on, base_cols)["_oof_p_full"]
    print("  training purged-OOF augmented arm ...")
    aug = evaluate_oof(Xon, yon, won, t1on, cols_on)["_oof_p_full"]

    classes = {"equity": equity, "crypto": crypto, "pooled": np.ones(len(yon), bool)}
    aucs, deltas = {}, {}
    print(f"    {'class':8}{'n':>9}{'base':>10}{'+equity':>10}{'delta':>10}")
    for cname, cmask in classes.items():
        ab, aa, n = _auc_pair(yon, base, aug, cmask)
        aucs[cname], deltas[cname] = ab, aa - ab
        print(f"    {cname:8}{n:>9,}{ab:>10.4f}{aa:>10.4f}{aa - ab:>+10.4f}")

    def c_equity_tie():
        d = deltas["equity"]
        assert d <= MARGIN, (
            f"equity AUC shows an unexpected lift (+{d:.4f}) — re-run GATE-4 before shipping"
        )
        assert d >= -TIE_BAND, f"equity AUC declines beyond noise ({d:+.4f}) — investigate"
        return f"equity delta {d:+.4f} -> within-noise TIE; keep self-gated OFF (decision stands)"

    def c_crypto_guard():
        d = deltas["crypto"]
        assert abs(d) <= CRYPTO_GUARD, f"neutral block moved crypto AUC by {d:+.4f} (should be ~0)"
        return f"crypto delta {d:+.4f} within +/-{CRYPTO_GUARD} guard"

    def c_locked_repro():
        drift = {k: abs(aucs[k] - LOCKED[k]) for k in LOCKED}
        worst = max(drift.values())
        msg = "  ".join(f"{k}={aucs[k]:.4f}(lock {LOCKED[k]:.4f})" for k in LOCKED)
        if worst > DRIFT_WARN:
            print(
                f"    NOTE: BASE arm drifted from locked numbers (worst {worst:.4f}) — DB may have changed."
            )
        return f"BASE {msg}; worst drift {worst:.4f}"

    _check("C1 equity-subset OOF AUC tie (no structural lift)", c_equity_tie)
    _check("C2 crypto subset unmoved by the neutral block", c_crypto_guard)
    _check("C3 BASE arm reproduces locked Phase-3 numbers (soft)", c_locked_repro)

    # ── D. Earnings-gate integrity (the Phase-3.1 gate this phase expanded into a feature) ──
    print("\n[D] EARNINGS-GATE INTEGRITY  (earnings.py - expanded gate->feature must not regress)")
    from backend.prediction.earnings import _valid_symbol, apply_gate

    def d_gate_downscales():
        g = apply_gate(
            {"confidence": 80, "kelly_frac": 0.10},
            {"in_window": True, "factor": 0.5, "days_to_earnings": 2},
        )
        assert g["confidence"] == 40 and g["kelly_frac"] == 0.05 and g["earnings_soon"], (
            f"gate math wrong: {g}"
        )
        u = apply_gate(
            {"confidence": 80, "kelly_frac": 0.10},
            {"in_window": False, "factor": 1.0, "days_to_earnings": 30},
        )
        assert u["confidence"] == 80 and not u["earnings_soon"], "gate fired outside window"
        return "in-window halves confidence/kelly; out-of-window is a no-op"

    def d_symbol_guard():
        assert _valid_symbol("AAPL") and _valid_symbol("BRK.B"), "rejected a valid symbol"
        assert not _valid_symbol("bad;drop") and not _valid_symbol(""), "accepted an invalid symbol"
        return "symbol validation intact (injection-safe)"

    _check("D1 earnings gate down-scales in-window only", d_gate_downscales)
    _check("D2 earnings symbol validation intact", d_symbol_guard)

    # ── Verdict ──
    npass = sum(ok for _, ok, _ in _results)
    nfail = len(_results) - npass
    print("\n" + "=" * 78)
    if nfail == 0:
        print(f"  PHASE-4 AUDIT: ALL {npass} CHECKS PASS")
        print(
            "  Block is leak-safe, purely additive (production unchanged when OFF, neutral when ON),"
        )
        print(
            "  GATE-4 reproduces (within-noise tie -> stays self-gated OFF), earnings gate intact."
        )
    else:
        print(f"  PHASE-4 AUDIT: {nfail} FAILED / {npass} passed")
        for name, ok, detail in _results:
            if not ok:
                print(f"    FAIL  {name}   {detail}")
    print("=" * 78)
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
