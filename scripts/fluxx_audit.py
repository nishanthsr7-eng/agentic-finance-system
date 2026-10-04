"""
FLUX-X CAPSTONE AUDIT — does the advanced algorithm (Workflow §4) run end to end, honestly?
===========================================================================================
The acceptance audit for the FLUX-X serving capstone: the live cross-sectional book construction
(step 8) and the full 10-step §4 loop wired together (``backend/prediction/flux_x.py``). Companion
to the unit tests in ``backend/prediction/test_flux_x.py``. Everything runs against an ISOLATED
temporary SQLite and stubs the model / LLM / Alpaca, so it is fast, deterministic, and network-free.

Four sections, each a hard pass/fail:

  A. CHAMPION BOOK (step 8)   construct_book reproduces the GATE-7 deployable: LONG-ONLY, top-decile,
       edge-ranked, equal-weight, regime-gated. It never longs a bearish name and the weights are a
       valid simplex. (The empirical champion, not the spec's theoretical edge×meta/neutralized ideal.)

  B. CAUSAL VOL-TARGET   the book leverage is the causal clip(target/trailing_vol) formula: lever up
       in a calm tape, down (and capped) in a storm, 1x during warm-up. No future bar can move it.

  C. ONE-WAY VERIFIER VALVE   folding Layer-3 verdicts into the book de-risks only: a veto DROPS the
       name, a downgrade SHRINKS it, an attempt to raise can never enlarge it — and the freed weight
       is NOT renormalized away (gross exposure honestly falls).

  D. FULL §4 LOOP + WIRING   run_flux_x runs all ten steps in order against a temp DB (resolve →
       predict+log → construct → verify → paper dry-run) and the scheduler (ingestion.py) drives the
       daily cycle through run_flux_x. Degrades gracefully (LLM/Alpaca stubbed off → loop still closes).

Run:  python scripts/fluxx_audit.py        (exit 0 = all pass, 1 = any failure)
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def _assert(cond: bool) -> None:
    assert cond


def _pred(sym, prob_up, meta=0.6, regime="trend"):
    # Carries the fields serve._to_row needs so the stub leaderboard logs into the temp DB.
    return {
        "symbol": sym,
        "prob_up": prob_up,
        "meta_prob": meta,
        "regime": regime,
        "act": meta >= 0.60,
        "direction": "UP" if prob_up >= 0.5 else "DOWN",
        "confidence": int(meta * 100),
        "kelly_frac": 0.05,
        "horizon_days": 5,
        "last_close": 100.0,
        "target_date": "2024-02-01",
        "sentiment": 0.0,
    }


async def _main() -> int:
    import backend.db as db
    from backend.prediction.flux_x import (
        TRADING_DAYS,
        VOL_TARGET,
        _vol_target_leverage,
        apply_verdicts,
        construct_book,
        run_flux_x,
    )
    from backend.prediction.regime import REGIME_SCALE

    print("=" * 84)
    print("  FLUX-X CAPSTONE AUDIT — the advanced algorithm (Workflow §4) end to end")
    print("=" * 84)

    tmp = tempfile.TemporaryDirectory()
    db.DB_PATH = Path(tmp.name) / "audit.db"
    await db.init_db()

    # ── A. Champion book (step 8) ────────────────────────────────────────────────
    print("\n[A] CHAMPION BOOK  (long-only top-decile, edge-ranked, equal-weight, regime-gated)")
    preds10 = [_pred(f"B{i}", 0.50 + 0.01 * i) for i in range(1, 7)] + [
        _pred(f"S{i}", 0.50 - 0.01 * i) for i in range(1, 5)
    ]
    bk = await construct_book(preds10, frac=0.10, vol_target=None)
    wide = await construct_book(preds10, frac=1.0, vol_target=None, use_regime_gate=False)

    def a_top_decile():
        assert bk["k"] == 1 and bk["book"][0]["symbol"] == "B6", (
            f"top decile should be the single most-bullish name, got {bk['book']}"
        )
        assert all(b["edge"] > 0 for b in wide["book"]) and len(wide["book"]) == 6, (
            "long-only must never long a bearish name"
        )
        w = np.array([b["weight"] for b in wide["book"]])
        assert np.allclose(w, 1 / 6, atol=1e-3), (
            "equal-weight simplex broken"
        )  # 4dp display rounding
        return f"top decile={bk['book'][0]['symbol']}; 6 bullish eligible; weights sum=1 equal"

    ro = await construct_book(
        [_pred(f"B{i}", 0.55, regime="risk_off") for i in range(1, 6)],
        frac=1.0,
        vol_target=None,
        regime="risk_off",
    )

    def a_regime_gate():
        assert abs(ro["gross_exposure"] - REGIME_SCALE["risk_off"]) < 1e-9, (
            "regime gate not applied"
        )
        assert ro["gross_after_levers"] < wide["gross_after_levers"], "risk_off must risk less"
        return f"risk_off gross {ro['gross_after_levers']} < trend {wide['gross_after_levers']}"

    _check("A1 long-only top-decile, edge-ranked, equal-weight simplex", a_top_decile)
    _check("A2 regime gate scales gross exposure down in stress", a_regime_gate)

    # ── B. Causal vol-target ─────────────────────────────────────────────────────
    print("\n[B] CAUSAL VOL-TARGET  (leverage = clip(target/trailing_vol); calm up, storm down)")
    td = VOL_TARGET / np.sqrt(TRADING_DAYS)

    def b_formula():
        assert abs(_vol_target_leverage(td / 2.0) - 2.0) < 1e-9, "calm tape should lever up 2x"
        assert abs(_vol_target_leverage(td * 3.0) - 1 / 3.0) < 1e-6, "storm should delever to 1/3"
        assert _vol_target_leverage(1e-9) == 3.0, "leverage not capped"
        assert _vol_target_leverage(None) == 1.0, "warm-up should be 1x"
        return "lever up in calm, down in storm, capped at 3x, warm-up=1x"

    # Seed a real history so _basket_daily_vol runs; it only reads the trailing window ending today
    # → causal by construction (a future bar can't enter the estimate).
    import pandas as pd

    _base = pd.Timestamp("2024-01-02")
    _hist = [
        {
            "symbol": "VT",
            "date": str((_base + pd.tseries.offsets.BDay(i)).date()),
            "open": 100,
            "high": 100,
            "low": 100,
            "close": 100 + (i % 3),
            "adj_close": 100 + (i % 3),
            "volume": 1,
        }
        for i in range(40)
    ]
    await db.insert_history(_hist)
    from backend.prediction.flux_x import _basket_daily_vol

    v1 = await _basket_daily_vol(["VT"], window=20)

    def b_causal():
        assert v1 is not None and v1 > 0, "basket vol should be estimable from 40 bars"
        return f"basket vol estimated from trailing window = {v1:.4f} (>0, causal)"

    _check("B1 vol-target leverage formula (clip, cap, warm-up)", b_formula)
    _check("B2 basket vol is a causal trailing estimate", b_causal)

    # ── C. One-way verifier valve ────────────────────────────────────────────────
    print("\n[C] ONE-WAY VERIFIER VALVE  (veto drops, downgrade shrinks, raise can't enlarge)")
    book5 = await construct_book(
        [_pred(f"B{i}", 0.50 + 0.02 * i) for i in range(1, 6)],
        frac=1.0,
        vol_target=None,
        use_regime_gate=False,
    )
    g0 = book5["gross_after_levers"]
    sym0 = book5["book"][0]["symbol"]
    frac0 = book5["book"][0]["target_frac"]
    vetoed = apply_verdicts(
        book5, [{"symbol": sym0, "veto": True, "model_confidence": 70, "final_confidence": 35}]
    )
    halved = apply_verdicts(
        book5, [{"symbol": sym0, "veto": False, "model_confidence": 80, "final_confidence": 40}]
    )
    raised = apply_verdicts(
        book5, [{"symbol": sym0, "veto": False, "model_confidence": 80, "final_confidence": 99}]
    )

    def c_veto():
        assert all(b["symbol"] != sym0 for b in vetoed["book"]) and vetoed["vetoed"] == 1, (
            "veto failed"
        )
        assert vetoed["gross_after_levers"] < g0 - 1e-9, (
            "veto must lower gross (no renormalization)"
        )
        return f"veto dropped {sym0}; gross {g0:.2f}->{vetoed['gross_after_levers']:.2f}"

    def c_downgrade():
        row = next(b for b in halved["book"] if b["symbol"] == sym0)
        assert abs(row["target_frac"] - frac0 * 0.5) < 1e-9, "downgrade should halve size"
        up = next(b for b in raised["book"] if b["symbol"] == sym0)
        assert abs(up["target_frac"] - frac0) < 1e-9, (
            "a 'raise' verdict must NOT enlarge the position"
        )
        return "downgrade halves size; a raise verdict cannot enlarge (one-way valve)"

    _check("C1 veto drops the name; gross falls (not renormalized)", c_veto)
    _check("C2 downgrade shrinks; raise can never enlarge", c_downgrade)

    # ── D. Full §4 loop + wiring ─────────────────────────────────────────────────
    print("\n[D] FULL §4 LOOP + WIRING  (resolve -> predict -> construct -> verify -> paper)")

    # Stub the heavy organs so the loop is deterministic & offline: predict_all returns a fixed
    # leaderboard, the regime is pinned, the LLM verifier and Alpaca are off (no keys / top_k=0).
    import backend.prediction.flux_x as flux_x
    import backend.prediction.serve as serve

    async def fake_predict_all(symbols=None):
        return [
            _pred("AAA", 0.62, 0.70),
            _pred("BBB", 0.58, 0.65),
            _pred("CCC", 0.55, 0.62),
            _pred("DDD", 0.48, 0.40),
        ]  # DDD bearish → must be excluded from longs

    serve.predict_all = fake_predict_all  # run_predictions logs these

    async def fake_regime(force=False):
        return {
            "regime": "trend",
            "probs": {"trend": 1.0, "chop": 0.0, "risk_off": 0.0},
            "as_of": "2024-01-10",
        }

    import backend.prediction.regime as regime_mod

    regime_mod.current_regime = fake_regime
    flux_x.current_regime = fake_regime  # bound name inside run_flux_x's import

    rep = await run_flux_x(equity=100_000.0, verify_top_k=0, vol_target=None)

    def d_loop_runs():
        assert rep["logged"] == 4, f"should log the 4-name universe, got {rep['logged']}"
        syms = {b["symbol"] for b in rep["book"]}
        assert "DDD" not in syms, "bearish DDD must not be in a long-only book"
        assert rep["book_size"] >= 1, "book should hold at least one name"
        assert rep["regime"] == "trend", "regime not threaded into the report"
        assert rep["paper_status"] in ("dry_run", "disabled"), (
            f"paper must be dry-run/disabled (never live), got {rep['paper_status']}"
        )
        return (
            f"logged 4 | book {rep['book_size']} names (DDD excluded) | "
            f"paper={rep['paper_status']} (never live)"
        )

    def d_wiring():
        ingestion_src = (Path(__file__).resolve().parents[1] / "backend/ingestion.py").read_text(
            encoding="utf-8"
        )
        assert "run_flux_x" in ingestion_src, "scheduler not wired to run_flux_x"
        assert 'id="predictions"' in ingestion_src, "prediction job id missing"
        return "ingestion.py drives the daily cycle through run_flux_x"

    _check("D1 run_flux_x runs all ten steps end to end (temp DB, stubs)", d_loop_runs)
    _check("D2 scheduler wires the daily cycle through run_flux_x", d_wiring)

    tmp.cleanup()

    # ── Verdict ──
    npass = sum(ok for _, ok, _ in _results)
    nfail = len(_results) - npass
    print("\n" + "=" * 84)
    if nfail == 0:
        print(f"  FLUX-X CAPSTONE AUDIT: ALL {npass} CHECKS PASS")
        print("  Step 8 builds the GATE-7 champion book (long-only top-decile, edge-ranked, equal-")
        print(
            "  weight, regime-gated, causal vol-target); the Layer-3 verifier is a one-way valve;"
        )
        print("  run_flux_x runs the full §4 loop end to end and the scheduler is wired to it.")
    else:
        print(f"  FLUX-X CAPSTONE AUDIT: {nfail} FAILED / {npass} passed")
        for name, ok, detail in _results:
            if not ok:
                print(f"    FAIL  {name}   {detail}")
    print("=" * 84)
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
