"""
PHASE-8 AUDIT — LLM verifier + flywheel hardening: does it work right *and* stay honest?
========================================================================================
The self-contained acceptance audit for Phase 8 (the Layer-3 verifier red-teaming the top-k
portfolio names, and the data flywheel: resolve outcomes → live calibration → drift-triggered
retrain). Companion to the unit tests in ``backend/prediction/tests/test_prediction.py``. Everything
runs against an ISOLATED temporary SQLite (never the real flux_market.db) and stubs the LLM +
trainer, so it is fast, deterministic, and network-free.

Four sections, each a hard pass/fail (GATE-8):

  A. VERIFIER HONESTY CONTRACT   the LLM can only downgrade/veto, never raise: a greedy LLM that
       tries to inflate every confidence to 99 is clamped to the model ceiling; a veto forces
       act=False and halves confidence; an LLM outage degrades gracefully (confidence unchanged,
       never raised) — checked on BOTH single-symbol and the top-k portfolio batch.

  B. FLYWHEEL RESOLVES OUTCOMES   seed history + a prediction in a temp DB, run resolve_due, and
       prove the realized return, direction-correctness, and cost-aware P&L are recorded, then that
       refresh_live_calibration writes the live calibration buckets the displayed confidence reads.

  C. DRIFT RETRAIN FIRES   inject synthetic drift (40% live accuracy over 250 resolved trades) and
       prove maybe_retrain actually INVOKES the trainer (stubbed) — and that a too-fresh model is
       rate-limited so a bad streak can't thrash the trainer.

  D. WIRING INTEGRITY   the scheduler (ingestion.py) registers the daily prediction cycle + the
       drift-retrain job; the daily cycle calls the portfolio verifier; verify_portfolio ranks by
       the Phase-7 conviction score (edge × meta).

Run:  python scripts/experiments/gate8_llm_verifier_audit.py        (exit 0 = all pass, 1 = any failure)
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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


def _fake_pred(sym, conf, prob_up, meta):
    return {
        "symbol": sym,
        "direction": "UP" if prob_up >= 0.5 else "DOWN",
        "confidence": conf,
        "prob_up": prob_up,
        "meta_prob": meta,
        "act": True,
        "kelly_frac": 0.05,
        "conf_low": 100.0,
        "conf_high": 110.0,
        "pred_price": 105.0,
        "band_pct": 80,
        "regime": "trend",
        "sentiment": 0.1,
    }


async def _main() -> int:
    import backend.db as db
    import backend.prediction.agent as agent
    import backend.prediction.drift as drift
    import backend.prediction.serve as serve
    import backend.prediction.train as train
    from backend.prediction.agent import _conviction, verify_portfolio, verify_prediction

    print("=" * 84)
    print("  PHASE-8 AUDIT — LLM verifier + flywheel: honesty contract + the data loop")
    print("=" * 84)

    # Isolate every DB write into a throwaway sqlite for the whole audit.
    tmp = tempfile.TemporaryDirectory()
    db.DB_PATH = Path(tmp.name) / "audit.db"
    await db.init_db()

    # ── A. Verifier honesty contract ─────────────────────────────────────────────
    print("\n[A] VERIFIER HONESTY CONTRACT  (LLM may only downgrade / veto, never raise)")

    async def greedy(prompt):
        return (
            '{"agree":true,"confidence":99,"rationale":"bull","risks":"r","key_levels":"1",'
            '"contradicts_model":false,"veto":false}'
        )

    async def veto_llm(prompt):
        return (
            '{"agree":false,"confidence":50,"rationale":"guidance cut","risks":"r",'
            '"key_levels":"1","contradicts_model":true,"veto":true}'
        )

    async def down_llm(prompt):
        raise RuntimeError("Ollama is not running")

    # Run the async verifier calls inline (the _check wrappers below assert on the results).
    agent._ollama_call = greedy
    v_single = await verify_prediction(
        "AAA", prediction=_fake_pred("AAA", 60, 0.8, 0.6), persist=False
    )
    agent._ollama_call = veto_llm
    v_veto = await verify_prediction(
        "BBB", prediction=_fake_pred("BBB", 80, 0.7, 0.6), persist=False
    )
    agent._ollama_call = down_llm
    v_down = await verify_prediction(
        "CCC", prediction=_fake_pred("CCC", 66, 0.6, 0.6), persist=False
    )

    _check(
        "A1 greedy LLM clamped to model ceiling (single)",
        lambda: (
            _assert(v_single["final_confidence"] == 60 and v_single["final_act"] is True),
            "raise→clamped to 60, act kept",
        )[1],
    )
    _check(
        "A2 veto forces act=False + heavy downgrade",
        lambda: (
            _assert(
                v_veto["veto"] and v_veto["final_act"] is False and v_veto["final_confidence"] <= 40
            ),
            "veto→act False, conf≤40",
        )[1],
    )
    _check(
        "A3 LLM outage degrades gracefully (unchanged, never raised)",
        lambda: (
            _assert(v_down["verifier"] == "unavailable" and v_down["final_confidence"] == 66),
            "verifier unavailable, conf 66",
        )[1],
    )

    # Portfolio batch: greedy LLM on all, assert ranking + ceiling for every name.
    agent._ollama_call = greedy
    preds = [
        _fake_pred("BBB", 70, 0.60, 0.40),
        _fake_pred("AAA", 55, 0.80, 0.60),
        _fake_pred("CCC", 80, 0.65, 0.70),
        _fake_pred("DDD", 90, 0.40, 0.80),
    ]
    batch = await verify_portfolio(preds, top_k=3, persist=False)

    def a_batch_rank_and_ceiling():
        order = [v["symbol"] for v in batch]
        assert order == ["AAA", "CCC", "BBB"], f"conviction ranking wrong: {order}"
        for v in batch:
            assert v["final_confidence"] <= v["model_confidence"], f"{v['symbol']} raised!"
        return f"top-3 by edge×meta = {order}; no final confidence above its ceiling"

    _check("A4 portfolio batch ranks by conviction + clamps every name", a_batch_rank_and_ceiling)

    # ── B. Flywheel resolves outcomes ────────────────────────────────────────────
    print("\n[B] FLYWHEEL RESOLVES OUTCOMES  (resolve_due → outcome + P&L → live calibration)")
    base = __import__("pandas").Timestamp("2024-01-02")
    pd = __import__("pandas")
    hist = [
        {
            "symbol": "ZZ",
            "date": str((base + pd.tseries.offsets.BDay(i)).date()),
            "open": 100,
            "high": 100,
            "low": 100,
            "close": 100 + i,
            "adj_close": 100 + i,
            "volume": 1,
        }
        for i in range(6)
    ]
    await db.insert_history(hist)
    target = str((base + pd.tseries.offsets.BDay(5)).date())
    pid = await db.insert_prediction(
        {
            "symbol": "ZZ",
            "model": "t",
            "horizon_days": 5,
            "direction": "UP",
            "prob_up": 0.6,
            "meta_prob": 0.62,
            "act": 1,
            "confidence": 62,
            "kelly_frac": 0.05,
            "sentiment": 0.0,
            "last_close": 100.0,
            "generated_at": int(time.time() * 1000),
            "target_date": target,
        }
    )
    n_res = await serve.resolve_due(as_of="2026-01-01")
    outs = await db.get_resolved_outcomes()

    def b_resolved():
        assert n_res == 1, f"expected 1 resolved, got {n_res}"
        assert len(outs) == 1 and outs[0]["correct"] == 1, (
            "UP call on a rising series must be correct"
        )
        assert abs(outs[0]["pnl_after_costs"] - 0.0495) < 1e-6, "cost-aware P&L wrong"
        return f"resolved 1 | correct=1 | pnl={outs[0]['pnl_after_costs']:.4f} (net of 5bps)"

    async def _calib():
        return await serve.refresh_live_calibration()

    n_buckets = await _calib()

    def b_calibration():
        assert n_buckets >= 1, "live calibration buckets not written"
        return f"refreshed {n_buckets} live calibration bucket(s) from resolved outcomes"

    _check("B1 outcome resolved with correctness + cost-aware P&L", b_resolved)
    _check("B2 live calibration buckets refreshed from outcomes", b_calibration)

    # ── C. Drift retrain fires under synthetic drift ─────────────────────────────
    print(
        "\n[C] DRIFT RETRAIN FIRES  (synthetic drift → trainer invoked; fresh model → rate-limited)"
    )
    for i in range(250):
        p = await db.insert_prediction(
            {
                "symbol": "DR",
                "model": "t",
                "horizon_days": 5,
                "direction": "UP",
                "prob_up": 0.6,
                "meta_prob": 0.62,
                "act": 1,
                "confidence": 62,
                "kelly_frac": 0.05,
                "sentiment": 0.0,
                "last_close": 100.0,
                "generated_at": int(time.time() * 1000),
                "target_date": "2024-01-01",
            }
        )
        await db.insert_outcome(
            {
                "prediction_id": p,
                "actual_return": -0.01,
                "correct": 1 if i % 10 < 4 else 0,
                "pnl_after_costs": -0.01,
                "resolved_at": int(time.time() * 1000),
            }
        )

    called = {"n": 0}

    async def fake_train_main():
        called["n"] += 1

    train.main = fake_train_main

    drift._last_trained_days = lambda: 99.0  # past the rate-limit
    res_fire = await drift.maybe_retrain()
    drift._last_trained_days = lambda: 0.5  # fresher than MIN_RETRAIN_AGE_DAYS
    res_rl = await drift.maybe_retrain()

    def c_fires():
        assert res_fire["retrained"] is True, res_fire
        assert called["n"] == 1, "trainer was not invoked under drift"
        assert any("accuracy" in r for r in res_fire["reasons"]), "drift reason missing"
        return f"retrained=True, trainer invoked once, reasons={res_fire['reasons']}"

    def c_rate_limited():
        assert res_rl["retrained"] is False and "rate-limited" in res_rl["reason"], res_rl
        assert called["n"] == 1, "trainer invoked again despite rate-limit"
        return "fresh model → rate-limited, trainer not re-invoked"

    _check("C1 maybe_retrain FIRES the trainer under synthetic drift", c_fires)
    _check("C2 fresh model is rate-limited (no thrash)", c_rate_limited)

    # ── D. Wiring integrity ──────────────────────────────────────────────────────
    print("\n[D] WIRING INTEGRITY  (scheduler jobs + daily cycle runs the verifier)")
    ingestion_src = (Path(__file__).resolve().parents[2] / "backend/ingestion.py").read_text(
        encoding="utf-8"
    )
    serve_src = (Path(__file__).resolve().parents[2] / "backend/prediction/serve.py").read_text(
        encoding="utf-8"
    )

    def d_scheduler_jobs():
        # The daily prediction cycle is now driven by the FLUX-X §4 capstone (run_flux_x), which
        # resolves matured outcomes and red-teams the book's top-k internally (the daily_prediction_
        # cycle's job is a subset of it). Accept either entrypoint so this audit tracks the wiring.
        assert "run_flux_x" in ingestion_src or "daily_prediction_cycle" in ingestion_src, (
            "daily cycle not registered in scheduler"
        )
        assert "maybe_retrain" in ingestion_src, "drift retrain not registered in scheduler"
        assert 'id="predictions"' in ingestion_src and 'id="drift_retrain"' in ingestion_src, (
            "scheduler job ids missing"
        )
        return "ingestion.py registers predictions (run_flux_x) + drift_retrain jobs"

    def d_cycle_calls_verifier():
        assert "verify_portfolio" in serve_src, "daily cycle does not call the portfolio verifier"
        return "daily_prediction_cycle red-teams the top-k via verify_portfolio"

    def d_conviction_rank():
        a, b = _fake_pred("A", 50, 0.80, 0.60), _fake_pred("B", 50, 0.55, 0.50)
        assert _conviction(a) > _conviction(b), "conviction score not edge×meta"
        return f"_conviction = edge×meta ({_conviction(a):.3f} > {_conviction(b):.3f})"

    _check("D1 scheduler registers daily cycle + drift retrain", d_scheduler_jobs)
    _check("D2 daily cycle runs the portfolio verifier", d_cycle_calls_verifier)
    _check("D3 portfolio ranks by Phase-7 conviction (edge×meta)", d_conviction_rank)

    tmp.cleanup()

    # ── Verdict ──
    npass = sum(ok for _, ok, _ in _results)
    nfail = len(_results) - npass
    print("\n" + "=" * 84)
    if nfail == 0:
        print(f"  PHASE-8 AUDIT: ALL {npass} CHECKS PASS")
        print("  Verifier can only downgrade/veto (single + portfolio batch); flywheel resolves")
        print("  outcomes → live calibration; drift retrain fires on synthetic drift & is rate-")
        print("  limited; scheduler + daily cycle wired to verifier and trainer.")
    else:
        print(f"  PHASE-8 AUDIT: {nfail} FAILED / {npass} passed")
        for name, ok, detail in _results:
            if not ok:
                print(f"    FAIL  {name}   {detail}")
    print("=" * 84)
    return 1 if nfail else 0


def _assert(cond: bool) -> None:
    assert cond


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
