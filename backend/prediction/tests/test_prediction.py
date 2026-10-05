"""
FLUX Prediction — Test Suite (serve / agent / conformal / garch / ensemble / safety)
====================================================================================
Fast, deterministic, network-free tests for the prediction stack. Async code is driven via
asyncio.run() so no pytest-asyncio plugin is needed. DB-touching tests run against an isolated
temporary SQLite (db.DB_PATH is monkeypatched), never the real flux_market.db.

Run:  pytest backend/prediction/tests/test_prediction.py -q
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def run(coro):
    return asyncio.run(coro)


# ── Temp-DB fixture (isolates every DB test from the real database) ──────────────
@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    import backend.db as db

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    run(db.init_db())
    return db


# ── GARCH (pure) ─────────────────────────────────────────────────────────────────
def test_garch_tracks_vol_cluster():
    from backend.prediction.garch import conditional_vol, forecast_h_vol

    rng = np.random.default_rng(0)
    vol = np.concatenate([np.full(400, 0.01), np.full(200, 0.04), np.full(400, 0.01)])
    rets = pd.Series(rng.normal(0, vol), index=pd.date_range("2015-01-01", periods=1000, freq="B"))
    cv = conditional_vol(rets)
    assert cv.iloc[450:550].mean() > cv.iloc[100:300].mean()  # stressed σ > calm σ
    assert forecast_h_vol(rets, 5) > 0


def test_garch_short_series_fallback():
    from backend.prediction.garch import conditional_vol

    s = pd.Series(np.random.default_rng(1).normal(0, 0.01, 50))  # < MIN_OBS → EWMA fallback
    cv = conditional_vol(s)
    assert len(cv) == len(s) and np.isfinite(cv.dropna()).all()


# ── Conformal (pure) ─────────────────────────────────────────────────────────────
def test_conformal_quantile_finite_sample():
    from backend.prediction.conformal import conformal_quantile

    scores = np.linspace(0, 1, 100)
    q = conformal_quantile(scores, alpha=0.2)
    assert 0.75 <= q <= 0.85  # ~80th percentile
    assert conformal_quantile(np.array([1.0, 2.0]), alpha=0.001) == float("inf")  # too few → inf


def test_conformal_coverage_holds():
    from backend.prediction.conformal import ConformalBands, empirical_coverage

    rng = np.random.default_rng(2)
    scale = rng.uniform(0.01, 0.05, 4000)
    resid = rng.normal(0, scale)
    bands = ConformalBands.fit(resid[:2000], scale[:2000], horizon=5, alphas=(0.2,))
    cov = empirical_coverage(np.abs(resid[2000:]) / scale[2000:], bands.q[0.2])
    assert abs(cov - 0.8) < 0.03  # 80% band covers ~80%
    lo, hi = bands.interval(0.0, 0.02, alpha=0.2)
    assert lo < 0 < hi


# ── Sizing (pure) ────────────────────────────────────────────────────────────────
def test_kelly_fraction_bounds():
    from backend.prediction.sizing import kelly_fraction

    assert kelly_fraction(0.5) == 0.0  # no edge → no bet
    assert kelly_fraction(0.4) == 0.0  # negative edge → no bet
    assert 0 < kelly_fraction(0.6) <= 0.5  # positive edge, capped
    assert kelly_fraction(0.99) <= 0.5  # cap respected


# ── Earnings gate (pure + security) ──────────────────────────────────────────────
def test_earnings_apply_gate_downscales():
    from backend.prediction.earnings import apply_gate

    pred = {"confidence": 80, "kelly_frac": 0.10}
    gated = apply_gate(pred, {"in_window": True, "factor": 0.5, "days_to_earnings": 2})
    assert gated["confidence"] == 40 and gated["kelly_frac"] == 0.05 and gated["earnings_soon"]
    ungated = apply_gate(pred, {"in_window": False, "factor": 1.0, "days_to_earnings": 30})
    assert ungated["confidence"] == 80 and not ungated["earnings_soon"]


def test_earnings_rejects_bad_symbol():
    from backend.prediction.earnings import _valid_symbol

    assert _valid_symbol("AAPL") and _valid_symbol("BRK.B")
    assert not _valid_symbol("bad;drop") and not _valid_symbol("a" * 20) and not _valid_symbol("")


# ── Ensemble (pure/synthetic) ────────────────────────────────────────────────────
def test_stacker_trains_and_bounds_proba():
    from backend.prediction.ensemble import FEATURES, train_stack

    rng = np.random.default_rng(3)
    n = 2000
    primary = rng.uniform(0.3, 0.7, n)
    y = (rng.uniform(size=n) < primary).astype(int)  # y correlates with primary
    feats = pd.DataFrame(
        {
            "primary_cal": primary,
            "meta_prob": rng.uniform(0.4, 0.6, n),
            "mag_oof": rng.normal(0, 0.02, n),
            "regime_scale": 1.0,
        }
    )
    stacker, rep = train_stack(feats, y)
    p = stacker.predict_proba(0.6, 0.55, 0.01, "trend")
    assert 0.0 <= p <= 1.0 and set(rep["coef"]) == set(FEATURES)


# ── FRED transform (pure) ────────────────────────────────────────────────────────
def test_fred_transform():
    from backend.prediction.fred import compute_fred_features

    idx = pd.to_datetime(["2020-01-01", "2021-01-01"])
    raw = pd.DataFrame(
        {"dgs10": [1.8, 1.1], "dgs2": [1.5, 0.2], "fed_funds": [1.6, 0.1], "cpi": [258.0, 262.0]},
        index=idx,
    )
    feat = compute_fred_features(raw)
    assert abs(feat["term_spread"].iloc[0] - 0.30) < 1e-9
    assert "cpi_yoy" in feat.columns
    assert compute_fred_features(pd.DataFrame()).empty  # no data → empty (no-op)


# ── Alpaca paper (pure planner + safety) ─────────────────────────────────────────
def test_paper_planner_validates_and_clamps():
    from backend.prediction.paper import MAX_NOTIONAL

    preds = [
        {"symbol": "AAPL", "act": True, "direction": "UP", "kelly_frac": 0.05},
        {"symbol": "NVDA", "act": False, "direction": "UP", "kelly_frac": 0.50},  # not acted
        {"symbol": "bad;x", "act": True, "direction": "UP", "kelly_frac": 0.05},  # bad symbol
        {"symbol": "TSLA", "act": True, "direction": "DOWN", "kelly_frac": 9.0},  # clamp
    ]
    orders = run_plan(preds)
    syms = {o["symbol"]: o for o in orders}
    assert "NVDA" not in syms and "BAD;X" not in syms and "bad;x" not in syms
    assert syms["AAPL"]["side"] == "buy" and syms["TSLA"]["side"] == "sell"
    assert syms["TSLA"]["notional"] <= MAX_NOTIONAL  # per-order clamp enforced


def run_plan(preds):
    from backend.prediction.paper import _plan_orders

    return _plan_orders(preds, equity=100_000)


def test_paper_disabled_without_keys():
    from backend.prediction.paper import submit_orders

    res = run(
        submit_orders([{"symbol": "AAPL", "act": True, "direction": "UP", "kelly_frac": 0.05}])
    )
    assert res["status"] == "disabled"  # empty keys → no trading


# ── Agent verifier enforcement (stubbed LLM — the honesty contract) ──────────────
def test_agent_cannot_raise_and_can_veto(monkeypatch):
    import backend.prediction.agent as agent

    async def fake_predict(symbol, **kw):
        return {
            "symbol": symbol,
            "direction": "UP",
            "confidence": 60,
            "meta_prob": 0.62,
            "act": True,
            "kelly_frac": 0.05,
            "conf_low": 100.0,
            "conf_high": 110.0,
            "pred_price": 105.0,
            "band_pct": 80,
            "regime": "trend",
            "sentiment": 0.1,
        }

    monkeypatch.setattr(agent, "predict", fake_predict)

    async def stub(resp):
        async def _f(prompt):
            return resp

        monkeypatch.setattr(agent, "_ollama_call", _f)

    # LLM tries to RAISE → clamped to model ceiling (60).
    run(
        stub(
            '{"agree":true,"confidence":90,"rationale":"x","risks":"y","key_levels":"1",'
            '"contradicts_model":false,"veto":false}'
        )
    )
    v = run(agent.verify_prediction("AAPL", persist=False))
    assert v["final_confidence"] == 60 and v["final_act"] is True

    # LLM VETO → act False, confidence heavily downgraded.
    run(
        stub(
            '{"agree":false,"confidence":55,"rationale":"cut","risks":"e","key_levels":"1",'
            '"contradicts_model":true,"veto":true}'
        )
    )
    v = run(agent.verify_prediction("AAPL", persist=False))
    assert v["veto"] is True and v["final_act"] is False and v["final_confidence"] <= 30


def test_agent_graceful_when_ollama_down(monkeypatch):
    import backend.prediction.agent as agent

    async def fake_predict(symbol, **kw):
        return {
            "symbol": symbol,
            "direction": "UP",
            "confidence": 66,
            "meta_prob": 0.6,
            "act": True,
            "kelly_frac": 0.05,
            "regime": "trend",
        }

    async def boom(prompt):
        raise RuntimeError("Ollama is not running")

    monkeypatch.setattr(agent, "predict", fake_predict)
    monkeypatch.setattr(agent, "_ollama_call", boom)
    v = run(agent.verify_prediction("AAPL", persist=False))
    assert v["verifier"] == "unavailable" and v["final_confidence"] == 66  # never raised


# ── serve resolve (temp DB, full flywheel math) ──────────────────────────────────
def test_serve_resolve_and_pnl(tmp_db):
    from backend.prediction.serve import resolve_due

    # Seed history so the realized return is computable: 100 → 105 over 5 bars.
    base = pd.Timestamp("2024-01-02")
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
    run(tmp_db.insert_history(hist))
    target = str((base + pd.tseries.offsets.BDay(5)).date())
    run(
        tmp_db.insert_prediction(
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
    )
    n = run(resolve_due(as_of="2026-01-01"))
    assert n == 1
    outs = run(tmp_db.get_resolved_outcomes())
    assert len(outs) == 1
    o = outs[0]
    assert o["correct"] == 1  # UP call, price rose → correct
    # pnl = side*ret - cost = (105/100-1) - 0.0005 = 0.0495
    assert abs(o["pnl_after_costs"] - 0.0495) < 1e-6


def test_serve_resolve_records_in_band(tmp_db):
    """resolve_due marks whether the realised close landed inside the 80% range."""
    from backend.prediction.serve import resolve_due

    base = pd.Timestamp("2024-01-02")
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
    run(tmp_db.insert_history(hist))
    target = str((base + pd.tseries.offsets.BDay(5)).date())  # close there is 105
    common = {
        "symbol": "ZZ",
        "model": "t",
        "horizon_days": 5,
        "direction": "UP",
        "prob_up": 0.6,
        "meta_prob": 0.62,
        "act": 0,
        "confidence": 62,
        "kelly_frac": 0.0,
        "sentiment": 0.0,
        "last_close": 100.0,
        "generated_at": int(time.time() * 1000),
        "target_date": target,
    }
    run(
        tmp_db.insert_prediction(
            {
                **common,
                "conf_low": 95.0,
                "conf_high": 110.0,
                "conf_low_90": 90.0,
                "conf_high_90": 115.0,
            }
        )
    )  # 105 inside
    run(tmp_db.insert_prediction({**common, "conf_low": 98.0, "conf_high": 102.0}))  # 105 outside
    run(tmp_db.insert_prediction(common))  # no band
    assert run(resolve_due(as_of="2026-01-01")) == 3
    rows = sorted(run(tmp_db.get_prediction_history("ZZ")), key=lambda r: r["id"])
    assert [r["in_band"] for r in rows] == [1, 0, None]
    assert rows[0]["conf_low_90"] == 90.0 and rows[0]["conf_high_90"] == 115.0


def test_conformal_wide_band_contains_default_band():
    """The 90% range served next to the 80% one is strictly wider around the same centre."""
    from backend.prediction.conformal import ConformalBands
    from backend.prediction.predict import BAND_ALPHA, MODELS_DIR, WIDE_ALPHA

    path = MODELS_DIR / "conformal.pkl"
    if not path.exists():
        pytest.skip("no shipped conformal.pkl")
    bands = ConformalBands.load(path)
    assert WIDE_ALPHA in bands.q, "conformal.pkl lacks the 90% level"
    lo80, hi80 = bands.interval(0.0, 0.05, alpha=BAND_ALPHA)
    lo90, hi90 = bands.interval(0.0, 0.05, alpha=WIDE_ALPHA)
    assert lo90 < lo80 < 0 < hi80 < hi90


# ── drift detection (temp DB) ────────────────────────────────────────────────────
def test_drift_flags_poor_live_accuracy(tmp_db):
    from backend.prediction.drift import check_drift

    for i in range(250):
        pid = run(
            tmp_db.insert_prediction(
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
                    "target_date": "2024-01-01",
                }
            )
        )
        run(
            tmp_db.insert_outcome(
                {
                    "prediction_id": pid,
                    "actual_return": -0.01,
                    "correct": 1 if i % 10 < 4 else 0,  # 40% accuracy
                    "pnl_after_costs": -0.01,
                    "resolved_at": int(time.time() * 1000),
                }
            )
        )
    d = run(check_drift())
    assert d["drift"] is True and d["metrics"]["status"] == "ok"
    assert any("accuracy" in r for r in d["reasons"])


def test_drift_insufficient_data_is_quiet(tmp_db):
    from backend.prediction.drift import check_drift

    d = run(check_drift())
    assert d["drift"] is False and d["metrics"]["status"] == "insufficient"


# ── Phase 8: drift-TRIGGERED retrain actually fires under synthetic drift ─────────
def test_drift_retrain_fires_on_synthetic_drift(tmp_db, monkeypatch):
    """GATE-8: inject synthetic drift (40% live accuracy) and prove maybe_retrain FIRES the
    trainer — not just that check_drift flags it. Trainer + rate-limit are stubbed so the test
    stays fast and never touches the real models."""
    import backend.prediction.train as train
    from backend.prediction import drift

    for i in range(250):
        pid = run(
            tmp_db.insert_prediction(
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
                    "target_date": "2024-01-01",
                }
            )
        )
        run(
            tmp_db.insert_outcome(
                {
                    "prediction_id": pid,
                    "actual_return": -0.01,
                    "correct": 1 if i % 10 < 4 else 0,  # 40% accuracy → drift
                    "pnl_after_costs": -0.01,
                    "resolved_at": int(time.time() * 1000),
                }
            )
        )

    monkeypatch.setattr(drift, "_last_trained_days", lambda: 99.0)  # past the rate-limit
    called = {"n": 0}

    async def fake_train_main():
        called["n"] += 1

    monkeypatch.setattr(train, "main", fake_train_main)  # don't really retrain

    res = run(drift.maybe_retrain())
    assert res["retrained"] is True, res
    assert called["n"] == 1  # the trainer was invoked
    assert any("accuracy" in r for r in res["reasons"])


def test_drift_retrain_rate_limited_when_model_fresh(tmp_db, monkeypatch):
    """Even under drift, a too-fresh model must NOT thrash the trainer (rate-limit holds)."""
    import backend.prediction.train as train
    from backend.prediction import drift

    for i in range(250):
        pid = run(
            tmp_db.insert_prediction(
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
                    "target_date": "2024-01-01",
                }
            )
        )
        run(
            tmp_db.insert_outcome(
                {
                    "prediction_id": pid,
                    "actual_return": -0.01,
                    "correct": 1 if i % 10 < 4 else 0,
                    "pnl_after_costs": -0.01,
                    "resolved_at": int(time.time() * 1000),
                }
            )
        )
    monkeypatch.setattr(drift, "_last_trained_days", lambda: 0.5)  # fresher than MIN_AGE
    called = {"n": 0}

    async def fake_train_main():
        called["n"] += 1

    monkeypatch.setattr(train, "main", fake_train_main)
    res = run(drift.maybe_retrain())
    assert res["retrained"] is False and "rate-limited" in res["reason"] and called["n"] == 0


# ── Phase 8: portfolio verifier red-teams the top-k and never raises confidence ──
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


def test_verify_portfolio_ranks_topk_and_never_raises(monkeypatch):
    """GATE-8: the portfolio red-team picks the highest-conviction (edge×meta) names and, even
    when the LLM tries to RAISE every confidence to 99, no final confidence exceeds its model
    ceiling."""
    import backend.prediction.agent as agent

    # An LLM that always tries to inflate confidence to 99 and never vetoes.
    async def greedy_llm(prompt):
        return (
            '{"agree":true,"confidence":99,"rationale":"bull","risks":"r","key_levels":"1",'
            '"contradicts_model":false,"veto":false}'
        )

    monkeypatch.setattr(agent, "_ollama_call", greedy_llm)

    # Conviction = (prob_up-0.5)*meta. Highest: AAA(.18) > CCC(.105) > BBB(.04) > DDD(<0).
    preds = [
        _fake_pred("BBB", 70, 0.60, 0.40),
        _fake_pred("AAA", 55, 0.80, 0.60),
        _fake_pred("CCC", 80, 0.65, 0.70),
        _fake_pred("DDD", 90, 0.40, 0.80),
    ]
    verdicts = run(agent.verify_portfolio(preds, top_k=3, persist=False))

    assert [v["symbol"] for v in verdicts] == ["AAA", "CCC", "BBB"]  # ranked by conviction
    for v in verdicts:
        assert v["final_confidence"] <= v["model_confidence"], v  # ceiling never breached
        assert v["final_confidence"] == v["model_confidence"]  # raise attempt clamped


def test_verify_portfolio_veto_downgrades_and_degrades_gracefully(monkeypatch):
    import backend.prediction.agent as agent

    # First name gets a VETO, the LLM is "down" for the rest (RuntimeError) → graceful, unchanged.
    state = {"calls": 0}

    async def flaky_llm(prompt):
        state["calls"] += 1
        if state["calls"] == 1:
            return (
                '{"agree":false,"confidence":50,"rationale":"guidance cut","risks":"r",'
                '"key_levels":"1","contradicts_model":true,"veto":true}'
            )
        raise RuntimeError("Ollama is not running")

    monkeypatch.setattr(agent, "_ollama_call", flaky_llm)

    preds = [_fake_pred("AAA", 60, 0.80, 0.60), _fake_pred("CCC", 80, 0.65, 0.70)]
    verdicts = run(agent.verify_portfolio(preds, top_k=2, persist=False))
    top = verdicts[0]  # AAA, highest conviction
    assert top["veto"] is True and top["final_act"] is False
    assert top["final_confidence"] <= top["model_confidence"] // 2  # veto heavily downgrades
    assert verdicts[1]["verifier"] == "unavailable"  # graceful per-name
    assert verdicts[1]["final_confidence"] == verdicts[1]["model_confidence"]  # never raised


# ── Phase 8: the daily flywheel cycle runs verify + returns its count ────────────
def test_daily_cycle_runs_verifier(tmp_db, monkeypatch):
    import backend.prediction.agent as agent
    import backend.prediction.serve as serve

    async def fake_resolve(as_of=None):
        return 0

    async def fake_run_predictions(*a, **k):
        return [_fake_pred("AAA", 60, 0.80, 0.60), _fake_pred("BBB", 70, 0.60, 0.40)]

    async def ok_llm(prompt):
        return (
            '{"agree":true,"confidence":50,"rationale":"x","risks":"y","key_levels":"1",'
            '"contradicts_model":false,"veto":false}'
        )

    monkeypatch.setattr(serve, "resolve_due", fake_resolve)
    monkeypatch.setattr(serve, "run_predictions", fake_run_predictions)
    monkeypatch.setattr(agent, "_ollama_call", ok_llm)
    # regime cache invalidation is harmless; let it run.
    res = run(serve.daily_prediction_cycle(verify_top_k=2))
    assert res["logged"] == 2 and res["verified"] == 2


# ── Options IV/skew (pure math via mocked chain + crypto guard + DB round-trip) ───
def test_options_iv_math_and_skew(monkeypatch):
    import sys
    import types
    from datetime import date, timedelta

    import backend.prediction.options as opt

    spot = 100.0
    near = str(date.today() + timedelta(days=30))
    far = str(date.today() + timedelta(days=90))

    def chain_df(strikes, ivs):
        return pd.DataFrame(
            {
                "strike": strikes,
                "impliedVolatility": ivs,
                "bid": [1.0] * len(strikes),
                "openInterest": [10] * len(strikes),
            }
        )

    class FakeTicker:
        def __init__(self, sym):
            self.options = [near, far]

        @property
        def fast_info(self):
            return {"last_price": spot}

        def option_chain(self, exp):
            if exp == near:  # ATM call=0.20 put=0.22 → 0.21
                return types.SimpleNamespace(
                    calls=chain_df([90, 100, 110], [0.30, 0.20, 0.18]),
                    puts=chain_df([90, 100, 110], [0.26, 0.22, 0.20]),
                )
            return types.SimpleNamespace(
                calls=chain_df([90, 100, 110], [0.32, 0.23, 0.20]),
                puts=chain_df([90, 100, 110], [0.28, 0.24, 0.22]),
            )

    monkeypatch.setattr(opt, "_is_optionable", lambda s: True)
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=FakeTicker))
    f = run(opt.compute_iv_features("FAKE"))
    assert f is not None
    assert abs(f["atm_iv"] - 0.21) < 1e-6  # mean(call@100, put@100)
    assert abs(f["skew"] - 0.08) < 1e-6  # put@90 (0.26) − call@110 (0.18)
    assert abs(f["term_slope"] - 0.025) < 1e-6  # far ATM 0.235 − near ATM 0.21
    assert f["n_contracts"] == 6


def test_options_crypto_guard_returns_none():
    import backend.prediction.options as opt

    # 'BTC' resolves to an unrelated equity on yfinance — the guard must short-circuit it (no net).
    assert opt._is_optionable("AAPL") is True
    assert opt._is_optionable("BTC") is False
    assert run(opt.compute_iv_features("BTC")) is None


def test_options_iv_db_roundtrip(tmp_db):
    from backend.prediction.options import symbol_iv

    run(
        tmp_db.insert_options_iv(
            [
                {
                    "symbol": "AAPL",
                    "date": "2026-06-06",
                    "spot": 307.3,
                    "atm_iv": 0.26,
                    "skew": 0.03,
                    "term_slope": 0.01,
                    "n_contracts": 50,
                    "snapshot_at": int(time.time() * 1000),
                }
            ]
        )
    )
    iv = run(symbol_iv("AAPL"))
    assert iv["available"] and iv["atm_iv"] == 0.26 and iv["skew"] == 0.03
    assert run(symbol_iv("ZZZ"))["available"] is False  # no snapshot → neutral


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    raise SystemExit(pytest.main([__file__, "-q"]))
