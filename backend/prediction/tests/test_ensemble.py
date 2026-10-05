"""
FLUX Prediction — Phase-6 regime-conditional ensemble tests (pure / synthetic / network-free)
=============================================================================================
Fast, deterministic unit checks for the mixture-of-experts stack (`ensemble.py`). The reusable
``_check_*`` helpers are also imported by ``scripts/experiments/gate6_regime_ensemble_audit.py`` so the acceptance audit and
the unit suite share one source of truth (same pattern as Phase-4/5).

Covered:
  • RegimeStacker output == manual [ soft posterior-mixture of experts, shrunk toward primary ]
  • scalar predict_proba == batch predict_proba on the same row; outputs bounded to [0,1]
  • hard routing limit: a one-hot regime posterior selects exactly that regime's expert
  • global fallback: regimes with too few rows reuse the global model; zero-posterior degrades to it
  • oof_base2 is purged-OOF aligned (full length, NaN only where a row was never test-folded)
  • GATE-6 report internal consistency (best_base_auc, gate flag) + the MoE earns its place on
    regime-switch truth where the decorrelated base learner gives it something to fuse
  • save / load round-trip is prediction-identical
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.prediction.ensemble import (
    BASE_SIGNALS,
    FEATURES,
    GATE_MARGIN,
    MIN_REGIME_ROWS,
    REGIME_PROBS,
    REGIMES,
    RegimeStacker,
    _fit_experts,
    _make_meta_learner,
    make_base2,
    oof_base2,
    train_regime_stack,
    train_stack,
)


# ── synthetic OOF panel where a regime-conditional blend genuinely helps ──────────
def _synth_oof(n: int = 6000, seed: int = 7) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build OOF-shaped signals + regime posteriors + label. The truth is regime-dependent:
    in 'trend' the primary view drives y, in 'risk_off' the decorrelated base2 view drives y,
    in 'chop' it's a wash. A single global logistic can't capture that switch but a per-regime
    mixture can — so the regime experts have room to beat the global stack here (unlike on the
    real panel, where the same machinery correctly self-disables).
    """
    rng = np.random.default_rng(seed)
    regime = rng.choice(REGIMES, size=n, p=[0.55, 0.30, 0.15])
    s1 = rng.normal(0, 1, n)  # primary latent
    s2 = 0.3 * s1 + rng.normal(0, 1, n)  # base2 latent (partly independent)
    z = np.where(
        regime == "trend", 1.2 * s1, np.where(regime == "risk_off", 1.2 * s2, 0.4 * s1 + 0.4 * s2)
    )
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(int)

    def _prob(latent, noise):
        return np.clip(1 / (1 + np.exp(-(latent + rng.normal(0, noise, n)))), 1e-4, 1 - 1e-4)

    base = {r: np.full(n, 0.1) for r in REGIMES}  # confident soft one-hot on the true regime
    for i, r in enumerate(regime):
        base[r][i] = 0.8
    feats = pd.DataFrame(
        {
            "primary_cal": _prob(s1, 0.5),
            "p2_base": _prob(s2, 0.5),
            "meta_prob": np.clip(rng.uniform(0.45, 0.6, n), 0, 1),
            "mag_oof": rng.normal(0, 0.02, n),
            "p_trend": base["trend"],
            "p_chop": base["chop"],
            "p_risk_off": base["risk_off"],
            "regime": regime,
            "_date": pd.date_range("2008-01-01", periods=n, freq="D"),
            "_y": y,
        }
    )
    return feats, y


# ══════════════════════════════════════════════════════════════════════════════════
#  Reusable checks (shared with scripts/experiments/gate6_regime_ensemble_audit.py)
# ══════════════════════════════════════════════════════════════════════════════════
def _manual_predict(stacker: RegimeStacker, feats: pd.DataFrame) -> np.ndarray:
    """Reference implementation: posterior-weighted mixture of experts, shrunk toward primary."""
    Xs = stacker.scaler.transform(feats[stacker.features].values.astype(float))
    rp = feats[REGIME_PROBS].values.astype(float)
    mix = np.zeros(len(feats))
    wsum = np.zeros(len(feats))
    for j, r in enumerate(REGIMES):
        m = stacker.experts.get(r, stacker.global_model)
        mix += rp[:, j] * m.predict_proba(Xs)[:, 1]
        wsum += rp[:, j]
    mix = np.where(
        wsum > 1e-9,
        mix / np.where(wsum > 1e-9, wsum, 1.0),
        stacker.global_model.predict_proba(Xs)[:, 1],
    )
    b = stacker.primary_blend
    return np.clip((1 - b) * mix + b * feats["primary_cal"].values.astype(float), 0, 1)


def _check_mixture_math(stacker: RegimeStacker, feats: pd.DataFrame) -> str:
    """Batch == manual [mixture shrunk toward primary]; scalar == batch on the same row."""
    manual = _manual_predict(stacker, feats)
    batch = stacker.predict_proba_batch(feats)
    assert np.allclose(batch, manual, atol=1e-9), "batch != manual posterior-mixture+shrink"

    row = feats.iloc[100]
    scalar = stacker.predict_proba(
        {c: float(row[c]) for c in BASE_SIGNALS}, {r: float(row[f"p_{r}"]) for r in REGIMES}
    )
    assert abs(scalar - batch[100]) < 1e-9, f"scalar {scalar} != batch {batch[100]}"
    return f"batch==manual; scalar==batch (blend->primary {stacker.primary_blend:.2f})"


def _check_bounds(stacker: RegimeStacker, feats: pd.DataFrame) -> str:
    p = stacker.predict_proba_batch(feats)
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), "stack proba out of [0,1]"
    return f"all {len(p):,} probabilities finite and in [0,1]"


def _check_hard_routing(feats: pd.DataFrame) -> str:
    """A one-hot regime posterior must collapse the mixture onto exactly that regime's expert.

    Built on a stacker with use_experts forced ON and no shrink, so the routing is observable.
    The regime posterior is the GATE only (not a feature), so the base-signal design matrix is
    unchanged when we flip the posterior — flipping it just selects the expert.
    """
    feats2, y2 = _synth_oof(4000, seed=11)
    from sklearn.preprocessing import StandardScaler

    X = feats2[BASE_SIGNALS].values.astype(float)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    glob = _make_meta_learner().fit(Xs, y2)
    experts = _fit_experts(Xs, y2, feats2["regime"].values, glob, min_rows=MIN_REGIME_ROWS)
    stacker = RegimeStacker(
        experts, glob, scaler, BASE_SIGNALS, primary_blend=0.0, use_experts=True
    )

    for r in REGIMES:
        expert = stacker.experts.get(r, stacker.global_model)
        f2 = feats2.copy()
        for rr in REGIMES:
            f2[f"p_{rr}"] = 1.0 if rr == r else 0.0  # one-hot gate
        want = expert.predict_proba(Xs)[:, 1]
        got = stacker.predict_proba_batch(f2)
        assert np.allclose(got, want, atol=1e-9), f"one-hot {r} did not select that expert"
    return "one-hot posterior routes to the matching expert exactly"


def _check_fallback() -> str:
    """Experts with < min_rows fall back to global; zero total posterior degrades to global."""
    feats, y = _synth_oof(2000, seed=3)
    from sklearn.preprocessing import StandardScaler

    X = feats[BASE_SIGNALS].values.astype(float)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    glob = _make_meta_learner().fit(Xs, y)
    experts = _fit_experts(Xs, y, feats["regime"].values, glob, min_rows=10**9)
    assert all(experts[r] is glob for r in REGIMES), "huge min_rows should force global fallback"

    stacker = RegimeStacker(
        experts, glob, scaler, BASE_SIGNALS, primary_blend=0.0, use_experts=False
    )
    f2 = feats.copy()  # zero total posterior -> global fallback
    for rr in REGIMES:
        f2[f"p_{rr}"] = 0.0
    got = stacker.predict_proba_batch(f2)
    want = glob.predict_proba(Xs)[:, 1]
    assert np.allclose(got, want, atol=1e-9), "zero-posterior did not degrade to the global expert"
    return "low-count experts + zero posterior both fall back to global"


def _check_oof_base2_alignment() -> str:
    """oof_base2 returns a full-length array, NaN only on rows never in an OOF test fold."""
    rng = np.random.default_rng(1)
    n, k = 1200, 5
    feat_cols = [f"f{i}" for i in range(k)]
    dates = pd.date_range("2010-01-01", periods=n, freq="D")
    Xd = pd.DataFrame(rng.normal(size=(n, k)), columns=feat_cols)
    Xd.index = dates
    y = (rng.uniform(size=n) < 0.5).astype(int)
    w = np.ones(n)
    t1 = pd.Series(dates + pd.Timedelta(days=5), index=Xd.index)
    oof = oof_base2(Xd, y, w, t1, feat_cols)
    assert oof.shape == (n,), f"oof_base2 wrong shape {oof.shape}"
    tested = ~np.isnan(oof)
    assert tested.sum() > n * 0.5, "too few rows received an OOF base2 prediction"
    assert np.isfinite(oof[tested]).all() and (oof[tested] >= 0).all() and (oof[tested] <= 1).all()
    return f"{tested.sum():,}/{n} rows OOF-scored, all in [0,1]"


def _check_gate_logic(rep: dict) -> str:
    """The report's gate flag and best-base bookkeeping must be internally consistent."""
    assert abs(rep["best_base_auc"] - max(rep["primary_auc"], rep["p2_auc"])) < 1e-12, (
        "best_base_auc != max(primary, p2)"
    )
    expected = rep["stack_auc"] > rep["best_base_auc"] + GATE_MARGIN
    assert rep["gate6_pass"] == expected, "gate6_pass inconsistent with stack vs best-base + margin"
    assert abs(rep["lift_vs_best_base"] - (rep["stack_auc"] - rep["best_base_auc"])) < 1e-12
    return (
        f"stack {rep['stack_auc']:.4f} vs best base {rep['best_base_auc']:.4f} "
        f"({rep['lift_vs_best_base']:+.4f}) -> gate6_pass={rep['gate6_pass']}"
    )


def _check_moe_earns_place(rep: dict) -> str:
    """On regime-switch truth the per-regime experts beat the global stack and the gate should pass."""
    assert rep["use_experts"], "regime experts should self-enable on regime-switch truth"
    assert rep["moe_stack_auc"] > rep["global_stack_auc"], (
        f"MoE did not beat global on switch truth ({rep['moe_stack_auc']} <= {rep['global_stack_auc']})"
    )
    assert rep["gate6_pass"] and rep["stack_auc"] > rep["best_base_auc"], (
        "stack should clear GATE-6"
    )
    return (
        f"MoE {rep['moe_stack_auc']:.4f} > global {rep['global_stack_auc']:.4f}; "
        f"stack {rep['stack_auc']:.4f} > best base {rep['best_base_auc']:.4f}"
    )


def _check_save_load(stacker: RegimeStacker, feats: pd.DataFrame, tmp_path) -> str:
    p = tmp_path / "regime_stack.pkl"
    stacker.save(p)
    loaded = RegimeStacker.load(p)
    assert np.allclose(
        loaded.predict_proba_batch(feats), stacker.predict_proba_batch(feats), atol=1e-12
    )
    assert loaded.report.get("gate6_pass") == stacker.report.get("gate6_pass")
    assert (
        loaded.use_experts == stacker.use_experts and loaded.primary_blend == stacker.primary_blend
    )
    return "save/load round-trip prediction-identical (experts + blend preserved)"


# ══════════════════════════════════════════════════════════════════════════════════
#  pytest wrappers
# ══════════════════════════════════════════════════════════════════════════════════
def test_regime_stack_trains_and_mixture_math():
    feats, y = _synth_oof()
    stacker, rep = train_regime_stack(feats, y)
    _check_mixture_math(stacker, feats)
    _check_bounds(stacker, feats)
    _check_gate_logic(rep)


def test_one_hot_posterior_routes_to_expert():
    assert _check_hard_routing(_synth_oof()[0])


def test_regime_experts_earn_place_on_switch_truth():
    feats, y = _synth_oof()
    _stacker, rep = train_regime_stack(feats, y)
    _check_moe_earns_place(rep)


def test_expert_fallback_and_zero_posterior():
    assert _check_fallback()


def test_oof_base2_is_purged_aligned():
    assert _check_oof_base2_alignment()


def test_regime_stack_save_load_roundtrip(tmp_path):
    feats, y = _synth_oof()
    stacker, _rep = train_regime_stack(feats, y)
    _check_save_load(stacker, feats, tmp_path)


def test_base2_pipeline_supports_sample_weight():
    """make_base2() must accept the pipeline-routed sample_weight kwarg used in training."""
    feats, y = _synth_oof(800, seed=5)
    cols = ["primary_cal", "p2_base", "meta_prob", "mag_oof"]
    m = make_base2()
    m.fit(feats[cols], y, logisticregression__sample_weight=np.ones(len(y)))
    p = m.predict_proba(feats[cols])[:, 1]
    assert p.shape == (len(y),) and (p >= 0).all() and (p <= 1).all()


def test_shrink_anchors_toward_primary():
    """primary_blend=1.0 must reproduce the calibrated primary exactly (pure shrink anchor)."""
    feats, y = _synth_oof()
    stacker, _rep = train_regime_stack(feats, y, primary_blend=1.0)
    got = stacker.predict_proba_batch(feats)
    assert np.allclose(got, feats["primary_cal"].values, atol=1e-9), "blend=1.0 != primary"


def test_legacy_flat_stacker_still_works():
    """Backward-compat: the original flat Stacker/train_stack/FEATURES path is untouched."""
    rng = np.random.default_rng(3)
    n = 2000
    primary = rng.uniform(0.3, 0.7, n)
    y = (rng.uniform(size=n) < primary).astype(int)
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
