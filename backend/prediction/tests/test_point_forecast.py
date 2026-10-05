"""
Point-forecast tests: does the single predicted price beat "no change", and are the
shipped model's reported numbers internally consistent?

There is no reinforcement-learning component in this project, so there is nothing RL to
score here; the ML point forecast (xgb_return.json) and its conformal band are what ship.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from backend.prediction.point_eval import point_forecast_has_skill, point_forecast_report

META = Path(__file__).resolve().parents[1] / "models" / "model_meta.json"


def test_perfect_forecast():
    a = np.array([0.02, -0.01, 0.03, -0.04])
    r = point_forecast_report(a, a)
    assert r["mae"] == 0 and r["skill"] == 1 and r["within_tol"] == 1 and r["direction_hit"] == 1


def test_no_change_forecast_has_zero_skill():
    a = np.random.default_rng(0).normal(0, 0.03, 5000)
    r = point_forecast_report(np.zeros_like(a), a)
    assert r["skill"] == pytest.approx(0.0)
    assert r["within_tol"] == r["within_tol_naive"]


def test_noise_forecast_has_negative_skill():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.03, 20_000)
    r = point_forecast_report(rng.normal(0, 0.03, a.size), a)  # confident but unrelated
    assert r["skill"] < -0.2
    assert abs(r["direction_hit"] - 0.5) < 0.02


def test_partly_informative_forecast_has_positive_skill():
    rng = np.random.default_rng(2)
    signal = rng.normal(0, 0.02, 20_000)
    a = signal + rng.normal(0, 0.02, signal.size)
    r = point_forecast_report(signal, a)
    assert r["skill"] > 0.1 and r["direction_hit"] > 0.6


def test_band_coverage_and_nan_rows():
    a = np.array([0.01, 0.05, -0.02, np.nan])
    r = point_forecast_report(np.zeros(4), a, lo=np.full(4, -0.03), hi=np.full(4, 0.03))
    assert r["n"] == 3 and r["band_coverage"] == pytest.approx(2 / 3)


def test_skill_flag():
    assert point_forecast_has_skill({"mae": 0.050, "mae_predict_zero": 0.055})
    assert not point_forecast_has_skill({"mae": 0.0541, "mae_predict_zero": 0.0535})
    assert not point_forecast_has_skill({})


# ── the shipped model ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def conformal():
    return json.loads(META.read_text())["conformal_report"]


def test_shipped_band_coverage_matches_target(conformal):
    """The range shown on the page must hold its stated coverage out of fold (±2 points)."""
    for alpha, cov in conformal["coverage"].items():
        assert abs(cov - (1 - float(alpha))) <= 0.02, (alpha, cov)


def test_shipped_point_forecast_skill_is_recorded(conformal, capsys):
    """Report, don't hide: the single-price skill vs 'no change' of the shipped regressor."""
    skill = 1 - conformal["mae"] / conformal["mae_predict_zero"]
    with capsys.disabled():
        print(
            f"\n  shipped point forecast: MAE {conformal['mae']:.4f} vs no-change "
            f"{conformal['mae_predict_zero']:.4f} -> skill {skill:+.3f} "
            f"({'beats' if skill > 0 else 'does NOT beat'} no-change; n={conformal['n_calib']:,})"
        )
    assert point_forecast_has_skill(conformal) == (skill > 0)
    assert -0.25 < skill < 0.25  # anything outside this is a bug, not a model
