"""Calibration ECE must be scored out of sample."""

import numpy as np

from backend.prediction.train import _ece, calibrate_oos, ece_calibrated_oos, fit_calibrator


def _noise(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.3, 0.7, n)  # model output with no real skill
    y = rng.integers(0, 2, n)
    fold = np.repeat(np.arange(5), n // 5)
    return p, y, fold


def test_insample_isotonic_ece_is_meaningless():
    p, y, _ = _noise()
    assert _ece(y, fit_calibrator(p, y).predict(p)) < 1e-6  # "perfect", like the old 4e-16


def test_oos_ece_is_a_real_number():
    p, y, fold = _noise()
    ece, n = ece_calibrated_oos(p, y, fold)
    assert n == 16_000  # folds 1-4 scored, fold 0 only trains
    assert ece > 1e-3


def test_well_calibrated_model_scores_low():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 50_000)
    y = (rng.uniform(0, 1, p.size) < p).astype(int)
    ece, _ = ece_calibrated_oos(p, y, np.repeat(np.arange(5), 10_000))
    assert ece < 0.02


def test_fold_biases_do_not_cancel():
    """Fold 1 over-predicts, fold 2 under-predicts: pooled they cancel, per fold they don't."""
    n = 10_000
    p = np.full(3 * n, 0.55)
    y = np.concatenate(
        [
            np.r_[np.ones(5_500), np.zeros(4_500)],  # fold 0: rate 0.55 (fit)
            np.r_[np.ones(4_500), np.zeros(5_500)],  # fold 1: rate 0.45
            np.r_[np.ones(6_500), np.zeros(3_500)],
        ]
    )  # fold 2: rate 0.65
    fold = np.repeat([0, 1, 2], n)
    ece, n_scored = ece_calibrated_oos(p, y, fold)
    assert n_scored == 2 * n
    assert ece > 0.05  # pooled would have been ~0


def test_calibrate_oos_uses_only_earlier_folds():
    """Each fold's calibrated probs come from a calibrator fit on earlier folds; fold 0 is NaN."""
    p, y, fold = _noise()
    p = p.copy()
    p[:10] = np.nan  # rows never in a test fold stay NaN
    out = calibrate_oos(p, y, fold)
    assert np.isnan(out[fold == 0]).all()
    later = fold > 0
    assert not np.isnan(out[later]).any()
    k1 = fold == 1  # fold 1 = calibrator fit on fold 0 alone
    ok0 = (fold == 0) & ~np.isnan(p)
    expect = fit_calibrator(p[ok0], y[ok0]).predict(p[k1])
    assert np.allclose(out[k1], expect)
