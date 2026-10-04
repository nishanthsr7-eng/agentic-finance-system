"""
FLUX Prediction — Conformal Prediction Bands (guaranteed coverage, Layer 2b)
============================================================================
A point forecast ("we expect +1.2%") is almost useless without a band, and a band is only
honest if it covers at its STATED rate out-of-sample. Conformal prediction gives exactly that:
ask for an 80% interval and ~80% of realised returns fall inside it — distribution-free, with
a finite-sample guarantee, under exchangeability.

How this fits the rest of the pipeline (López de Prado-style, leak-free):
  1. An XGBoost regressor predicts the h-day return point estimate  ŷ(x).
  2. We collect OUT-OF-FOLD residuals e_i = y_i − ŷ(x_i) using the SAME PurgedWalkForwardSplit
     the classifier uses — so no overlapping-label leakage inflates the band's apparent tightness.
  3. GARCH (garch.py) gives a per-event volatility scale s_i. We normalise: u_i = |e_i| / s_i.
     GARCH SHAPES the band (wider in turbulent markets); conformal GUARANTEES its coverage.
  4. The conformal quantile q_α of {u_i} (with the finite-sample +1 correction) is the band
     half-width in σ-units. At serving: interval = ŷ ± q_α · s_today.

Why not MAPIE? MAPIE's built-in CV is not purged, so on overlapping triple-barrier labels it
would leak and under-cover live. Reusing our own purged splitter is both correct and lighter.

Public API:
    conformal_quantile(scores, alpha)         -> q_α with finite-sample correction
    empirical_coverage(scores, q)             -> realised coverage of a quantile
    ConformalBands.fit(residuals, scales, …)  -> calibrated bands (+ self-reported coverage)
    bands.interval(point, scale, alpha)       -> (low, high)
    bands.save(path) / ConformalBands.load(p)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    The conformal quantile of nonconformity `scores` at miscoverage `alpha` (so a 1-alpha band).
    Uses the finite-sample correction level = ceil((n+1)(1-alpha)) / n, which is what makes the
    coverage guarantee hold for finite n (Vovk; Angelopoulos & Bates). If the corrected level
    exceeds 1 (too few samples for the requested confidence) the band is unbounded → +inf.
    """
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    n = s.size
    if n == 0:
        return float("inf")
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    if level > 1.0:
        return float("inf")
    return float(np.quantile(s, level, method="higher"))


def empirical_coverage(scores: np.ndarray, q: float) -> float:
    """Fraction of scores within the quantile — i.e. the realised coverage of band half-width q."""
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan")
    return float((s <= q).mean())


@dataclass
class ConformalBands:
    """
    Calibrated conformal half-widths in σ-units, keyed by miscoverage alpha. `interval()` turns
    a point estimate + a (GARCH) scale into a concrete [low, high] return band.
    """

    horizon: int
    q: dict[float, float] = field(default_factory=dict)  # alpha -> half-width (σ units)
    coverage: dict[float, float] = field(default_factory=dict)  # alpha -> OOF realised coverage
    n_calib: int = 0
    scale_floor: float = 1e-6  # avoid div-by-zero on flat σ

    @classmethod
    def fit(cls, residuals, scales, horizon: int, alphas=(0.2, 0.1)) -> ConformalBands:
        """
        Calibrate from out-of-fold residuals and matching per-event GARCH scales.
            residuals : y_true - y_pred  (h-day return units)
            scales    : σ per event (same length/order); band half-width = q · σ
        Self-reports realised OOF coverage per alpha so train.py can assert it ≈ 1-alpha.
        """
        r = np.asarray(residuals, dtype=float)
        s = np.asarray(scales, dtype=float)
        m = np.isfinite(r) & np.isfinite(s) & (s > 0)
        r, s = r[m], s[m]
        u = np.abs(r) / np.maximum(s, 1e-12)  # normalised nonconformity
        self = cls(horizon=int(horizon), n_calib=int(u.size))
        for a in alphas:
            qa = conformal_quantile(u, a)
            self.q[float(a)] = qa
            self.coverage[float(a)] = empirical_coverage(u, qa)
        return self

    def interval(self, point: float, scale: float, alpha: float = 0.2) -> tuple[float, float]:
        """Return (low, high) for a point estimate and a per-event σ scale, at miscoverage alpha."""
        q = self.q.get(float(alpha))
        if q is None:  # nearest calibrated alpha
            q = self.q[min(self.q, key=lambda a: abs(a - alpha))]
        half = q * max(float(scale), self.scale_floor)
        return float(point - half), float(point + half)

    # ── persistence (plain dict via joblib, like the other artifacts) ─────────────
    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump(
            {
                "horizon": self.horizon,
                "q": self.q,
                "coverage": self.coverage,
                "n_calib": self.n_calib,
                "scale_floor": self.scale_floor,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> ConformalBands:
        import joblib

        d = joblib.load(path)
        return cls(
            horizon=d["horizon"],
            q=d["q"],
            coverage=d["coverage"],
            n_calib=d.get("n_calib", 0),
            scale_floor=d.get("scale_floor", 1e-6),
        )


if __name__ == "__main__":
    # Sanity: with correctly-specified scale, an 80% band should cover ~80% on held-out draws.
    rng = np.random.default_rng(1)
    n = 5000
    scale = rng.uniform(0.01, 0.05, n)  # heteroskedastic "GARCH" σ
    resid = rng.normal(0, scale)  # residuals genuinely scale with σ
    cut = n // 2
    bands = ConformalBands.fit(resid[:cut], scale[:cut], horizon=5, alphas=(0.2, 0.1))
    for a in (0.2, 0.1):
        q = bands.q[a]
        cov = empirical_coverage(np.abs(resid[cut:]) / scale[cut:], q)
        print(
            f"  target {1 - a:.0%} band: half-width {q:.2f}sigma  "
            f"calib-cov {bands.coverage[a]:.1%}  holdout-cov {cov:.1%}"
        )
