"""
FLUX Prediction — Regime-Conditional Stacking Ensemble (Layer 2 fusion, Phase 6)
================================================================================
Until now the layers were fused by hand: `confidence = meta_prob`, tilted by a fixed sentiment
factor and scaled by a fixed regime multiplier. Hand rules can't find the *optimal* blend. A
stacking meta-learner does: it trains a small model on the OUT-OF-FOLD predictions of the base
layers and learns how much to trust each, conditional on the others (López de Prado; Wolpert
stacking) — IF it is given decorrelated, informative inputs. The flat stack failed earlier
because its inputs were all near-duplicate views of one thin XGBoost signal. Phase 6 finally
earns the stack its place by fixing that.

WHAT MOVED THE NEEDLE (verified on the leak-free OOF, scripts/experiments/gate6_regime_ensemble_diagnostic.py):
  • A DECORRELATED 2nd base learner — an ElasticNet logistic (`p2_base`) over the full feature
    set. corr(primary_cal, p2_base) ≈ 0.42, NOT a duplicate: trees and a regularised linear model
    make different errors, and that diversity is what lets a stack add AUC. With it, a parsimonious
    global logistic over the base signals beats the best single base learner on the held-out split
    (GATE-6 PASS), shrunk 50% toward the calibrated primary for distribution-shift robustness.

HOW THE HMM REGIME IS USED (all three asked-for ways were implemented AND evaluated honestly):
  (a) as STACK FEATURES — REJECTED. Adding the regime posteriors to the linear stack design matrix
      HURT the held-out AUC (≈ −0.025): the regime mix shifts over time (early=trend-heavy,
      late=risk_off-heavy), so a coefficient learned early flips sign late — the exact
      non-stationary common-mode failure the Phase-1 FRED ablation found. So regime is kept OUT of
      the stack feature vector. The evidence lives in report["regime_feature_stack_auc"].
  (b) as PER-REGIME EXPERT WEIGHTS — a logistic per trend/chop/risk_off, softly mixed by the live
      posterior (a true mixture-of-experts, degrades gracefully when the regime is uncertain). The
      machinery is built and self-selecting: `use_experts` is turned on at fit time ONLY if the
      regime experts beat the single global stack on the held-out split. On the production panel
      they DON'T (expert starvation + the same early→late regime shift), so it collapses to the
      global stack — but a future, richer panel can flip it on with no code change.
  (c) as the SIZE GATE — unchanged and load-bearing (REGIME_SCALE in predict.py); this is regime's
      proven home (cuts max drawdown −57%→−45%), consistent with (a)'s rejection.

Base signals fed to the stacker (all leak-free OOF):
  • primary_cal  — calibrated P(up) from the XGBoost direction model (Layer 2a)
  • p2_base      — P(up) from the decorrelated ElasticNet logistic base learner (Phase 6)
  • meta_prob    — P(primary call correct) from the meta model (conviction)
  • mag_oof      — OOF point return from the magnitude regressor (Layer 2b), in σ-ish units
The HMM posteriors [p_trend, p_chop, p_risk_off] ride along in the OOF frame but are used ONLY as
the mixture GATE (b), never as design-matrix features (a). Sentiment (2c) stays a real-time tilt at
serving (no usable history). Kelly sizing + the regime size gate are applied on top.

Honest evaluation: every model is fit on the EARLY 70% of OOF rows and scored on the LATER 30%
(time-ordered), so the reported lift is itself out-of-sample. GATE-6: the chosen stack's held-out
AUC must exceed the BEST SINGLE base learner's (max of primary_cal, p2_base). The self-gate in
predict.py enforces this against the persisted report before the stack is ever used at serving.

Public API:
    RegimeStacker.load(path).predict_proba(signals, regime_probs) -> P(up)
    await fit_and_save()   # offline: trains the stack on OOF signals, persists artifacts

Backward-compat: the original flat `Stacker` / `train_stack` / `FEATURES` are retained below and
still used by the legacy unit test; the production path now uses `RegimeStacker`.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("flux.prediction.ensemble")

MODELS_DIR = Path(__file__).parent / "models"

# ── Legacy flat stack (kept for backward compatibility / unit test) ───────────────
FEATURES = ["primary_cal", "meta_prob", "mag_oof", "regime_scale"]
_REGIME_SCALE = {"trend": 1.0, "chop": 0.6, "risk_off": 0.25}  # mirror regime.REGIME_SCALE

# ── Phase-6 regime-conditional stack ──────────────────────────────────────────────
REGIMES = ("trend", "chop", "risk_off")  # order matches REGIME_PROBS
BASE_SIGNALS = ["primary_cal", "p2_base", "meta_prob", "mag_oof"]  # the stack DESIGN MATRIX
REGIME_PROBS = ["p_trend", "p_chop", "p_risk_off"]  # mixture GATE only (NOT features)
OOF_COLS = BASE_SIGNALS + REGIME_PROBS  # columns the OOF frame must carry
MIN_REGIME_ROWS = 500  # an expert with fewer training rows falls back to the global stack
GATE_MARGIN = 1e-4  # AUC lift above best base learner required to "earn" the stack
PRIMARY_BLEND = 0.5  # shrink the stack this far toward the calibrated primary (robustness)


# ══════════════════════════════════════════════════════════════════════════════════
#  Legacy flat Stacker (unchanged API — retained for the existing unit test)
# ══════════════════════════════════════════════════════════════════════════════════
class Stacker:
    """Thin wrapper around a fitted logistic meta-learner + its scaler and feature order."""

    def __init__(self, model, scaler, features=FEATURES, report: dict | None = None):
        self.model = model
        self.scaler = scaler
        self.features = list(features)
        self.report = report or {}

    def predict_proba(self, primary_cal: float, meta_prob: float, mag: float, regime: str) -> float:
        row = np.array([[primary_cal, meta_prob, mag, _REGIME_SCALE.get(regime, 1.0)]], dtype=float)
        row = self.scaler.transform(row)
        return float(self.model.predict_proba(row)[:, 1][0])

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "features": self.features,
                "report": self.report,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> Stacker:
        # NOTE: only ever load artifacts this process trained (local, trusted). joblib uses
        # pickle, so loading an untrusted file would be an RCE risk — never load uploads here.
        import joblib

        d = joblib.load(path)
        return cls(d["model"], d["scaler"], d.get("features", FEATURES), d.get("report", {}))


def _ece(y, p, bins=10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            e += abs(p[m].mean() - y[m].mean()) * m.sum() / len(p)
    return float(e)


def train_stack(feats: pd.DataFrame, y: np.ndarray) -> tuple[Stacker, dict]:
    """
    LEGACY flat stack. Fit the logistic meta-learner on the EARLY 70% of OOF rows, evaluate on
    the LATER 30%. Returns (Stacker, report). Kept for the original unit test; production uses
    `train_regime_stack`.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    X = feats[FEATURES].values.astype(float)
    n = len(X)
    cut = int(n * 0.7)
    scaler = StandardScaler().fit(X[:cut])
    Xs = scaler.transform(X)
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(Xs[:cut], y[:cut])

    p_stack = clf.predict_proba(Xs[cut:])[:, 1]
    yte = y[cut:]
    p_primary = feats["primary_cal"].values[cut:]
    report = {
        "n_eval": int(len(yte)),
        "stack_auc": float(roc_auc_score(yte, p_stack)),
        "primary_auc": float(roc_auc_score(yte, p_primary)),
        "stack_acc": float(((p_stack >= 0.5).astype(int) == yte).mean()),
        "primary_acc": float(((p_primary >= 0.5).astype(int) == yte).mean()),
        "stack_ece": _ece(yte, p_stack),
        "primary_ece": _ece(yte, p_primary),
        "coef": dict(zip(FEATURES, clf.coef_[0].round(4).tolist())),
    }
    scaler_full = StandardScaler().fit(X)
    clf_full = LogisticRegression(C=1.0, max_iter=1000).fit(scaler_full.transform(X), y)
    return Stacker(clf_full, scaler_full, FEATURES, report), report


# ══════════════════════════════════════════════════════════════════════════════════
#  Phase-6 model factories
# ══════════════════════════════════════════════════════════════════════════════════
def _make_meta_learner():
    """The stack's meta-learner: a small L2 logistic over the (few, dense) base signals."""
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=1.0, max_iter=1000)


def make_base2():
    """
    The DIVERSE 2nd base learner (Phase 6): an ElasticNet logistic over the full feature set.
    Standardised inside a pipeline (saga needs scaling); L1+L2 keeps it sparse/robust on the wide,
    collinear feature panel. Its job is to be DECORRELATED from the XGBoost primary (corr ≈ 0.42 on
    the OOF panel), not to win on its own — that decorrelation is what lets the stack add AUC.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=0.5,
            max_iter=2000,
            tol=1e-3,
            n_jobs=-1,
            random_state=42,
        ),
    )


def oof_base2(X, y, w, t1, feat_cols) -> np.ndarray:
    """
    Leak-free OOF predictions from the 2nd base learner via the SAME purged splitter as the
    primary, so its `p2_base` column lines up with `primary_cal` row-for-row (no peek).
    Returns a full-length array aligned to X (NaN where a row was never OOF-tested).
    """
    from backend.prediction.cv import PurgedWalkForwardSplit
    from backend.prediction.train import EMBARGO, N_SPLITS

    cv = PurgedWalkForwardSplit(n_splits=N_SPLITS, embargo=EMBARGO)
    oof = np.full(len(y), np.nan)
    Xf = X[feat_cols]
    for tr, te in cv.split(X, t1):
        m = make_base2()
        m.fit(Xf.iloc[tr], y[tr], logisticregression__sample_weight=w[tr])
        oof[te] = m.predict_proba(Xf.iloc[te])[:, 1]
    return oof


# ══════════════════════════════════════════════════════════════════════════════════
#  Phase-6 RegimeStacker (mixture-of-experts, gated by the live HMM posterior)
# ══════════════════════════════════════════════════════════════════════════════════
class RegimeStacker:
    """
    A stack over the decorrelated base signals, shrunk toward the calibrated primary, optionally
    regime-conditional. The design matrix is BASE_SIGNALS only — the regime posteriors are the
    mixture GATE, never features (usage (a) was tested and rejected). When `use_experts` is False
    the per-regime experts all equal the global model, so the mixture collapses to one global stack
    regardless of the posterior (the production case). When True, one logistic expert per regime is
    softly mixed by the posterior — p_stack = Σ_r P(r)·expert_r(x) / Σ_r P(r) — with the global
    model as the fallback for any regime that lacked rows or has zero posterior weight.

    Final output = (1−primary_blend)·p_stack + primary_blend·primary_cal, clipped to [0,1]. The
    shrink toward the isotonic-calibrated primary is a deliberate variance reducer: it keeps the
    blend robust to the distribution shift that wrecked the un-anchored variants out-of-sample.
    """

    def __init__(
        self,
        experts: dict,
        global_model,
        scaler,
        features=BASE_SIGNALS,
        primary_blend: float = PRIMARY_BLEND,
        use_experts: bool = False,
        report: dict | None = None,
    ):
        self.experts = experts  # {regime: fitted logistic}
        self.global_model = global_model
        self.scaler = scaler
        self.features = list(features)  # base-signal design matrix
        self.primary_blend = float(
            primary_blend
        )  # weight on the calibrated primary (shrink anchor)
        self.use_experts = bool(use_experts)
        self.report = report or {}

    # ── raw soft-mixture over a scaled design matrix (collapses to global if use_experts False) ──
    def _stack_raw(self, Xs: np.ndarray, regime_probs: np.ndarray) -> np.ndarray:
        preds = np.zeros(len(Xs))
        wsum = np.zeros(len(Xs))
        for j, r in enumerate(REGIMES):
            m = self.experts.get(r, self.global_model)
            pr = m.predict_proba(Xs)[:, 1]
            wj = np.clip(regime_probs[:, j], 0.0, None)
            preds += wj * pr
            wsum += wj
        return np.where(
            wsum > 1e-9,
            preds / np.where(wsum > 1e-9, wsum, 1.0),
            self.global_model.predict_proba(Xs)[:, 1],
        )

    def _blend(self, stack_p: np.ndarray, primary_cal: np.ndarray) -> np.ndarray:
        b = self.primary_blend
        return np.clip((1.0 - b) * stack_p + b * primary_cal, 0.0, 1.0)

    def predict_proba(self, signals: dict, regime_probs: dict) -> float:
        """
        signals      : {primary_cal, p2_base, meta_prob, mag_oof}
        regime_probs : {trend, chop, risk_off}  (HMM posterior; need not sum to 1, it's renormed)
        """
        row = np.array([[signals[c] for c in self.features]], dtype=float)
        Xs = self.scaler.transform(row)
        rp = np.array([[regime_probs.get(r, 0.0) for r in REGIMES]], dtype=float)
        sp = self._stack_raw(Xs, rp)
        return float(self._blend(sp, np.array([signals["primary_cal"]]))[0])

    def predict_proba_batch(self, feats: pd.DataFrame) -> np.ndarray:
        """Vectorised scoring over a frame carrying the base signals + regime posteriors."""
        Xs = self.scaler.transform(feats[self.features].values.astype(float))
        rp = feats[REGIME_PROBS].values.astype(float)
        sp = self._stack_raw(Xs, rp)
        return self._blend(sp, feats["primary_cal"].values.astype(float))

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump(
            {
                "experts": self.experts,
                "global_model": self.global_model,
                "scaler": self.scaler,
                "features": self.features,
                "primary_blend": self.primary_blend,
                "use_experts": self.use_experts,
                "report": self.report,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> RegimeStacker:
        # Local, trusted artifacts only (joblib/pickle == RCE if untrusted). Never load uploads.
        import joblib

        d = joblib.load(path)
        return cls(
            d["experts"],
            d["global_model"],
            d["scaler"],
            d.get("features", BASE_SIGNALS),
            d.get("primary_blend", PRIMARY_BLEND),
            d.get("use_experts", False),
            d.get("report", {}),
        )


def _fit_experts(
    Xs: np.ndarray,
    y: np.ndarray,
    regime_label: np.ndarray,
    global_model,
    min_rows: int = MIN_REGIME_ROWS,
) -> dict:
    """Fit one logistic per regime on its rows; fall back to the global model if too few / one-class."""
    experts = {}
    for r in REGIMES:
        m = regime_label == r
        if m.sum() >= min_rows and len(np.unique(y[m])) > 1:
            experts[r] = _make_meta_learner().fit(Xs[m], y[m])
        else:
            experts[r] = global_model
    return experts


def train_regime_stack(
    feats: pd.DataFrame,
    y: np.ndarray,
    min_rows: int = MIN_REGIME_ROWS,
    primary_blend: float = PRIMARY_BLEND,
) -> tuple[RegimeStacker, dict]:
    """
    Fit the stack on the EARLY 70% of OOF rows, evaluate on the LATER 30%, and DATA-DRIVE the two
    regime choices honestly:
      • use_experts — per-regime mixture is kept ONLY if it beats the single global stack OOS.
      • regime-as-feature is measured (report["regime_feature_stack_auc"]) and never shipped.
    GATE-6 compares the chosen, shrunk stack's held-out AUC to the best single base learner.
    Returns (RegimeStacker, report). The persisted model is refit on ALL rows.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    X = feats[BASE_SIGNALS].values.astype(float)
    rp = feats[REGIME_PROBS].values.astype(float)
    rlabel = feats["regime"].astype(str).values
    prim = feats["primary_cal"].values.astype(float)
    p2 = feats["p2_base"].values.astype(float)
    n = len(X)
    cut = int(n * 0.7)
    yte = y[cut:]

    def _auc(p):
        return float(roc_auc_score(yte, p)) if len(np.unique(yte)) > 1 else float("nan")

    # ── honest time-split evaluation on base signals ─────────────────────────────
    scaler = StandardScaler().fit(X[:cut])
    Xs = scaler.transform(X)
    glob_eval = _make_meta_learner().fit(Xs[:cut], y[:cut])
    experts_eval = _fit_experts(Xs[:cut], y[:cut], rlabel[:cut], glob_eval, min_rows)
    moe_eval = RegimeStacker(
        experts_eval, glob_eval, scaler, BASE_SIGNALS, primary_blend=0.0, use_experts=True
    )

    p_global = glob_eval.predict_proba(Xs[cut:])[:, 1]
    p_moe = moe_eval._stack_raw(Xs[cut:], rp[cut:])
    global_auc, moe_auc = _auc(p_global), _auc(p_moe)
    use_experts = bool(moe_auc > global_auc + GATE_MARGIN)
    chosen_raw = p_moe if use_experts else p_global

    # ablation: regime posteriors AS FEATURES (usage (a)) — measured to justify rejecting it.
    Xr = feats[BASE_SIGNALS + REGIME_PROBS].values.astype(float)
    sc_r = StandardScaler().fit(Xr[:cut])
    glob_r = _make_meta_learner().fit(sc_r.transform(Xr[:cut]), y[:cut])
    regime_feature_auc = _auc(glob_r.predict_proba(sc_r.transform(Xr[cut:]))[:, 1])

    # shrink the chosen stack toward the calibrated primary (robustness anchor)
    p_final = np.clip((1 - primary_blend) * chosen_raw + primary_blend * prim[cut:], 0, 1)
    stack_auc = _auc(p_final)
    stack_unshrunk_auc = _auc(chosen_raw)

    base_aucs = {"primary_cal": _auc(prim[cut:]), "p2_base": _auc(p2[cut:])}
    best_base_name = max(base_aucs, key=lambda k: base_aucs[k])
    best_base_auc = base_aucs[best_base_name]

    # per-regime held-out AUC of the shipped stack vs primary (diagnostic)
    per_regime = {}
    rl_te = rlabel[cut:]
    for r in REGIMES:
        mr = rl_te == r
        if mr.sum() >= 50 and len(np.unique(yte[mr])) > 1:
            per_regime[r] = {
                "n": int(mr.sum()),
                "stack_auc": float(roc_auc_score(yte[mr], p_final[mr])),
                "primary_auc": float(roc_auc_score(yte[mr], prim[cut:][mr])),
            }
        else:
            per_regime[r] = {"n": int(mr.sum()), "stack_auc": None, "primary_auc": None}

    report = {
        "n_eval": int(len(yte)),
        "stack_auc": stack_auc,
        "stack_unshrunk_auc": stack_unshrunk_auc,
        "global_stack_auc": global_auc,
        "moe_stack_auc": moe_auc,
        "regime_feature_stack_auc": regime_feature_auc,
        "use_experts": use_experts,
        "primary_blend": primary_blend,
        "primary_auc": base_aucs["primary_cal"],
        "p2_auc": base_aucs["p2_base"],
        "p2_corr_primary": float(np.corrcoef(prim, p2)[0, 1]),
        "best_base_name": best_base_name,
        "best_base_auc": best_base_auc,
        "lift_vs_best_base": stack_auc - best_base_auc,
        "gate6_pass": bool(stack_auc > best_base_auc + GATE_MARGIN),
        "stack_acc": float(((p_final >= 0.5).astype(int) == yte).mean()),
        "primary_acc": float(((prim[cut:] >= 0.5).astype(int) == yte).mean()),
        "stack_ece": _ece(yte, p_final),
        "primary_ece": _ece(yte, prim[cut:]),
        "per_regime": per_regime,
        "regime_train_counts": {r: int((rlabel[:cut] == r).sum()) for r in REGIMES},
    }

    # ── refit on ALL rows for the persisted serving model ────────────────────────
    scaler_full = StandardScaler().fit(X)
    Xs_full = scaler_full.transform(X)
    glob_full = _make_meta_learner().fit(Xs_full, y)
    experts_full = (
        _fit_experts(Xs_full, y, rlabel, glob_full, min_rows)
        if use_experts
        else {r: glob_full for r in REGIMES}
    )
    report["coef"] = dict(zip(BASE_SIGNALS, np.round(glob_full.coef_[0], 4).tolist()))
    return RegimeStacker(
        experts_full, glob_full, scaler_full, BASE_SIGNALS, primary_blend, use_experts, report
    ), report


# ══════════════════════════════════════════════════════════════════════════════════
#  OOF assembly + offline entrypoint
# ══════════════════════════════════════════════════════════════════════════════════
async def _assemble_oof():
    """
    Run the leak-free OOF base signals (primary + 2nd learner + meta + magnitude) and attach the
    causal HMM regime posteriors per event date. Returns (feats, y, ctx) where ctx carries the
    full feature matrix needed to refit the serving base-2 learner.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend.prediction.cv import PurgedWalkForwardSplit
    from backend.prediction.regime import decode_regimes
    from backend.prediction.train import (
        EMBARGO,
        N_SPLITS,
        _make_regressor,
        generate_oof_signals,
        load_dataset,
    )

    X, y, w, t1, feat_cols, fd_orders, data = await load_dataset()
    primary_cal, meta_full, _iso, _mrep = generate_oof_signals(X, y, w, t1, feat_cols)

    # 2nd diverse base learner (decorrelated linear view), same purged splitter.
    p2 = oof_base2(X, y, w, t1, feat_cols)

    # Magnitude OOF (Layer-2b) — the σ-ish point return for the stack.
    ret = data["_ret"].values.astype(float)
    cv = PurgedWalkForwardSplit(n_splits=N_SPLITS, embargo=EMBARGO)
    mag = np.full(len(ret), np.nan)
    for tr, te in cv.split(X, t1):
        m = _make_regressor()
        m.fit(X.iloc[tr][feat_cols], ret[tr], sample_weight=w[tr])
        mag[te] = m.predict(X.iloc[te][feat_cols])

    # Causal HMM regime posteriors per event date (gate (b) + the rejected-feature ablation (a)).
    reg = await decode_regimes()
    dates = pd.to_datetime(data["_date"].values).normalize()
    full_idx = pd.date_range(reg.index.min().normalize(), dates.max())
    reg_daily = reg.reindex(full_idx).ffill()
    rd = reg_daily.reindex(dates)
    regime_label = rd["regime"].fillna("trend").values
    p_tr = rd["p_trend"].fillna(1 / 3).values
    p_ch = rd["p_chop"].fillna(1 / 3).values
    p_ro = rd["p_risk_off"].fillna(1 / 3).values

    df = (
        pd.DataFrame(
            {
                "primary_cal": primary_cal,
                "p2_base": p2,
                "meta_prob": meta_full,
                "mag_oof": mag,
                "p_trend": p_tr,
                "p_chop": p_ch,
                "p_risk_off": p_ro,
                "regime": regime_label,
                "_date": dates,
                "_y": y,
            }
        )
        .dropna(subset=OOF_COLS + ["_y"])
        .sort_values("_date")
        .reset_index(drop=True)
    )

    ctx = {"X": X, "y": y, "w": w, "feat_cols": feat_cols}
    return df, df["_y"].values.astype(int), ctx


async def fit_and_save() -> dict:
    """Offline entrypoint: train the regime stack on OOF signals, persist regime_stack.pkl + base2.pkl."""
    import joblib

    t0 = time.time()
    print("Assembling OOF base signals (primary + base2 + meta + magnitude + regime posteriors)...")
    feats, y, ctx = await _assemble_oof()
    print(
        f"Stack training rows: {len(feats):,}  "
        f"(regime mix: {pd.Series(feats['regime']).value_counts().to_dict()})"
    )

    stacker, rep = train_regime_stack(feats, y)
    stacker.save(MODELS_DIR / "regime_stack.pkl")

    # Persist the serving base-2 learner (refit on ALL data) so predict.py can produce p2_base live.
    print("Refitting serving base-2 learner on all data...")
    base2 = make_base2()
    base2.fit(ctx["X"][ctx["feat_cols"]], ctx["y"], logisticregression__sample_weight=ctx["w"])
    joblib.dump({"model": base2, "feature_columns": ctx["feat_cols"]}, MODELS_DIR / "base2.pkl")

    print("\nRegime-conditional stack (time-split OOS eval):")
    print(
        f"  base learners : primary {rep['primary_auc']:.4f}  |  base2(EN) {rep['p2_auc']:.4f}  "
        f"(corr {rep['p2_corr_primary']:.3f})  -> best base = {rep['best_base_name']} "
        f"{rep['best_base_auc']:.4f}"
    )
    print(
        f"  shipped stack : {rep['stack_auc']:.4f}   (unshrunk {rep['stack_unshrunk_auc']:.4f}, "
        f"blend->primary {rep['primary_blend']:.2f})  lift vs best base {rep['lift_vs_best_base']:+.4f}"
    )
    print(
        f"  regime ablation: global {rep['global_stack_auc']:.4f}  |  MoE {rep['moe_stack_auc']:.4f}  "
        f"|  +regime-as-feature {rep['regime_feature_stack_auc']:.4f}  "
        f"-> use_experts={rep['use_experts']}"
    )
    print(f"  ACC  : stack {rep['stack_acc']:.4f}  vs primary {rep['primary_acc']:.4f}")
    print(f"  ECE  : stack {rep['stack_ece']:.4f}  vs primary {rep['primary_ece']:.4f}")
    print("  per-regime held-out AUC (stack vs primary):")
    for r in REGIMES:
        pr = rep["per_regime"][r]
        if pr["stack_auc"] is not None:
            print(
                f"      {r:9} n={pr['n']:>6,}  stack {pr['stack_auc']:.4f}  primary {pr['primary_auc']:.4f}"
            )
        else:
            print(f"      {r:9} n={pr['n']:>6,}  (too few to score)")
    verdict = (
        "PASS — stack beats the best base learner; ENABLED at serving (self-gate clears)"
        if rep["gate6_pass"]
        else "FAIL — no lift over best base learner; stays self-gated OFF (honesty contract)"
    )
    print(f"\n  GATE-6 (stack AUC > best single base learner): {verdict}")
    print(f"  saved: regime_stack.pkl, base2.pkl  ({time.time() - t0:.1f}s)")
    return rep


if __name__ == "__main__":
    import asyncio

    asyncio.run(fit_and_save())
