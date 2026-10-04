"""
PHASE-6 DIAGNOSTIC — why does the stack fail GATE-6, and is there an honest design that passes?
Assembles the OOF stack frame ONCE (cached to scripts/_gate6_oof_cache.pkl), then evaluates several
stack designs on the SAME early-70/late-30 time split so we can compare like-for-like:

  base       : best single base learner (primary_cal, p2_base)  [the bar to beat]
  V1 global  : one logistic over BASE SIGNALS ONLY (no regime feats)
  V2 global+r: one logistic over base + regime posteriors        [the current global stack]
  V3 MoE base: regime experts over BASE SIGNALS ONLY, gated by posterior
  V4 MoE+r   : regime experts over base + regime posteriors      [the current production design]
  V5 shrink  : V1 blended 50/50 with primary_cal (robust shrink toward the calibrated base)

Also reports corr(primary, p2) — if ~1 there is no diversity for stacking to exploit.

    python scripts/gate6_regime_ensemble_diagnostic.py
"""

from __future__ import annotations

import asyncio
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

CACHE = Path(__file__).parent / "_gate6_oof_cache.pkl"


async def _get_feats():
    if CACHE.exists():
        print(f"Loading cached OOF from {CACHE.name}")
        with open(CACHE, "rb") as f:
            d = pickle.load(f)
        return d["feats"], d["y"]
    from backend.prediction.ensemble import _assemble_oof

    feats, y, _ctx = await _assemble_oof()
    with open(CACHE, "wb") as f:
        pickle.dump({"feats": feats, "y": y}, f)
    print(f"Cached OOF to {CACHE.name}")
    return feats, y


def _auc(y, p):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def _fit_logit(Xtr, ytr, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xtr), ytr)
    return sc, clf


def _moe(feats, y, cols, gate=True, C=1.0):
    """Fit per-regime (or single) logistic on early 70%, score late 30% softly gated by posterior."""
    from backend.prediction.ensemble import REGIME_PROBS, REGIMES

    X = feats[cols].values.astype(float)
    rp = feats[REGIME_PROBS].values.astype(float)
    rl = feats["regime"].astype(str).values
    n = len(X)
    cut = int(n * 0.7)
    sc, glob = _fit_logit(X[:cut], y[:cut], C)
    Xs = sc.transform(X)
    if not gate:
        return glob.predict_proba(Xs[cut:])[:, 1]
    experts = {}
    for r in REGIMES:
        m = rl[:cut] == r
        if m.sum() >= 500 and len(np.unique(y[:cut][m])) > 1:
            from sklearn.linear_model import LogisticRegression

            experts[r] = LogisticRegression(C=C, max_iter=2000).fit(Xs[:cut][m], y[:cut][m])
        else:
            experts[r] = glob
    pred = np.zeros(n - cut)
    wsum = np.zeros(n - cut)
    for j, r in enumerate(REGIMES):
        pr = experts[r].predict_proba(Xs[cut:])[:, 1]
        wj = np.clip(rp[cut:, j], 0, None)
        pred += wj * pr
        wsum += wj
    return np.where(
        wsum > 1e-9, pred / np.where(wsum > 1e-9, wsum, 1), glob.predict_proba(Xs[cut:])[:, 1]
    )


async def _main():
    from backend.prediction.ensemble import BASE_SIGNALS, STACK_COLS

    feats, y = await _get_feats()
    n = len(feats)
    cut = int(n * 0.7)
    yte = y[cut:]
    prim = feats["primary_cal"].values
    p2 = feats["p2_base"].values

    corr = float(np.corrcoef(prim, p2)[0, 1])
    best_base = max(_auc(yte, prim[cut:]), _auc(yte, p2[cut:]))
    print(f"\nrows {n:,} | late-eval {len(yte):,} | corr(primary, p2) = {corr:.3f}")
    print(f"{'design':14}{'AUC(late)':>12}{'vs best base':>14}")
    print(f"{'best base':14}{best_base:>12.4f}{0.0:>+14.4f}")

    variants = {
        "V1 global": _moe(feats, y, BASE_SIGNALS, gate=False),
        "V2 global+r": _moe(feats, y, STACK_COLS, gate=False),
        "V3 MoE base": _moe(feats, y, BASE_SIGNALS, gate=True),
        "V4 MoE+r": _moe(feats, y, STACK_COLS, gate=True),
    }
    # V5: shrink V1 toward calibrated primary (robust to non-stationary stack drift)
    variants["V5 shrink"] = 0.5 * variants["V1 global"] + 0.5 * prim[cut:]
    for name, p in variants.items():
        a = _auc(yte, p)
        print(f"{name:14}{a:>12.4f}{a - best_base:>+14.4f}")

    # also: does a stronger L2 (less overfit) help the global stack?
    for C in (0.1, 0.03, 0.01):
        p = _moe(feats, y, BASE_SIGNALS, gate=False, C=C)
        a = _auc(yte, p)
        print(f"{'V1 C=' + str(C):14}{a:>12.4f}{a - best_base:>+14.4f}")


if __name__ == "__main__":
    asyncio.run(_main())
