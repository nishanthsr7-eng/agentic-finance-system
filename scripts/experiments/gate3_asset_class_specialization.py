"""
GATE-3 — Asset-class specialization: do crypto/equity experts (or an asset_type flag) beat the
single pooled model on EACH class's purged OOF AUC?

Three approaches on the IDENTICAL panel (production feature set; crypto-features OFF), each evaluated
with the same leak-free PurgedWalkForwardSplit:
  A. POOLED            — one model on all symbols (current production).
  B. POOLED + flag     — one model + an `is_crypto` asset_type feature (soft specialization; a tree
                         can branch on it and learn class-conditional sub-trees while sharing data).
  C. TWO EXPERTS       — a crypto-only model and an equity-only model (hard separation).

Per-class AUC is compared LIKE-FOR-LIKE: for each class we score every approach on the SAME rows —
the class rows that were out-of-fold-tested under BOTH approaches being compared.

GATE-3: a specialized approach passes only if it is >= pooled on BOTH classes' OOF AUC AND delivers a
genuine lift (> +MARGIN on at least one class, not merely a within-noise tie). The phase is about a
STRUCTURAL LIFT — a wash that is fractionally below pooled on both classes is a FAIL and we keep the
pooled model.

    python scripts/experiments/gate3_asset_class_specialization.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

MARGIN = 1e-3  # a class delta within ±MARGIN is a "tie" (no lift); strict pass needs a real gain


def _oof_full(X, y, w, t1, feat_cols):
    """Pooled OOF prob array (len N, NaN where never tested)."""
    from backend.prediction.train import evaluate_oof

    return evaluate_oof(X, y, w, t1, feat_cols)["_oof_p_full"]


def _expert_oof_full(X, y, w, t1, feat_cols, class_idx):
    """Train an expert on ONE class only; return a full-length OOF array (NaN off-class/untested).
    The subset of a date-sorted frame stays sorted, so subset position j -> full position pos[j]."""
    from backend.prediction.train import evaluate_oof

    pos = np.where(class_idx)[0]
    sub = evaluate_oof(X.iloc[pos], y[pos], w[pos], t1.iloc[pos], feat_cols)["_oof_p_full"]
    out = np.full(len(y), np.nan)
    out[pos] = sub
    return out


def _auc_pair(y, a, b, class_mask, name_a, name_b):
    """AUC of `a` and `b` on the SAME rows: class ∩ both-tested. Returns (auc_a, auc_b, n)."""
    common = class_mask & ~np.isnan(a) & ~np.isnan(b)
    yt = y[common]
    if yt.min() == yt.max():
        return float("nan"), float("nan"), int(common.sum())
    return (
        float(roc_auc_score(yt, a[common])),
        float(roc_auc_score(yt, b[common])),
        int(common.sum()),
    )


async def _main():
    from backend.prediction.datasources import CRYPTO_SYMBOLS
    from backend.prediction.train import load_dataset

    X, y, w, t1, feat_cols, _fd, data = await load_dataset()
    crypto = data["_sym"].isin(CRYPTO_SYMBOLS).values
    equity = ~crypto
    classes = {"crypto": crypto, "equity": equity}
    print(
        f"Panel: {len(X):,} events ({crypto.sum():,} crypto / {equity.sum():,} equity) | "
        f"{len(feat_cols)} features\n"
    )

    # A. pooled
    pooled = _oof_full(X, y, w, t1, feat_cols)

    # B. pooled + asset_type flag
    Xf = X.copy()
    Xf["is_crypto"] = crypto.astype(float)
    pooled_flag = _oof_full(Xf, y, w, t1, feat_cols + ["is_crypto"])

    # C. two experts (crypto-only model, equity-only model) -> merged full-length array
    experts = np.full(len(y), np.nan)
    for cm in classes.values():
        e = _expert_oof_full(X, y, w, t1, feat_cols, cm)
        experts[~np.isnan(e)] = e[~np.isnan(e)]

    print("=" * 72)
    print("  GATE-3 — per-class OOF AUC (like-for-like vs pooled on shared rows)")
    print("=" * 72)

    verdicts = {}
    for label, spec in (("B: pooled+flag", pooled_flag), ("C: two-experts", experts)):
        print(f"\n  {label}")
        print(f"    {'class':8}{'n':>9}{'pooled':>10}{'spec':>10}{'delta':>10}")
        deltas = []
        for cname, cmask in classes.items():
            ap, asp, n = _auc_pair(y, pooled, spec, cmask, "pooled", label)
            d = asp - ap
            deltas.append(d)
            print(f"    {cname:8}{n:>9,}{ap:>10.4f}{asp:>10.4f}{d:>+10.4f}")
        not_worse = all(d >= -MARGIN for d in deltas)  # >= pooled on both (within noise)
        real_lift = any(d > MARGIN for d in deltas)  # a genuine gain somewhere
        ok = not_worse and real_lift
        verdicts[label] = ok
        if ok:
            note = ">= pooled on both AND a real lift (PASS)"
        elif not_worse:
            note = "tie within noise, no lift -> not a structural win (FAIL)"
        else:
            note = "worse on a class (FAIL)"
        print(f"    -> {note}")

    print("\n" + "-" * 72)
    any_pass = any(verdicts.values())
    winner = next((k for k, v in verdicts.items() if v), None)
    print(f"  GATE-3: {'PASS via ' + winner if any_pass else 'FAIL — keep the pooled model'}")
    print(
        "  -> refactor train.py to adopt it."
        if any_pass
        else "  -> no structural lift; keep the pooled model (the low-risk default)."
    )


if __name__ == "__main__":
    asyncio.run(_main())
