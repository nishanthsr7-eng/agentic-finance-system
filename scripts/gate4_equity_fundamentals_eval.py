"""
GATE-4 — Equity fundamentals & events: does the Phase-4 fundamentals block (earnings surprise/drift,
revenue revision, valuation z, accruals — point-in-time from SEC filings) beat the Phase-3 pooled
model on the EQUITY subset's purged-OOF AUC?

Method (one panel, like-for-like — the cleanest comparison):
  Build the production pooled panel ONCE with the equity block joined (FLUX_EQUITY_FEATURES=1). The
  block is neutral-(0)-filled, so it adds NO NaN -> the row set is byte-identical to the Phase-3 panel;
  only the feature COLUMNS differ. Then run the same leak-free PurgedWalkForwardSplit twice on those
  identical rows:
    BASE  = evaluate_oof on the Phase-3 feature columns          (== the Phase-3 pooled model)
    AUG   = evaluate_oof on Phase-3 columns + the 5 equity cols  (Phase-4)
  Score each on the EQUITY rows that were out-of-fold-tested under BOTH (shared rows). Crypto + pooled
  AUC are reported too, as a guard that the neutral block doesn't disturb the rest.

GATE-4: equity-subset OOF AUC(AUG) >= AUC(BASE). A genuine lift (> +MARGIN) means ship the block ON
for the equity subset; a tie/decline within noise means keep it self-gated OFF (honesty contract),
exactly as crypto-features and FRED were handled.

    python scripts/gate4_equity_fundamentals_eval.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

MARGIN = 1e-3  # an AUC delta within ±MARGIN is a "tie" (no structural lift)


def _auc_on(y, p, mask):
    """AUC of OOF probs p on `mask` rows that were actually tested (non-NaN). (auc, n)."""
    m = mask & ~np.isnan(p)
    yt = y[m]
    if yt.size == 0 or yt.min() == yt.max():
        return float("nan"), int(m.sum())
    return float(roc_auc_score(yt, p[m])), int(m.sum())


def _auc_pair(y, base, aug, mask):
    """AUC of base & aug on the SAME rows: mask ∩ both-tested. (auc_base, auc_aug, n)."""
    common = mask & ~np.isnan(base) & ~np.isnan(aug)
    yt = y[common]
    if yt.size == 0 or yt.min() == yt.max():
        return float("nan"), float("nan"), int(common.sum())
    return (
        float(roc_auc_score(yt, base[common])),
        float(roc_auc_score(yt, aug[common])),
        int(common.sum()),
    )


async def _main():
    os.environ["FLUX_EQUITY_FEATURES"] = "1"  # join the equity block into the panel
    os.environ.setdefault("FLUX_CRYPTO_FEATURES", "0")

    from backend.prediction.datasources import CRYPTO_SYMBOLS
    from backend.prediction.equity_features import EQUITY_FEATURE_COLS
    from backend.prediction.train import evaluate_oof, load_dataset

    X, y, w, t1, feat_cols, _fd, data = await load_dataset()
    base_cols = [c for c in feat_cols if c not in EQUITY_FEATURE_COLS]
    eq_cols = [c for c in feat_cols if c in EQUITY_FEATURE_COLS]
    assert eq_cols, "equity block not present — is FLUX_EQUITY_FEATURES respected in load_dataset?"

    crypto = data["_sym"].isin(CRYPTO_SYMBOLS).values
    equity = ~crypto
    classes = {"equity": equity, "crypto": crypto, "pooled": np.ones(len(y), bool)}
    print(
        f"Panel: {len(X):,} events ({equity.sum():,} equity / {crypto.sum():,} crypto) | "
        f"base {len(base_cols)} + equity {len(eq_cols)} feats {eq_cols}\n"
    )

    base = evaluate_oof(X, y, w, t1, base_cols)["_oof_p_full"]
    aug = evaluate_oof(X, y, w, t1, feat_cols)["_oof_p_full"]

    print("=" * 70)
    print("  GATE-4 — per-class OOF AUC: Phase-3 base vs base+equity (shared rows)")
    print("=" * 70)
    print(f"    {'class':8}{'n':>9}{'base':>10}{'+equity':>10}{'delta':>10}")
    deltas = {}
    for cname, cmask in classes.items():
        ab, aa, n = _auc_pair(y, base, aug, cmask)
        deltas[cname] = aa - ab
        print(f"    {cname:8}{n:>9,}{ab:>10.4f}{aa:>10.4f}{aa - ab:>+10.4f}")

    eq_delta = deltas["equity"]
    print("\n" + "-" * 70)
    not_worse = eq_delta >= -MARGIN
    real_lift = eq_delta > MARGIN
    if real_lift:
        verdict = "PASS — equity AUC lifts; ship the block ON for the equity subset"
    elif not_worse:
        verdict = "TIE within noise — no structural lift; keep self-gated OFF (honesty contract)"
    else:
        verdict = "FAIL — equity AUC declines; keep self-gated OFF"
    print(
        f"  GATE-4 (equity >= Phase-3 equity): {'>=' if not_worse else '<'} baseline "
        f"(delta {eq_delta:+.4f}) -> {verdict}"
    )
    # Guard: the neutral block must not move the crypto subset.
    if abs(deltas["crypto"]) > 5e-3:
        print(
            f"  WARNING: crypto AUC moved by {deltas['crypto']:+.4f} — the block should be neutral there."
        )


if __name__ == "__main__":
    asyncio.run(_main())
