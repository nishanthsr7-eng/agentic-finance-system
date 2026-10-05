"""
GATE-2 ablation — does the crypto-native feature block earn its place on the CRYPTO subset?

Same like-for-like discipline as gate1_fred_macro_ablation.py: load the pooled dataset ONCE (crypto features
ON), then run purged-walk-forward OOF on the IDENTICAL rows/splits with two feature sets —
  • WITH crypto    : all features
  • WITHOUT crypto : the same set minus the 14 CRYPTO_FEATURE_COLS
The only difference is the crypto block, so ΔAUC is its pure contribution. Because the columns are
all-zero on equities, the model is trained pooled (realistic) but we score AUC on the CRYPTO ROWS
ONLY — that's where the block can possibly help.

GATE-2: crypto-subset OOF AUC(with) >= AUC(without) + 0.01   (target ≈ 0.55+).
If it fails, the per-feature contribution table below shows which columns to drop / rebuild.

    python scripts/experiments/gate2_crypto_features_ablation.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.metrics import roc_auc_score  # noqa: E402


def _crypto_subset_auc(X, y, t1, w, feat_cols, crypto_mask):
    """Pooled OOF (train on everything) → AUC measured on the crypto rows only."""
    from backend.prediction.train import evaluate_oof

    r = evaluate_oof(X, y, w, t1, feat_cols)
    oof_full, tested = r["_oof_p_full"], r["_mask"]
    m = tested & crypto_mask
    return float(roc_auc_score(y[m], oof_full[m])), int(m.sum()), r["model_auc"]


async def _main():
    from backend.prediction.crypto_features import CRYPTO_FEATURE_COLS
    from backend.prediction.datasources import CRYPTO_SYMBOLS
    from backend.prediction.train import load_dataset

    X, y, w, t1, feat_cols, _fd, data = await load_dataset()
    crypto_cols = [c for c in CRYPTO_FEATURE_COLS if c in feat_cols]
    price_cols = [c for c in feat_cols if c not in CRYPTO_FEATURE_COLS]
    crypto_mask = data["_sym"].isin(CRYPTO_SYMBOLS).values

    print(
        f"Sample: {len(X):,} events ({crypto_mask.sum():,} crypto / {(~crypto_mask).sum():,} equity)"
    )
    print(f"Features: {len(feat_cols)} total = {len(price_cols)} base + {len(crypto_cols)} crypto")
    print(f"Crypto columns: {crypto_cols}\n")

    auc_w, n_c, full_w = _crypto_subset_auc(X, y, t1, w, feat_cols, crypto_mask)
    auc_o, _, full_o = _crypto_subset_auc(X, y, t1, w, price_cols, crypto_mask)
    delta = auc_w - auc_o

    print("=" * 66)
    print("  GATE-2 ABLATION  (identical sample & splits; AUC on crypto rows)")
    print("=" * 66)
    print(f"  crypto OOF rows scored : {n_c:,}")
    print(f"  {'':18}{'crypto AUC':>12}{'pooled AUC':>12}")
    print(f"  {'base (no crypto)':18}{auc_o:>12.4f}{full_o:>12.4f}")
    print(f"  {'+ crypto block':18}{auc_w:>12.4f}{full_w:>12.4f}")
    print(f"  {'delta (crypto)':18}{delta:>+12.4f}")
    print("-" * 66)
    passed = delta >= 0.01
    print(
        f"  GATE-2 (>= +0.0100 on crypto subset): "
        f"{'PASS' if passed else 'FAIL'}  (delta {delta:+.4f}, target AUC>=0.55 -> {auc_w:.4f})"
    )

    # Per-feature leave-one-IN contribution: AUC(base + this one column) - AUC(base). Identifies the
    # weakest features to drop if the gate is borderline/failing.
    print("\n  Per-feature crypto-subset AUC lift (base + single column):")
    print(f"    {'feature':18}{'AUC':>10}{'lift':>10}")
    lifts = []
    for c in crypto_cols:
        a, _, _ = _crypto_subset_auc(X, y, t1, w, price_cols + [c], crypto_mask)
        lifts.append((c, a, a - auc_o))
    for c, a, lift in sorted(lifts, key=lambda x: -x[2]):
        print(f"    {c:18}{a:>10.4f}{lift:>+10.4f}")


if __name__ == "__main__":
    asyncio.run(_main())
