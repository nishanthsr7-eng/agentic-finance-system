"""GATE-2 follow-up: can a CURATED crypto subset pass where the full block failed?
Tests a few hand-picked subsets (+ greedy forward selection) on the identical sample/splits."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import roc_auc_score  # noqa: E402


async def _main():
    from backend.prediction.crypto_features import CRYPTO_FEATURE_COLS
    from backend.prediction.datasources import CRYPTO_SYMBOLS
    from backend.prediction.train import evaluate_oof, load_dataset

    X, y, w, t1, feat_cols, _fd, data = await load_dataset()
    crypto_cols = [c for c in CRYPTO_FEATURE_COLS if c in feat_cols]
    base = [c for c in feat_cols if c not in CRYPTO_FEATURE_COLS]
    cmask = data["_sym"].isin(CRYPTO_SYMBOLS).values

    def auc(cols):
        r = evaluate_oof(X, y, w, t1, cols)
        m = r["_mask"] & cmask
        return float(roc_auc_score(y[m], r["_oof_p_full"][m]))

    base_auc = auc(base)
    print(f"base (no crypto) crypto-AUC = {base_auc:.4f}\n")

    subsets = {
        "btc_lead_lag only": ["btc_lead_lag"],
        "per-symbol non-broadcast": [
            "btc_lead_lag",
            "nvt",
            "oi_z",
            "dvol_chg",
            "dvol_level",
            "oi_to_vol",
            "oi_chg",
            "addr_growth",
            "tx_growth",
        ],
        "top3 by marginal lift": ["btc_lead_lag", "nvt", "oi_z"],
        "drop fng + funding block": [
            "btc_lead_lag",
            "nvt",
            "oi_z",
            "dvol_chg",
            "dvol_level",
            "oi_to_vol",
            "oi_chg",
            "addr_growth",
            "tx_growth",
        ],
    }
    for name, cols in subsets.items():
        a = auc(base + cols)
        print(
            f"  {name:30} +{len(cols):>2} cols  AUC {a:.4f}  delta {a - base_auc:+.4f}  "
            f"{'PASS' if a - base_auc >= 0.01 else ''}"
        )

    # Greedy forward selection: keep adding the single best-improving crypto col.
    print("\nGreedy forward selection:")
    chosen, cur = [], base_auc
    remaining = list(crypto_cols)
    while remaining:
        scored = [(c, auc(base + chosen + [c])) for c in remaining]
        c, a = max(scored, key=lambda t: t[1])
        if a <= cur + 1e-4:
            break
        chosen.append(c)
        remaining.remove(c)
        cur = a
        print(f"  + {c:18} -> AUC {a:.4f}  (delta vs base {a - base_auc:+.4f})")
    print(f"\nBest greedy subset: {chosen or '(none — nothing improved)'}")
    print(
        f"Best crypto-AUC {cur:.4f}  delta {cur - base_auc:+.4f}  "
        f"GATE-2 {'PASS' if cur - base_auc >= 0.01 else 'FAIL'}"
    )


if __name__ == "__main__":
    asyncio.run(_main())
