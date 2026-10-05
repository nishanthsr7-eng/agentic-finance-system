"""
Phase 2 — Sharpe ablation: does the crypto-native block help/hurt the DEPLOYABLE crypto book?

GATE-2 (AUC) already fails, but AUC and the deployable cross-sectional Sharpe disagreed for FRED
(Phase 1), so we check the money metric too before shelving the block for Phase 6. Same isolation as
gate1_fred_macro_sharpe_ablation.py: load ONE panel with crypto features ON, generate OOF signals with the full
feature set vs the price-only set, then run the SAME edge-ranked portfolio — restricted to the CRYPTO
universe (the model is still trained pooled on everything; only the book is crypto). Only difference =
the 14 crypto columns.

    python scripts/experiments/gate2_crypto_features_sharpe_ablation.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

os.environ["FLUX_CRYPTO_FEATURES"] = "1"  # force the block ON so both feature sets exist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402


async def _main():
    from backend.prediction import portfolio
    from backend.prediction.crypto_features import CRYPTO_FEATURE_COLS
    from backend.prediction.datasources import CRYPTO_SYMBOLS
    from backend.prediction.regime import decode_regimes
    from backend.prediction.train import generate_oof_signals, load_dataset

    X, y, w, t1, feat_cols, _fd, data = await load_dataset()
    crypto_cols = [c for c in CRYPTO_FEATURE_COLS if c in feat_cols]
    price_cols = [c for c in feat_cols if c not in CRYPTO_FEATURE_COLS]
    cmask = data["_sym"].isin(CRYPTO_SYMBOLS).values

    reg = await decode_regimes()
    reg_daily = (
        reg["regime"]
        .reindex(pd.date_range(reg.index.min(), pd.to_datetime(data["_date"]).max()))
        .ffill()
    )
    print(
        f"Panel: {len(X):,} events ({cmask.sum():,} crypto) | "
        f"with-crypto {len(feat_cols)} feats / price-only {len(price_cols)}"
    )
    print(f"Crypto columns: {crypto_cols}\n")

    def metrics_for(cols):
        prim, meta, _iso, _m = generate_oof_signals(X, y, w, t1, cols)
        panel = (
            pd.DataFrame(
                {
                    "date": pd.to_datetime(data["_date"].values),
                    "sym": data["_sym"].values,
                    "edge": prim - 0.5,
                    "meta": meta,
                    "ret": data["_ret"].values,
                }
            )[cmask]
            .dropna(subset=["edge", "ret"])
            .sort_values("date")
            .reset_index(drop=True)
        )
        out = {}
        for frac in (0.33, 0.20, 0.10):
            out[f"LO{int(frac * 100)}"] = portfolio.run_portfolio(
                panel, reg_daily, frac=frac, long_short=False
            ).get("sharpe")
        out["LO10+reg"] = portfolio.run_portfolio(
            panel, reg_daily, frac=0.10, long_short=False, use_regime=True
        ).get("sharpe")
        out["LS20"] = portfolio.run_portfolio(panel, reg_daily, frac=0.20, long_short=True).get(
            "sharpe"
        )
        out["bench"] = portfolio.benchmark(panel).get("sharpe")
        bm = portfolio.benchmark(panel)
        out["_periods"] = bm.get("periods")
        return out

    W = metrics_for(feat_cols)
    O = metrics_for(price_cols)

    cfgs = ["LO33", "LO20", "LO10", "LO10+reg", "LS20", "bench"]
    print("=" * 78)
    print("  PHASE-2 SHARPE ABLATION — crypto-only book (identical panel & splits)")
    print("=" * 78)
    print(f"  rebalances scored: {W.get('_periods')}")
    print(f"  {'feature set':16}" + "".join(f"{c:>10}" for c in cfgs))
    print(f"  {'price-only':16}" + "".join(f"{(O.get(c) or 0):>10.3f}" for c in cfgs))
    print(f"  {'+ crypto block':16}" + "".join(f"{(W.get(c) or 0):>10.3f}" for c in cfgs))
    print(
        f"  {'delta (crypto)':16}"
        + "".join(f"{((W.get(c) or 0) - (O.get(c) or 0)):>+10.3f}" for c in cfgs)
    )
    print("-" * 78)

    # Deployable = best of the long-only configs (the project's chosen live config).
    dep = ["LO33", "LO20", "LO10", "LO10+reg"]
    best_o = max((O.get(c) or 0) for c in dep)
    best_w = max((W.get(c) or 0) for c in dep)
    verdict = (
        "HELPS"
        if best_w >= best_o + 0.02
        else ("NEUTRAL" if abs(best_w - best_o) < 0.02 else "HURTS")
    )
    print(
        f"  deployable (best long-only) Sharpe: price-only {best_o:.3f}  ->  +crypto {best_w:.3f}  "
        f"=> {verdict}"
    )
    print(f"  GATE-2 (AUC) already FAILED (+0.0055 < +0.01); Sharpe check: {verdict}")


if __name__ == "__main__":
    asyncio.run(_main())
