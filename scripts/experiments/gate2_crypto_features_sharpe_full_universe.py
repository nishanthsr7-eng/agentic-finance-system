"""
Phase 2 — deployment-decision Sharpe ablation: crypto block on the FULL universe vs crypto-only book.

The crypto-only book Sharpe improves with the block (gate2_crypto_features_sharpe_ablation.py), but the production
model trades the full 33-symbol universe. This runs ONE OOF pass per feature set (with/without the 14
crypto cols) and scores the deployable long-only portfolio on BOTH books from the same signals:
  • FULL universe (all 33 symbols)  -> decides the production default
  • CRYPTO-only subset              -> confirms the crypto-book win
Only difference between columns = the crypto block (all-zero on equities, so equities are unaffected
except via the shared pooled fit).

    python scripts/experiments/gate2_crypto_features_sharpe_full_universe.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

os.environ["FLUX_CRYPTO_FEATURES"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


async def _main():
    from backend.prediction import portfolio
    from backend.prediction.crypto_features import CRYPTO_FEATURE_COLS
    from backend.prediction.datasources import CRYPTO_SYMBOLS
    from backend.prediction.regime import decode_regimes
    from backend.prediction.train import generate_oof_signals, load_dataset

    X, y, w, t1, feat_cols, _fd, data = await load_dataset()
    price_cols = [c for c in feat_cols if c not in CRYPTO_FEATURE_COLS]
    cmask = data["_sym"].isin(CRYPTO_SYMBOLS).values
    syms = data["_sym"].values
    dates = pd.to_datetime(data["_date"].values)
    rets = data["_ret"].values

    reg = await decode_regimes()
    reg_daily = reg["regime"].reindex(pd.date_range(reg.index.min(), dates.max())).ffill()
    print(
        f"Panel: {len(X):,} events ({cmask.sum():,} crypto / {(~cmask).sum():,} equity) | "
        f"with-crypto {len(feat_cols)} / price-only {len(price_cols)}\n"
    )

    def books_for(cols):
        prim, meta, _iso, _m = generate_oof_signals(X, y, w, t1, cols)
        full = pd.DataFrame(
            {"date": dates, "sym": syms, "edge": prim - 0.5, "meta": meta, "ret": rets}
        ).dropna(subset=["edge", "ret"])
        out = {}
        for book, panel in (("FULL", full), ("CRYPTO", full[full["sym"].isin(CRYPTO_SYMBOLS)])):
            p = panel.sort_values("date").reset_index(drop=True)
            row = {}
            for frac in (0.33, 0.20, 0.10):
                row[f"LO{int(frac * 100)}"] = portfolio.run_portfolio(
                    p, reg_daily, frac=frac, long_short=False
                ).get("sharpe")
            row["LO10+reg"] = portfolio.run_portfolio(
                p, reg_daily, frac=0.10, long_short=False, use_regime=True
            ).get("sharpe")
            row["bench"] = portfolio.benchmark(p).get("sharpe")
            out[book] = row
        return out

    W = books_for(feat_cols)
    O = books_for(price_cols)

    cfgs = ["LO33", "LO20", "LO10", "LO10+reg", "bench"]
    for book in ("FULL", "CRYPTO"):
        print("=" * 74)
        print(f"  {book} universe — long-only Sharpe (identical panel & splits)")
        print("=" * 74)
        print(f"  {'feature set':16}" + "".join(f"{c:>11}" for c in cfgs))
        print(f"  {'price-only':16}" + "".join(f"{(O[book].get(c) or 0):>11.3f}" for c in cfgs))
        print(f"  {'+ crypto block':16}" + "".join(f"{(W[book].get(c) or 0):>11.3f}" for c in cfgs))
        print(
            f"  {'delta (crypto)':16}"
            + "".join(f"{((W[book].get(c) or 0) - (O[book].get(c) or 0)):>+11.3f}" for c in cfgs)
        )
        dep = ["LO33", "LO20", "LO10", "LO10+reg"]
        bo = max(O[book].get(c) or 0 for c in dep)
        bw = max(W[book].get(c) or 0 for c in dep)
        v = "HELPS" if bw >= bo + 0.02 else ("NEUTRAL" if abs(bw - bo) < 0.02 else "HURTS")
        print(f"  deployable best long-only: price-only {bo:.3f} -> +crypto {bw:.3f}  => {v}\n")

    print(
        "DECISION RULE: enable in production if FULL universe is HELPS/NEUTRAL; "
        "otherwise keep OFF and use only for a crypto-only book (which is HELPS)."
    )


if __name__ == "__main__":
    asyncio.run(_main())
