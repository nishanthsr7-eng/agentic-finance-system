"""
Phase 1 — Sharpe ablation: does FRED macro help or hurt the CROSS-SECTIONAL portfolio?

GATE-1 (AUC/ECE) passes, but the deployable Sharpe fell 0.835 -> ~0.46 after adding FRED. Macro
features are cross-sectionally CONSTANT (term_spread is the same for every symbol on a given day),
so they can't sharpen within-date ranking and may inject common-mode noise. This isolates the effect:
load ONE FRED-on panel, generate OOF signals with 47 feats vs the 42 price-only feats, and run the
SAME long-only top-decile portfolio on each. Only difference = the 5 macro columns.

    python scripts/experiments/gate1_fred_macro_sharpe_ablation.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from backend.config import settings  # noqa: E402

FRED_COLS = ["term_spread", "credit_spread", "dgs10", "fed_funds", "cpi_yoy"]


async def _main():
    assert settings.FRED_API_KEY, "needs FRED ON"
    from backend.prediction import portfolio
    from backend.prediction.regime import decode_regimes
    from backend.prediction.train import generate_oof_signals, load_dataset

    X, y, w, t1, feat_cols, _fd, data = await load_dataset()
    price_cols = [c for c in feat_cols if c not in FRED_COLS]
    reg = await decode_regimes()
    reg_daily = (
        reg["regime"]
        .reindex(pd.date_range(reg.index.min(), pd.to_datetime(data["_date"]).max()))
        .ffill()
    )
    print(
        f"Panel: {len(X):,} events | with-FRED {len(feat_cols)} feats / price-only {len(price_cols)}\n"
    )

    def sharpe_for(cols):
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
            )
            .dropna(subset=["edge", "ret"])
            .sort_values("date")
            .reset_index(drop=True)
        )
        LO = portfolio.run_portfolio(panel, reg_daily, frac=0.10, long_short=False)
        LOG = portfolio.run_portfolio(
            panel, reg_daily, frac=0.10, long_short=False, use_regime=True
        )
        B = portfolio.benchmark(panel)
        return LO["sharpe"], LOG["sharpe"], B["sharpe"], LO["max_dd"]

    lo_w, log_w, b_w, dd_w = sharpe_for(feat_cols)
    lo_o, log_o, b_o, dd_o = sharpe_for(price_cols)

    print("=" * 60)
    print("  SHARPE ABLATION (identical panel; deployable = long-only top decile)")
    print("=" * 60)
    print(f"  {'':14}{'LO Sharpe':>11}{'LO+regime':>11}{'benchmark':>11}")
    print(f"  {'price-only':14}{lo_o:>11.3f}{log_o:>11.3f}{b_o:>11.3f}")
    print(f"  {'+ FRED macro':14}{lo_w:>11.3f}{log_w:>11.3f}{b_w:>11.3f}")
    print(f"  {'delta (macro)':14}{lo_w - lo_o:>+11.3f}{log_w - log_o:>+11.3f}{b_w - b_o:>+11.3f}")
    print("-" * 60)
    best_o, best_w = max(lo_o, log_o), max(lo_w, log_w)
    print(
        f"  deployable Sharpe: price-only {best_o:.3f}  ->  +FRED {best_w:.3f}  "
        f"({'HELPS' if best_w >= best_o else 'HURTS'})"
    )


if __name__ == "__main__":
    asyncio.run(_main())
