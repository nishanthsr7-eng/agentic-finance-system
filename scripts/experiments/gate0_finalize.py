"""
GATE-0 finalize — reproduce the deployable portfolio Sharpe and LOG it to model_meta.json.

train.py already logged OOF AUC / ECE / precision. backtest.py reproduces the per-symbol stream.
The DEPLOYABLE Sharpe, however, comes from the cross-sectional portfolio (portfolio.py): long-only
edge-ranked top-decile, with and without the regime gate. This script reruns that panel (FRED OFF,
to match the price-only baseline), prints the table, and writes a `gate0_baseline` block into
models/model_meta.json so the single artifact carries every Phase-0 headline number FLUX-X must beat.

    python scripts/experiments/gate0_finalize.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.config import settings

settings.FRED_API_KEY = ""  # price-only Phase-0 baseline

from backend.prediction import portfolio  # noqa: E402

MODELS = Path(__file__).resolve().parents[2] / "backend" / "prediction" / "models"


async def _main():
    panel, reg_daily = await portfolio.build_panel()
    B = portfolio.benchmark(panel)
    LO = portfolio.run_portfolio(panel, reg_daily, frac=0.10, long_short=False)
    LOG = portfolio.run_portfolio(panel, reg_daily, frac=0.10, long_short=False, use_regime=True)
    LS = portfolio.run_portfolio(panel, reg_daily, frac=0.20, long_short=True)

    print(
        f"Panel: {len(panel):,} symbol-events, {panel['sym'].nunique()} symbols, "
        f"{B['periods']} rebalances"
    )
    for name, m in [
        ("long-only top10%", LO),
        ("long-only top10% +regime", LOG),
        ("long/short top-bot20%", LS),
        ("benchmark eq-wt long", B),
    ]:
        print(
            f"  {name:<28} Sharpe {m['sharpe']:>5.2f} | maxDD {m['max_dd']:>6.1%} "
            f"| ann {m['ann_return']:>+6.1%} | totRet {m['total_return']:>+8.1%}"
        )

    deployable = max(LO["sharpe"], LOG["sharpe"])
    strip = lambda m: {k: v for k, v in m.items() if not k.startswith("_")}

    meta_fp = MODELS / "model_meta.json"
    meta = json.loads(meta_fp.read_text())
    tr = meta.get("metrics", {})
    prec60 = next(
        (
            b["precision"]
            for b in meta.get("meta_report", {}).get("buckets", [])
            if b["thr"] == 0.60
        ),
        None,
    )
    meta["gate0_baseline"] = {
        "note": "Locked price-only baseline FLUX-X must beat (FRED OFF = Phase 0). "
        "Deployable config = long-only edge-ranked top-decile + regime gate.",
        "oof_auc": tr.get("model_auc"),
        "oof_acc": tr.get("model_acc"),
        "ece_raw": tr.get("ece_raw"),
        "ece_cal": tr.get("ece_cal"),
        "precision_at_meta_0.60": prec60,
        "deployable_sharpe": deployable,
        "portfolio": {
            "long_only_top10": strip(LO),
            "long_only_top10_regime": strip(LOG),
            "long_short_top20": strip(LS),
            "benchmark_eqw_long": strip(B),
        },
        "recorded_at": int(time.time() * 1000),
    }
    meta_fp.write_text(json.dumps(meta, indent=2, default=str))
    print(
        f"\nGATE-0: deployable Sharpe {deployable:.2f} (benchmark {B['sharpe']:.2f}); "
        f"logged gate0_baseline -> {meta_fp.relative_to(MODELS.parents[2])}"
    )


if __name__ == "__main__":
    asyncio.run(_main())
