"""
Phase 1 — activate FRED macro (term + credit spread, rates, CPI) and RETRAIN, then check GATE-1.

FRED_API_KEY is set, so this runs train.py AS-IS (FRED ON → 47-feature model) plus the portfolio
backtest for the deployable Sharpe. It then compares against the locked Phase-0 baseline (read from
models/baseline_backup/model_meta_gate0.json) and writes a `gate1` block into the new model_meta.json.

GATE-1 (from Workflow.md): OOF AUC >= baseline AND ECE not worse (macro must not hurt).

    python scripts/gate1_fred_macro_retrain.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import settings  # noqa: E402

MODELS = Path(__file__).resolve().parents[1] / "backend" / "prediction" / "models"
GATE0_FP = MODELS / "baseline_backup" / "model_meta_gate0.json"
ECE_TOL = 0.005  # "not worse" tolerance on calibrated ECE (abs)


async def _main():
    assert settings.FRED_API_KEY, "FRED_API_KEY not set — Phase 1 needs FRED ON"
    from backend.prediction import portfolio, train

    print("=" * 70, "\nPhase 1 — retrain with FRED macro ON\n", "=" * 70, sep="")
    await train.main()

    # Deployable Sharpe (FRED-on panel), same config as GATE-0.
    panel, reg_daily = await portfolio.build_panel()
    B = portfolio.benchmark(panel)
    LO = portfolio.run_portfolio(panel, reg_daily, frac=0.10, long_short=False)
    LOG = portfolio.run_portfolio(panel, reg_daily, frac=0.10, long_short=False, use_regime=True)
    deployable = max(LO["sharpe"], LOG["sharpe"])

    # New metrics
    meta = json.loads((MODELS / "model_meta.json").read_text())
    m = meta["metrics"]
    new_auc, new_ece_raw, new_ece_cal = m["model_auc"], m["ece_raw"], m["ece_cal"]
    prec60 = next(
        (
            b["precision"]
            for b in meta.get("meta_report", {}).get("buckets", [])
            if b["thr"] == 0.60
        ),
        None,
    )
    fred_cols = [
        c
        for c in ("term_spread", "credit_spread", "dgs10", "fed_funds", "cpi_yoy")
        if c in meta["feature_columns"]
    ]

    # Baseline (Phase 0)
    g0 = json.loads(GATE0_FP.read_text()).get("gate0_baseline", {})
    base_auc, base_ece_cal = g0.get("oof_auc"), g0.get("ece_cal")
    base_sharpe = g0.get("deployable_sharpe")

    auc_ok = new_auc >= base_auc
    ece_ok = new_ece_cal <= base_ece_cal + ECE_TOL
    gate1_pass = auc_ok and ece_ok

    meta["gate0_baseline"] = g0  # re-attach for side-by-side reference
    meta["gate1"] = {
        "note": "FRED macro ON (term+credit spread, rates, CPI). GATE-1 = AUC>=baseline & ECE not worse.",
        "fred_features": fred_cols,
        "n_features": len(meta["feature_columns"]),
        "baseline_auc": base_auc,
        "fred_auc": new_auc,
        "auc_delta": new_auc - base_auc,
        "baseline_ece_cal": base_ece_cal,
        "fred_ece_cal": new_ece_cal,
        "fred_ece_raw": new_ece_raw,
        "precision_at_meta_0.60": prec60,
        "baseline_sharpe": base_sharpe,
        "fred_deployable_sharpe": deployable,
        "auc_not_worse": bool(auc_ok),
        "ece_not_worse": bool(ece_ok),
        "GATE1_PASS": bool(gate1_pass),
        "recorded_at": int(time.time() * 1000),
    }
    (MODELS / "model_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    print("\n" + "=" * 70)
    print("GATE-1 RESULT")
    print("=" * 70)
    print(f"  features         : {len(meta['feature_columns'])} (FRED: {fred_cols})")
    print(
        f"  OOF AUC          : baseline {base_auc:.4f} -> FRED {new_auc:.4f}  "
        f"(delta {new_auc - base_auc:+.4f})  {'OK' if auc_ok else 'WORSE'}"
    )
    print(
        f"  ECE (calibrated) : baseline {base_ece_cal:.4f} -> FRED {new_ece_cal:.4f}  "
        f"{'OK' if ece_ok else 'WORSE'}"
    )
    print(f"  ECE (raw)        : {new_ece_raw:.4f}")
    print(f"  precision@0.60   : {prec60:.4f}")
    print(f"  deployable Sharpe: baseline {base_sharpe:.3f} -> FRED {deployable:.3f}")
    print(
        f"\n  GATE-1: {'PASS — macro does not hurt' if gate1_pass else 'FAIL — macro hurts, revert/drop'}"
    )


if __name__ == "__main__":
    asyncio.run(_main())
