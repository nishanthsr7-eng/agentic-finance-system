"""
GATE-1 ablation — the clean, like-for-like test of whether FRED macro helps or hurts.

Activating FRED truncates training to the FRED window (build_macro fetches from 2000-01-01), so the
FRED model's sample != the locked baseline's sample. Comparing their AUCs directly conflates "macro
effect" with "sample change". This script removes that confound: it loads the dataset ONCE (FRED ON),
then runs purged-walk-forward OOF on the SAME rows/splits with two feature sets —
  • WITH FRED   : all 47 features
  • WITHOUT FRED: the 42 price-only features (FRED columns dropped)
Only difference = the 5 macro columns ⇒ ΔAUC / ΔECE are the pure macro contribution.

GATE-1: macro must NOT hurt → AUC(with) >= AUC(without) on the identical sample, ECE not worse.

    python scripts/gate1_fred_macro_ablation.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import settings  # noqa: E402

FRED_COLS = ["term_spread", "credit_spread", "dgs10", "fed_funds", "cpi_yoy"]


async def _main():
    assert settings.FRED_API_KEY, "needs FRED ON"
    from backend.prediction.train import _ece, evaluate_oof, fit_calibrator, load_dataset

    X, y, w, t1, feat_cols, _fd, _data = await load_dataset()
    fred_in = [c for c in FRED_COLS if c in feat_cols]
    price_cols = [c for c in feat_cols if c not in FRED_COLS]
    print(
        f"Sample: {len(X):,} events | with-FRED {len(feat_cols)} feats / price-only {len(price_cols)} feats"
    )
    print(f"FRED columns in model: {fred_in}\n")

    def run(cols):
        r = evaluate_oof(X, y, w, t1, cols)
        iso = fit_calibrator(r["_oof_p"], r["_oof_y"])
        ece_cal = _ece(r["_oof_y"], iso.predict(r["_oof_p"]))
        return r["model_auc"], r["ece_raw"], ece_cal, r["model_acc"]

    auc_w, ece_raw_w, ece_cal_w, acc_w = run(feat_cols)
    auc_o, ece_raw_o, ece_cal_o, acc_o = run(price_cols)

    auc_ok = auc_w >= auc_o
    ece_ok = ece_cal_w <= ece_cal_o + 0.005
    print("=" * 64)
    print("  GATE-1 ABLATION (identical sample, identical splits)")
    print("=" * 64)
    print(f"  {'':16}{'AUC':>10}{'acc':>9}{'ECE_raw':>10}{'ECE_cal':>10}")
    print(f"  {'price-only':16}{auc_o:>10.4f}{acc_o:>9.4f}{ece_raw_o:>10.4f}{ece_cal_o:>10.4f}")
    print(f"  {'+ FRED macro':16}{auc_w:>10.4f}{acc_w:>9.4f}{ece_raw_w:>10.4f}{ece_cal_w:>10.4f}")
    print(
        f"  {'delta (macro)':16}{auc_w - auc_o:>+10.4f}{acc_w - acc_o:>+9.4f}"
        f"{ece_raw_w - ece_raw_o:>+10.4f}{ece_cal_w - ece_cal_o:>+10.4f}"
    )
    print("-" * 64)
    print(f"  AUC not worse: {auc_ok}   |   ECE not worse: {ece_ok}")
    print(
        f"  GATE-1 (ablation): {'PASS — macro helps/neutral' if (auc_ok and ece_ok) else 'FAIL — macro hurts'}"
    )


if __name__ == "__main__":
    asyncio.run(_main())
