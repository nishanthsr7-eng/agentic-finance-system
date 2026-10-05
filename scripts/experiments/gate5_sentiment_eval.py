"""
GATE-5 — Multi-source sentiment: does the trainable sentiment block (GDELT tone + Alpha Vantage news
sentiment, point-in-time historical) beat the prior pooled model on OOF AUC?

Method (one panel, like-for-like — same as the Phase-2/4 gates):
  Build the production pooled panel ONCE with the sentiment block joined (FLUX_SENTIMENT_FEATURES=1).
  The block is neutral-(0)-filled, so it adds NO NaN -> the row set is byte-identical to the prior
  panel; only the feature COLUMNS differ. Then run the same leak-free PurgedWalkForwardSplit twice:
    BASE = evaluate_oof on the prior feature columns               (the current pooled model)
    AUG  = evaluate_oof on prior columns + the sentiment columns   (Phase-5)
  Score per class on the rows out-of-fold-tested under BOTH (shared rows).

COVERAGE NOTE: the sentiment streams are best-effort (GDELT is keyless but throttles to 429; AV
NEWS_SENTIMENT needs a key + is rate-limited to ~25 req/day). If no historical sentiment has been
back-filled into the caches yet, every sentiment column is neutral 0 -> AUG == BASE by construction and
the gate is a TIE *for lack of data*, NOT evidence the signal is dead. The harness detects and says so.
Back-fill via:  python -m backend.prediction.datasources.gdelt_tone --fetch
                python -m backend.prediction.datasources.av_news   --fetch
then re-run this gate. Per the honesty contract the block stays self-gated OFF until it shows real lift.

    python scripts/experiments/gate5_sentiment_eval.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

MARGIN = 1e-3


def _auc_pair(y, base, aug, mask):
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
    os.environ["FLUX_SENTIMENT_FEATURES"] = "1"
    os.environ.setdefault("FLUX_CRYPTO_FEATURES", "0")
    os.environ.setdefault("FLUX_EQUITY_FEATURES", "0")

    from backend.prediction.datasources import CRYPTO_SYMBOLS
    from backend.prediction.sentiment_features import SENTIMENT_FEATURE_COLS
    from backend.prediction.train import evaluate_oof, load_dataset

    X, y, w, t1, feat_cols, _fd, data = await load_dataset()
    sent_cols = [c for c in feat_cols if c in SENTIMENT_FEATURE_COLS]
    base_cols = [c for c in feat_cols if c not in SENTIMENT_FEATURE_COLS]
    assert sent_cols, (
        "sentiment block not present - is FLUX_SENTIMENT_FEATURES respected in load_dataset?"
    )

    coverage = int((data[sent_cols].abs().to_numpy() > 1e-9).any(axis=1).sum())
    crypto = data["_sym"].isin(CRYPTO_SYMBOLS).values
    equity = ~crypto
    classes = {"equity": equity, "crypto": crypto, "pooled": np.ones(len(y), bool)}
    print(
        f"Panel: {len(X):,} events ({equity.sum():,} equity / {crypto.sum():,} crypto) | "
        f"base {len(base_cols)} + sentiment {len(sent_cols)} feats {sent_cols}"
    )
    print(
        f"Sentiment coverage: {coverage:,}/{len(X):,} rows have a non-neutral sentiment value "
        f"({coverage / len(X):.1%})\n"
    )

    if coverage == 0:
        print("=" * 72)
        print(
            "  GATE-5: NO SENTIMENT COVERAGE in the caches -> block is all-neutral -> AUG == BASE."
        )
        print(
            "  This is a TIE FOR LACK OF DATA, not a dead signal. Back-fill the streams and re-run:"
        )
        print("    python -m backend.prediction.datasources.gdelt_tone --fetch")
        print("    python -m backend.prediction.datasources.av_news   --fetch")
        print("  Block stays self-gated OFF (FLUX_SENTIMENT_FEATURES=0) until it shows real lift.")
        print("=" * 72)
        return

    base_res = evaluate_oof(X, y, w, t1, base_cols)
    aug_res = evaluate_oof(X, y, w, t1, feat_cols)
    base, aug = base_res["_oof_p_full"], aug_res["_oof_p_full"]

    print("=" * 72)
    print("  GATE-5(a) - per-class OOF AUC: prior base vs base+sentiment (shared rows)")
    print("=" * 72)
    print(f"    {'class':8}{'n':>9}{'base':>10}{'+sent':>10}{'delta':>10}")
    deltas = {}
    for cname, cmask in classes.items():
        ab, aa, n = _auc_pair(y, base, aug, cmask)
        deltas[cname] = aa - ab
        print(f"    {cname:8}{n:>9,}{ab:>10.4f}{aa:>10.4f}{aa - ab:>+10.4f}")

    # GATE-5(b) proxy on the historical panel: calibration (ECE) of base vs augmented. (The *live*
    # calibration-improvement metric needs resolved flywheel outcomes over time; this is the offline
    # complement — does the sentiment feature make the OOF probabilities better calibrated?)
    base_ece, aug_ece = base_res["ece_raw"], aug_res["ece_raw"]
    print("\n  GATE-5(b) proxy - OOF calibration (ECE, lower is better):")
    print(
        f"    base ECE {base_ece:.4f}  ->  +sent ECE {aug_ece:.4f}   delta {aug_ece - base_ece:+.4f}"
    )

    pooled_delta = deltas["pooled"]
    print("\n" + "-" * 72)
    if pooled_delta > MARGIN:
        verdict = "PASS - sentiment lifts OOF AUC; ship ON only if ECE also not worse"
    elif pooled_delta >= -MARGIN:
        verdict = "TIE within noise - keep self-gated OFF (honesty contract)"
    else:
        verdict = "FAIL - sentiment hurts OOF AUC; keep self-gated OFF"
    print(f"  GATE-5(a) (pooled AUC >= prior): delta {pooled_delta:+.4f} -> {verdict}")
    print(
        f"  GATE-5(b) (ECE not worse):        delta {aug_ece - base_ece:+.4f} -> "
        f"{'OK' if aug_ece <= base_ece + 1e-4 else 'calibration worse'}"
    )


if __name__ == "__main__":
    asyncio.run(_main())
