"""
PHASE-5 (item 3) — FinBERT scoring-quality eval on the Financial PhraseBank.

Confirms the sentiment scorer FLUX relies on actually classifies financial sentiment well, on a labeled
gold set (Malo et al. 2014, ``Dataset/financial_phrasebank/Sentences_<agree>Agree.csv``). This is the
"confirm scoring quality" half of Phase-5 item 3 — we EVALUATE FinBERT rather than fine-tune it (the
pretrained ProsusAI/finbert already targets this exact task; fine-tuning risks overfitting a 3-class
3.5k-sentence set and is not needed to clear the quality bar).

Routes through the SAME ``sentiment.score_texts`` the pipeline uses, so this measures the production
scorer, not a throwaway. Reports accuracy + macro-F1 + per-class precision/recall + confusion matrix.

    python scripts/gate5_finbert_phrasebank_eval.py                 # default: 75%-agreement split
    python scripts/gate5_finbert_phrasebank_eval.py --agree AllAgree --limit 1000
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_DIR = Path(__file__).resolve().parents[1] / "Dataset" / "financial_phrasebank"
_LABELS = ["negative", "neutral", "positive"]


def _load(agree: str) -> pd.DataFrame:
    fp = _DIR / f"Sentences_{agree}.csv"
    if not fp.exists():
        alt = sorted(_DIR.glob("Sentences_*.csv"))
        raise SystemExit(f"{fp.name} not found. Available: {[p.name for p in alt]}")
    df = pd.read_csv(fp)
    df["label"] = df["label"].str.strip().str.lower()
    return df[df["label"].isin(_LABELS)].reset_index(drop=True)


def _confusion(gold: list[str], pred: list[str]) -> pd.DataFrame:
    idx = {l: i for i, l in enumerate(_LABELS)}
    m = np.zeros((3, 3), int)
    for g, p in zip(gold, pred):
        m[idx[g], idx.get(p, 1)] += 1
    return pd.DataFrame(
        m, index=[f"true_{l}" for l in _LABELS], columns=[f"pred_{l}" for l in _LABELS]
    )


def _metrics(cm: pd.DataFrame) -> tuple[float, float, dict]:
    m = cm.to_numpy()
    acc = m.trace() / m.sum()
    per, f1s = {}, []
    for i, l in enumerate(_LABELS):
        tp = m[i, i]
        prec = tp / m[:, i].sum() if m[:, i].sum() else 0.0
        rec = tp / m[i, :].sum() if m[i, :].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[l] = {"precision": prec, "recall": rec, "f1": f1, "support": int(m[i, :].sum())}
        f1s.append(f1)
    return float(acc), float(np.mean(f1s)), per


def main() -> int:
    agree = "75Agree"
    limit = None
    args = sys.argv[1:]
    if "--agree" in args:
        agree = args[args.index("--agree") + 1]
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])

    df = _load(agree)
    if limit:
        df = df.sample(min(limit, len(df)), random_state=42).reset_index(drop=True)
    print(
        f"Financial PhraseBank ({agree}): {len(df):,} sentences "
        f"({df['label'].value_counts().to_dict()})"
    )
    print("Scoring with FinBERT via sentiment.score_texts (equity model) ...")

    from backend.prediction.sentiment import score_texts

    texts = df["sentence"].astype(str).tolist()
    pred = []
    B = 256
    for i in range(0, len(texts), B):
        pred.extend(lbl for lbl, _ in score_texts(texts[i : i + B], asset_type="equity"))
        print(f"  scored {min(i + B, len(texts)):,}/{len(texts):,}", end="\r")
    print()

    cm = _confusion(df["label"].tolist(), pred)
    acc, macro_f1, per = _metrics(cm)
    print("\n" + "=" * 64)
    print(f"  FinBERT on PhraseBank-{agree}:  accuracy {acc:.4f}   macro-F1 {macro_f1:.4f}")
    print("=" * 64)
    print(f"    {'class':10}{'precision':>11}{'recall':>9}{'f1':>9}{'support':>9}")
    for l in _LABELS:
        p = per[l]
        print(
            f"    {l:10}{p['precision']:>11.3f}{p['recall']:>9.3f}{p['f1']:>9.3f}{p['support']:>9}"
        )
    print("\n  confusion matrix:")
    print("    " + cm.to_string().replace("\n", "\n    "))
    # Quality bar: FinBERT is well-documented at ~0.85+ accuracy on this set; flag if far below.
    ok = acc >= 0.80
    note = "" if ok else "  (below 0.80 - check model/cache/label mapping)"
    print(
        f"\n  GATE-5 (item 3) scoring-quality: accuracy {acc:.4f} "
        f"{'>=' if ok else '<'} 0.80 -> {'PASS' if ok else 'REVIEW'}{note}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
