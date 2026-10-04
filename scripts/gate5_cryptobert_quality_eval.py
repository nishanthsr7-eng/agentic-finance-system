"""
PHASE-5 — CryptoBERT scoring-quality check (the crypto-side equivalent of the FinBERT PhraseBank eval).

There is no staged labeled crypto-sentiment corpus, so this uses a small CURATED set of unambiguous
crypto SOCIAL-MEDIA posts with known polarity (bullish / bearish / neutral) and confirms CryptoBERT —
routed through the SAME production ``sentiment.score_texts(asset_type='crypto')`` path — classifies
them correctly.

KEY FINDING (why social-media style, not headlines): CryptoBERT (ElKulako/cryptobert) is fine-tuned on
StockTwits/Reddit crypto SOCIAL posts. Measured here it nails social directional polarity (~0.9) but
reads formal NEWS HEADLINES ("Bitcoin smashes through resistance...") as Neutral — on a news-headline
set it scored directional 0.50, FinBERT-on-PhraseBank 0.94. So: CryptoBERT belongs on social/Reddit
text (where ``refresh_reddit`` routes it); FinBERT is the better scorer for crypto *news headlines*.
"neutral" is CryptoBERT's hard class regardless, so we gate on DIRECTIONAL accuracy and report neutral.

    python scripts/gate5_cryptobert_quality_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

# (post, gold_label) — curated, unambiguous crypto SOCIAL-MEDIA posts (CryptoBERT's training domain).
_SET: list[tuple[str, str]] = [
    # bullish (social slang)
    ("btc to the moon, lfg, massive pump incoming", "positive"),
    ("buying more eth, diamond hands, wagmi", "positive"),
    ("this alt is gonna explode, bullish af, send it", "positive"),
    ("hodl strong, we are all gonna make it, green days ahead", "positive"),
    ("sol ripping higher, bought the dip, easy money", "positive"),
    ("ada looking ready to breakout, loading my bags, so bullish", "positive"),
    ("doge pumping hard, lets go, to the moon", "positive"),
    ("link finally mooning, bullish, time to accumulate", "positive"),
    # bearish (social slang)
    ("rekt again, btc dumping, ngmi", "negative"),
    ("sold everything, this is a scam, super bearish", "negative"),
    ("eth tanking, panic, get out now", "negative"),
    ("my bags are bleeding, total rug, cope", "negative"),
    ("market crashing, got liquidated, pure pain", "negative"),
    ("this coin is dead, dumping to zero, bearish", "negative"),
    ("whales selling, capitulation, run for the exit", "negative"),
    ("another hack, funds gone, bearish as hell", "negative"),
    # neutral (factual)
    ("eth merge scheduled for next week", "neutral"),
    ("wallet app update released today", "neutral"),
    ("conference panel on scaling solutions tomorrow", "neutral"),
    ("exchange maintenance window announced", "neutral"),
    ("new docs published for the protocol", "neutral"),
    ("staking dashboard got a minor ui tweak", "neutral"),
    ("token migration planned for next month", "neutral"),
    ("weekly trading volume was about average", "neutral"),
]
_LABELS = ["negative", "neutral", "positive"]


def main() -> int:
    from backend.prediction.sentiment import _MODELS, _load_pipe, score_texts

    sentences = [s for s, _ in _SET]
    gold = [g for _, g in _SET]
    # Confirm the crypto model actually loaded (not the FinBERT fallback) before trusting the result.
    pipe = _load_pipe("crypto")
    model_name = getattr(getattr(pipe, "model", None), "name_or_path", "?")
    used_crypto = _MODELS["crypto"] in str(model_name)
    print(
        f"crypto pipeline model: {model_name}  ({'CryptoBERT' if used_crypto else 'FALLBACK (FinBERT)'})"
    )
    if not used_crypto:
        print(
            "  NOTE: CryptoBERT is not cached -> ran the FinBERT fallback. Pull the model and re-run\n"
            "        to evaluate CryptoBERT itself (this still proves the fallback path works)."
        )

    pred = [lbl for lbl, _ in score_texts(sentences, asset_type="crypto")]

    idx = {l: i for i, l in enumerate(_LABELS)}
    cm = np.zeros((3, 3), int)
    for g, p in zip(gold, pred):
        cm[idx[g], idx.get(p, 1)] += 1
    acc = cm.trace() / cm.sum()
    # Directional accuracy = correct on the bullish/bearish items (the classes that drive the tilt).
    dir_total = sum(1 for g in gold if g != "neutral")
    dir_correct = sum(1 for g, p in zip(gold, pred) if g != "neutral" and g == p)
    dir_acc = dir_correct / dir_total

    print("\n" + "=" * 60)
    print(f"  CryptoBERT curated eval:  overall acc {acc:.3f}   directional acc {dir_acc:.3f}")
    print("=" * 60)
    print(f"    {'':10}" + "".join(f"pred_{l[:3]:>8}" for l in _LABELS))
    for i, l in enumerate(_LABELS):
        print(f"    true_{l[:5]:<6}" + "".join(f"{cm[i, j]:>12}" for j in range(3)))
    for s, g, p in zip(sentences, gold, pred):
        flag = "  " if g == p else "XX"
        print(f"   {flag} [{g:>8} -> {p:>8}] {s[:60]}")

    ok = dir_acc >= 0.85  # gate on directional; neutral is CryptoBERT's hard class
    print(
        f"\n  CryptoBERT scoring-quality (social domain): directional {dir_acc:.3f} (>=0.85), "
        f"overall {acc:.3f} -> {'PASS' if ok else 'REVIEW'}"
    )
    print("  (CryptoBERT is social-media-tuned; use FinBERT for crypto NEWS HEADLINES.)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
