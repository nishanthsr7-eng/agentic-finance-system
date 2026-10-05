"""
Tests for Phase-5 multi-source sentiment: the asset-routed scorer (sentiment.py) and the trainable
historical feature block (sentiment_features.py).

Fast, deterministic, NETWORK-FREE and MODEL-FREE: model routing is exercised with fake pipelines
injected into the per-asset cache (no FinBERT/CryptoBERT download), and the feature block runs on a
synthetic in-memory GDELT panel (no API, no DB). The decisive checks are routing correctness +
label-vocabulary normalisation (FinBERT positive/negative vs CryptoBERT bullish/bearish) and the
leak-safety truncation-invariance of the feature block.

Run:   pytest backend/prediction/tests/test_sentiment_features.py -q
Or:    python -m backend.prediction.test_sentiment_features
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.prediction import sentiment as S
from backend.prediction import sentiment_features as SF
from backend.prediction.sentiment_features import SENTIMENT_FEATURE_COLS, compute_sentiment_features


# ── Fake pipelines (callable like a transformers pipeline) ─────────────────────────
def _fake_pipe(label_scores):
    """Return a callable that emits the same [{label,score},...] for every input text."""

    def _call(texts, **kw):
        return [[{"label": l, "score": s} for l, s in label_scores] for _ in texts]

    return _call


def _install_fakes():
    """Inject fake FinBERT (positive) and CryptoBERT (bearish) pipelines into the per-asset cache."""
    S._pipes.clear()
    S._pipes["equity"] = _fake_pipe([("positive", 0.90), ("negative", 0.05), ("neutral", 0.05)])
    S._pipes["crypto"] = _fake_pipe([("Bearish", 0.90), ("Bullish", 0.05), ("Neutral", 0.05)])


# ── 1. Label normalisation handles BOTH model vocabularies ─────────────────────────
def _check_label_normalisation():
    lbl, sc = S._normalise(
        [
            {"label": "positive", "score": 0.8},
            {"label": "negative", "score": 0.1},
            {"label": "neutral", "score": 0.1},
        ]
    )
    assert lbl == "positive" and abs(sc - 0.7) < 1e-9, (lbl, sc)
    # CryptoBERT vocabulary (Bullish/Bearish) maps onto the same signed convention.
    lbl, sc = S._normalise(
        [
            {"label": "Bearish", "score": 0.7},
            {"label": "Bullish", "score": 0.2},
            {"label": "Neutral", "score": 0.1},
        ]
    )
    assert lbl == "negative" and abs(sc - (-0.5)) < 1e-9, (lbl, sc)
    print(
        "  [1] label-normalisation     OK   (FinBERT pos/neg & CryptoBERT bull/bear -> signed [-1,1])"
    )


# ── 2. asset_type routing: equity->FinBERT, crypto->CryptoBERT ─────────────────────
def _check_routing():
    assert S._asset_type("AAPL") == "equity" and S._asset_type("BTC") == "crypto"
    _install_fakes()
    eq = S.score_texts(["x"], asset_type="equity")[0]
    cr = S.score_texts(["x"], asset_type="crypto")[0]
    assert eq[0] == "positive" and eq[1] > 0, eq
    assert cr[0] == "negative" and cr[1] < 0, cr  # proves the crypto model (not FinBERT) ran
    print("  [2] asset_type-routing      OK   (equity->FinBERT +, crypto->CryptoBERT -)")


# ── 3. score_texts_routed groups a mixed batch by symbol and preserves order ───────
def _check_routed_batch():
    _install_fakes()
    out = S.score_texts_routed([("a", "AAPL"), ("b", "BTC"), ("c", "NVDA")])
    assert [o[0] for o in out] == ["positive", "negative", "positive"], out
    print("  [3] routed-batch            OK   (per-symbol model routing, order preserved)")


# ── 4. CryptoBERT fallback: if the crypto model can't load, degrade to FinBERT ─────
def _check_fallback(monkeypatch=None):
    import types

    S._pipes.clear()
    fin = _fake_pipe([("positive", 0.9), ("negative", 0.05), ("neutral", 0.05)])

    def fake_pipeline(task, model=None, **kw):
        if model == S._MODELS["crypto"]:
            raise OSError("crypto model not cached / offline")
        return fin

    fake_tf = types.SimpleNamespace(pipeline=fake_pipeline)
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    saved = {k: sys.modules.get(k) for k in ("transformers", "torch")}
    sys.modules["transformers"], sys.modules["torch"] = fake_tf, fake_torch
    try:
        pipe = S._load_pipe("crypto")  # must fall back, not raise
        assert pipe is fin, "crypto load failure did not fall back to FinBERT"
    finally:
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v
            else:
                sys.modules.pop(k, None)
        S._pipes.clear()
    print("  [4] cryptobert-fallback     OK   (missing crypto model -> FinBERT, no crash)")


# ── Feature block: synthetic GDELT panel (no network/DB) ───────────────────────────
def _synthetic_gdelt(sym="TEST", n=400):
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    tone = np.cumsum(rng.normal(0, 0.4, n)).clip(-15, 15)  # autocorrelated tone walk
    vol = rng.integers(5, 200, n)
    return pd.DataFrame({"symbol": sym, "date": dates, "tone": tone, "vol": vol})


def _price_frame(n=400):
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame({"close": np.linspace(100, 120, n)}, index=idx)


def _patch_panels(monkeypatch, gdelt, av=None):
    SF.clear_cache()
    monkeypatch.setattr(SF, "_gdelt_panel", lambda: gdelt)
    monkeypatch.setattr(SF, "_av_panel", lambda: av if av is not None else pd.DataFrame())


# ── 5. Neutral when there is no coverage (all-zero, so dropna spares the row) ───────
def _check_neutral(monkeypatch):
    _patch_panels(monkeypatch, pd.DataFrame())  # empty panels
    sf = compute_sentiment_features("AAPL", _price_frame())
    assert list(sf.columns) == SENTIMENT_FEATURE_COLS and sf.index.equals(_price_frame().index)
    assert (sf.to_numpy() == 0.0).all(), "no-coverage block must be all-zero (neutral)"
    print("  [5] neutral-no-coverage     OK   (empty streams -> all-zero, finite, index-aligned)")


# ── 6. Populated + finite on a covered symbol; neutral for an uncovered one ─────────
def _check_populated(monkeypatch):
    g = _synthetic_gdelt("TEST")
    _patch_panels(monkeypatch, g)
    price = _price_frame()
    sf = compute_sentiment_features("TEST", price)
    assert np.isfinite(sf.to_numpy()).all(), "non-finite sentiment feature"
    active = int((sf.abs() > 1e-9).any().sum())
    assert active >= 3, f"covered symbol should populate tone cols, got {active}"
    other = compute_sentiment_features("AAPL", price)  # not in the synthetic panel
    assert (other.to_numpy() == 0.0).all(), "uncovered symbol must stay neutral"
    print(
        f"  [6] populated-and-finite    OK   (TEST {active}/5 cols active; uncovered symbol neutral)"
    )


# ── 7. Leak-safety: truncating future price rows can't move earlier feature values ──
def _check_leak_safe(monkeypatch):
    g = _synthetic_gdelt("TEST")
    _patch_panels(monkeypatch, g)
    price = _price_frame()
    full = compute_sentiment_features("TEST", price)
    cut = int(len(price) * 0.6)
    trunc = compute_sentiment_features("TEST", price.iloc[:cut])
    d = float(np.max(np.abs(full.iloc[:cut].to_numpy() - trunc.to_numpy())))
    assert d < 1e-9, f"feature changed when future price removed (d={d:.2e}) -> LEAK"
    print(f"  [7] leak-safe               OK   (truncation-invariant, d={d:.1e})")


# ── pytest entry points ────────────────────────────────────────────────────────────
def test_label_normalisation():
    _check_label_normalisation()


def test_routing():
    _check_routing()


def test_routed_batch():
    _check_routed_batch()


def test_fallback():
    _check_fallback()


def test_neutral(monkeypatch):
    _check_neutral(monkeypatch)


def test_populated(monkeypatch):
    _check_populated(monkeypatch)


def test_leak_safe(monkeypatch):
    _check_leak_safe(monkeypatch)


if __name__ == "__main__":
    print("Running Phase-5 sentiment tests:")
    _check_label_normalisation()
    _check_routing()
    _check_routed_batch()
    _check_fallback()
    # the monkeypatch-based ones via a tiny shim for direct runs
    import contextlib

    class _MP:
        def __init__(self):
            self._undo = []

        def setattr(self, obj, name, val):
            self._undo.append((obj, name, getattr(obj, name)))
            setattr(obj, name, val)

        def undo(self):
            for obj, name, val in reversed(self._undo):
                setattr(obj, name, val)
            self._undo.clear()

    mp = _MP()
    for fn in (_check_neutral, _check_populated, _check_leak_safe):
        with contextlib.suppress(Exception):
            pass
        fn(mp)
        mp.undo()
    print("All sentiment tests passed.")
