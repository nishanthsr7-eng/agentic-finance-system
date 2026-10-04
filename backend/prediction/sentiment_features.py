"""
FLUX Prediction — Multi-source Sentiment Feature Block (Phase 5)
===============================================================
The TRAINABLE sentiment block: turns the historical news streams (``datasources/gdelt_tone.py`` —
keyless GDELT tone, all symbols; ``datasources/av_news.py`` — Alpha Vantage ticker sentiment, equities)
into a leak-safe per-(symbol, date) feature frame. This is the part of Phase 5 that GATE-5 (a) tests —
"sentiment-augmented OOF AUC >= prior" — and the historical complement to the *live* FinBERT/CryptoBERT
tilt in ``sentiment.py`` + ``predict.py`` (which feeds GATE-5 (b), calibration).

Why a separate historical block: the real-time ``news_sentiment`` table is only days/weeks deep, far too
shallow to enter purged walk-forward OOF training. GDELT (history to 2017, keyless) and AV NEWS_SENTIMENT
(``time_from`` back-fill) give the multi-year daily history a trainable feature needs.

Same contract / honesty discipline as ``crypto_features.py`` and ``equity_features.py``:
  • LEAK-SAFE. A value on row ``D`` is the tone/sentiment of news *published on D*, finalised by end of
    day D; the pipeline reads row D to predict the move starting D+1 (>=1-day lag), so no forward peek.
    Every rolling/shift here is causal (trailing window only). Verified by test_sentiment_features.py
    with a strict truncation-invariance test.
  • NEUTRAL WHEN ABSENT. Any symbol/date without news coverage gets an all-zero (not NaN) block, so the
    ``load_dataset`` dropna() never wipes a row. 0 is the sensible neutral (no news = no tilt). Because
    the streams are best-effort (throttling / rate limits / no key), most cells may be neutral on a
    given machine — that's fine, the block then simply contributes nothing and GATE-5 reports a tie.

Features (all centred so neutral == 0, all causal):
  news_tone      GDELT average tone of the day's coverage, scaled to ~[-1,1] (negative = bad news).
  news_tone_z    causal 30-day z-score of tone — is today's coverage unusually positive/negative?
  news_tone_mom  tone momentum: 7-day mean minus 30-day mean (improving vs deteriorating narrative).
  news_vol_z     causal z-score of article VOLUME — an attention/novelty spike (orthogonal to sign).
  news_av_sent   Alpha Vantage relevance-weighted ticker sentiment (equities; neutral elsewhere).

Public API:
    compute_sentiment_features(symbol, price) -> DataFrame   # aligned to price.index, neutral-filled
    SENTIMENT_FEATURE_COLS                                   # the 5 column names
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import pandas as pd

from .datasources.av_news import load_av_news
from .datasources.gdelt_tone import load_gdelt_tone

log = logging.getLogger("flux.prediction.sentiment_features")

SENTIMENT_FEATURE_COLS = [
    "news_tone",  # GDELT tone level (scaled), step/daily
    "news_tone_z",  # causal 30d z-score of tone
    "news_tone_mom",  # 7d-mean minus 30d-mean tone momentum
    "news_vol_z",  # causal z-score of article volume (attention spike)
    "news_av_sent",  # Alpha Vantage relevance-weighted ticker sentiment (equities)
]

_CLIP = 1.0
_TONE_SCALE = 10.0  # GDELT tone is ~[-10,+10] in practice -> divide to land in ~[-1,1]
_Z_WIN = 30
_Z_MIN = 10
_FFILL_LIMIT = 5  # bridge short gaps causally (row D carries the most recent print on/before D)


@lru_cache(maxsize=1)
def _gdelt_panel() -> pd.DataFrame:
    return load_gdelt_tone()


@lru_cache(maxsize=1)
def _av_panel() -> pd.DataFrame:
    return load_av_news()


def clear_cache() -> None:
    """Drop memoised panels (call after refreshing the GDELT/AV caches in a live process)."""
    _gdelt_panel.cache_clear()
    _av_panel.cache_clear()


def _sym_frame(panel: pd.DataFrame, sym: str) -> pd.DataFrame:
    if panel.empty or "symbol" not in panel:
        return pd.DataFrame()
    d = panel[panel["symbol"] == sym]
    return pd.DataFrame() if d.empty else d.set_index("date").sort_index()


def _z(s: pd.Series, win: int, min_p: int) -> pd.Series:
    mu = s.rolling(win, min_periods=min_p).mean()
    sd = s.rolling(win, min_periods=min_p).std()
    return (s - mu) / sd.replace(0, np.nan)


def _gdelt_feats(g: pd.DataFrame) -> pd.DataFrame:
    """GDELT panel [tone, vol] -> news_tone, news_tone_z, news_tone_mom, news_vol_z (causal)."""
    out = pd.DataFrame(index=g.index)
    tone = pd.to_numeric(g["tone"], errors="coerce")
    out["news_tone"] = (tone / _TONE_SCALE).clip(-_CLIP, _CLIP)
    out["news_tone_z"] = _z(tone, _Z_WIN, _Z_MIN).clip(-4.0, 4.0)
    mom = tone.rolling(7, min_periods=3).mean() - tone.rolling(_Z_WIN, min_periods=_Z_MIN).mean()
    out["news_tone_mom"] = (mom / _TONE_SCALE).clip(-_CLIP, _CLIP)
    if "vol" in g:
        vol = pd.to_numeric(g["vol"], errors="coerce")
        out["news_vol_z"] = _z(np.log1p(vol.where(vol >= 0)), _Z_WIN, _Z_MIN).clip(-4.0, 4.0)
    return out


def compute_sentiment_features(symbol: str, price: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-source sentiment features aligned to ``price.index`` (a per-symbol DatetimeIndex).

    Returns a frame indexed exactly by ``price.index`` with all ``SENTIMENT_FEATURE_COLS``, finite and
    neutral-(0)-filled. Symbols/dates without news coverage get an all-zero block (so the caller's
    dropna() never drops a row on their account).
    """
    out = pd.DataFrame(0.0, index=price.index, columns=SENTIMENT_FEATURE_COLS)
    idx = pd.DatetimeIndex(pd.to_datetime(price.index))
    sym = symbol.upper()

    g = _sym_frame(_gdelt_panel(), sym)
    if not g.empty:
        feats = _gdelt_feats(g).reindex(idx, method="ffill", limit=_FFILL_LIMIT)
        for col in feats.columns:
            out[col] = feats[col].values

    a = _sym_frame(_av_panel(), sym)
    if not a.empty and "av_sent" in a:
        av = pd.to_numeric(a["av_sent"], errors="coerce").clip(-_CLIP, _CLIP)
        out["news_av_sent"] = av.reindex(idx, method="ffill", limit=_FFILL_LIMIT).values

    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    async def _demo():
        from backend.db import get_history, init_db
        from backend.prediction.features import build_features_from_df, calibrate_fd_order

        await init_db()
        gd, av = _gdelt_panel(), _av_panel()
        print(
            f"GDELT panel: {len(gd):,} rows / {gd.symbol.nunique() if not gd.empty else 0} symbols ; "
            f"AV panel: {len(av):,} rows / {av.symbol.nunique() if not av.empty else 0} symbols"
        )
        for sym in ("AAPL", "NVDA", "BTC", "ETH"):
            rows = await get_history(sym)
            if not rows:
                print(f"  {sym:5} no history")
                continue
            df = pd.DataFrame(rows)
            d = calibrate_fd_order(pd.Series(df["adj_close"].astype(float).values[: len(df) // 2]))
            feat = build_features_from_df(df, fd_order=d)
            sf = compute_sentiment_features(sym, feat)
            assert sf.index.equals(feat.index) and np.isfinite(sf.values).all()
            active = [c for c in SENTIMENT_FEATURE_COLS if (sf[c].abs() > 1e-9).any()]
            print(f"  {sym:5} active={len(active)}/{len(SENTIMENT_FEATURE_COLS)} {active}")
        print("OK — neutral (all-zero) where there is no news coverage; finite & index-aligned.")

    asyncio.run(_demo())
