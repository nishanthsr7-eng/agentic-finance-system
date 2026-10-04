"""
FLUX Prediction — News Sentiment (Layer 2c)
===========================================
Scores financial news and aggregates a per-symbol sentiment signal in [-1, +1].
This is a REAL-TIME signal fused at inference/agent time (news history is too short
to train on), orthogonal to the price/macro model.

Two interchangeable backends, chosen by SENTIMENT_BACKEND:

  * "transformers" — FinBERT/CryptoBERT locally. Calibrated probabilities, and
    the better signal, but it needs torch (~445 MB resident).
  * "llm"          — the configured chat model (Groq et al) scores the headline.
    Coarser, but it runs where torch does not fit.
  * "auto"         — transformers when importable, else llm, else no-op.

Sources (all free, keys already in .env):
  • news_cache         — general finance headlines already ingested (NewsAPI)
  • Finnhub company-news — per-ticker articles (better symbol-specific signal)

Public API:
    score_texts(texts)            -> [(label, signed_score), ...]   (transformers only)
    await ascore_texts(texts)     -> [(label, signed_score), ...]   (either backend)
    await score_news_cache()      -> int (rows scored into news_sentiment)
    await refresh_symbol(symbol)  -> fetch+score Finnhub company-news for one ticker
    await symbol_sentiment(sym)   -> {'score','label','n','as_of'} aggregate
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

import httpx

from ..config import settings
from ..db import get_symbol_sentiment, get_unscored_news, insert_sentiment
from . import sentiment_llm
from .datasources import CRYPTO_SYMBOLS

log = logging.getLogger("flux.prediction.sentiment")

# Phase-5: asset-class-routed sentiment models. FinBERT reads equity/financial English; CryptoBERT is
# fine-tuned on crypto social/news ("Bullish/Neutral/Bearish"). We route by asset_type so each headline
# is scored by the model that understands its domain (equity -> FinBERT, crypto -> CryptoBERT).
_MODELS = {"equity": "ProsusAI/finbert", "crypto": "ElKulako/cryptobert"}
_pipes: dict[str, object] = {}  # asset_type -> loaded pipeline (lazy, cached)
_HAS_TRANSFORMERS: bool | None = None  # resolved once by _transformers_available()

# Label-vocabulary normalisation: FinBERT emits positive/negative/neutral; CryptoBERT bullish/bearish/
# neutral. Map both onto a signed score = P(bullish-ish) - P(bearish-ish) in [-1, 1].
_POS_LABELS = {"positive", "bullish", "pos"}
_NEG_LABELS = {"negative", "bearish", "neg"}

# Ticker → keywords for mapping a headline to a symbol.
from ..ingestion import STOCK_META  # noqa: E402

_SYMBOL_KEYWORDS = {
    sym: [sym.lower(), name.lower().split()[0].lower()]
    for sym, (name, _sector) in STOCK_META.items()
}


def _asset_type(symbol: str) -> str:
    """'crypto' if the symbol is in the crypto universe, else 'equity'."""
    return "crypto" if (symbol or "").upper() in CRYPTO_SYMBOLS else "equity"


def _transformers_available() -> bool:
    """True if the FinBERT path can actually run. Cached — importlib is not free."""
    global _HAS_TRANSFORMERS
    if _HAS_TRANSFORMERS is None:
        from importlib.util import find_spec

        _HAS_TRANSFORMERS = bool(find_spec("torch") and find_spec("transformers"))
    return _HAS_TRANSFORMERS


def backend() -> str:
    """Which scorer is live: "transformers", "llm" or "none"."""
    choice = (settings.SENTIMENT_BACKEND or "auto").strip().lower()
    if choice == "transformers":
        return "transformers" if _transformers_available() else "none"
    if choice == "llm":
        return "llm" if sentiment_llm.available() else "none"
    # auto — prefer the calibrated model, fall back to the one that fits.
    if _transformers_available():
        return "transformers"
    if sentiment_llm.available():
        return "llm"
    return "none"


def _load_pipe(asset_type: str = "equity"):
    """Lazily load (and cache) the pipeline for an asset class. If the crypto model can't be pulled
    (offline / not cached), fall back to FinBERT so scoring degrades gracefully instead of crashing."""
    asset_type = asset_type if asset_type in _MODELS else "equity"
    if asset_type in _pipes:
        return _pipes[asset_type]
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    name = _MODELS[asset_type]
    try:
        pipe = pipeline("text-classification", model=name, top_k=None, device=device)
        log.info("%s loaded on %s", name, "GPU" if device == 0 else "CPU")
    except Exception as exc:  # model missing/offline -> degrade
        if asset_type != "equity":
            log.warning(
                "%s unavailable (%s) - falling back to FinBERT for %s sentiment",
                name,
                exc,
                asset_type,
            )
            pipe = _load_pipe("equity")
        else:
            raise
    _pipes[asset_type] = pipe
    return pipe


def _normalise(scores: list[dict]) -> tuple[str, float]:
    """One pipeline output (list of {label,score}) -> (normalised_label, signed_score in [-1,1])."""
    d = {s["label"].lower(): float(s["score"]) for s in scores}
    signed = sum(v for k, v in d.items() if k in _POS_LABELS) - sum(
        v for k, v in d.items() if k in _NEG_LABELS
    )
    raw = max(d, key=d.get)
    label = "positive" if raw in _POS_LABELS else "negative" if raw in _NEG_LABELS else "neutral"
    return label, round(signed, 4)


def score_texts(texts: list[str], asset_type: str = "equity") -> list[tuple[str, float]]:
    """Return (label, signed_score) per text, scored by the asset-class model (FinBERT/CryptoBERT).

    signed_score = P(positive/bullish) - P(negative/bearish) in [-1, 1]; label is normalised to
    positive/negative/neutral regardless of which model's vocabulary produced it.
    """
    if not texts:
        return []
    pipe = _load_pipe(asset_type)
    return [_normalise(scores) for scores in pipe(texts, truncation=True, max_length=256)]


def score_texts_routed(items: list[tuple[str, str]]) -> list[tuple[str, float]]:
    """Score [(text, symbol), ...] routing each text to its asset-class model; preserves input order."""
    if not items:
        return []
    out: list[tuple[str, float] | None] = [None] * len(items)
    groups: dict[str, list[int]] = {}
    for i, (_txt, sym) in enumerate(items):
        groups.setdefault(_asset_type(sym), []).append(i)
    for atype, idxs in groups.items():
        scored = score_texts([items[i][0] for i in idxs], asset_type=atype)
        for i, sc in zip(idxs, scored):
            out[i] = sc
    return [o if o is not None else ("neutral", 0.0) for o in out]


async def ascore_texts(texts: list[str], asset_type: str = "equity") -> list[tuple[str, float]]:
    """
    Score texts with whichever backend is live.

    The transformers pipeline is blocking and slow enough to stall the event
    loop for seconds on a batch, so it runs in a worker thread; the LLM path is
    already async. Returns all-neutral rather than raising when no backend is
    configured — sentiment is one input to the agent, not a hard dependency.
    """
    if not texts:
        return []
    b = backend()
    if b == "transformers":
        import asyncio

        return await asyncio.to_thread(score_texts, texts, asset_type)
    if b == "llm":
        return await sentiment_llm.score_texts_llm(texts, asset_type)
    log.debug("No sentiment backend configured — %d texts scored neutral", len(texts))
    return [("neutral", 0.0)] * len(texts)


async def ascore_texts_routed(items: list[tuple[str, str]]) -> list[tuple[str, float]]:
    """Async twin of score_texts_routed: route each text to its asset-class scorer."""
    if not items:
        return []
    out: list[tuple[str, float] | None] = [None] * len(items)
    groups: dict[str, list[int]] = {}
    for i, (_txt, sym) in enumerate(items):
        groups.setdefault(_asset_type(sym), []).append(i)
    for atype, idxs in groups.items():
        scored = await ascore_texts([items[i][0] for i in idxs], asset_type=atype)
        for i, sc in zip(idxs, scored):
            out[i] = sc
    return [o if o is not None else ("neutral", 0.0) for o in out]


def _map_symbol(text: str) -> str:
    """Best-effort map a headline to a ticker; '' = general market."""
    t = text.lower()
    for sym, kws in _SYMBOL_KEYWORDS.items():
        if any(kw in t for kw in kws):
            return sym
    return ""


async def score_news_cache(limit: int = 200) -> int:
    """Score not-yet-scored news_cache rows into news_sentiment."""
    rows = await get_unscored_news(limit)
    if not rows:
        return 0
    texts = [f"{r['title']} {r.get('summary', '')}".strip() for r in rows]
    syms = [_map_symbol(t) for t in texts]  # route each headline to its asset model
    scored = await ascore_texts_routed(list(zip(texts, syms)))
    ts = int(time.time() * 1000)
    out = []
    for r, sym, (label, signed) in zip(rows, syms, scored):
        out.append(
            {
                "url": r["url"],
                "symbol": sym,
                "source": "newsapi",
                "title": r["title"],
                "score": signed,
                "label": label,
                "published_at": r.get("published_at", ""),
                "scored_at": ts,
            }
        )
    await insert_sentiment(out)
    log.info("Scored %d news_cache articles", len(out))
    return len(out)


async def refresh_symbol(symbol: str, days: int = 14) -> int:
    """Fetch Finnhub company-news for `symbol`, score it, store in news_sentiment."""
    if not settings.FINNHUB_API_KEY:
        return 0
    to = datetime.utcnow().date()
    frm = to - timedelta(days=days)
    try:
        async with httpx.AsyncClient(timeout=15.0) as cli:
            r = await cli.get(
                "https://finnhub.io/api/v1/company-news",
                params={
                    "symbol": symbol.upper(),
                    "from": str(frm),
                    "to": str(to),
                    "token": settings.FINNHUB_API_KEY,
                },
            )
            r.raise_for_status()
            articles = r.json()[:40]
    except Exception as exc:
        log.warning("Finnhub company-news %s: %s", symbol, exc)
        return 0
    if not articles:
        return 0

    texts = [f"{a.get('headline', '')} {a.get('summary', '')}".strip() for a in articles]
    scored = await ascore_texts(texts, asset_type=_asset_type(symbol))
    ts = int(time.time() * 1000)
    out = []
    for a, (label, signed) in zip(articles, scored):
        url = a.get("url") or f"finnhub:{a.get('id')}"
        out.append(
            {
                "url": url,
                "symbol": symbol.upper(),
                "source": "finnhub",
                "title": a.get("headline", ""),
                "score": signed,
                "label": label,
                "published_at": datetime.utcfromtimestamp(a.get("datetime", 0)).isoformat(),
                "scored_at": ts,
            }
        )
    await insert_sentiment(out)
    log.info("Scored %d Finnhub articles for %s", len(out), symbol)
    return len(out)


async def refresh_reddit(symbol: str, limit: int = 40) -> int:
    """
    Retail-flow sentiment: pull recent r/wallstreetbets + r/stocks posts mentioning `symbol`,
    score the titles with the active sentiment backend, and store under
    source='reddit' in news_sentiment.

    No-op (returns 0) unless BOTH Reddit client id AND secret are configured — so on this
    machine (empty secret) it stays a clean no-op. praw is read-only here; symbols are
    validated and the post count is bounded.
    """
    import re

    if not (settings.REDDIT_CLIENT_ID and settings.REDDIT_CLIENT_SECRET):
        log.debug("Reddit disabled (no client id/secret)")
        return 0
    if not re.match(r"^[A-Z0-9.\-]{1,12}$", symbol.upper()):
        return 0

    def _pull() -> list[str]:  # blocking praw → run in a thread
        import praw

        reddit = praw.Reddit(
            client_id=settings.REDDIT_CLIENT_ID,
            client_secret=settings.REDDIT_CLIENT_SECRET,
            user_agent=settings.REDDIT_USER_AGENT,
            check_for_async=False,
        )
        reddit.read_only = True
        titles: list[str] = []
        for sub in ("wallstreetbets", "stocks"):
            for post in reddit.subreddit(sub).search(symbol.upper(), sort="new", limit=limit // 2):
                if post.title:
                    titles.append(post.title)
        return titles

    try:
        import asyncio

        titles = await asyncio.to_thread(_pull)
    except Exception as exc:
        log.warning("Reddit fetch for %s failed: %s", symbol, exc)
        return 0
    if not titles:
        return 0

    scored = await ascore_texts(titles, asset_type=_asset_type(symbol))
    ts = int(time.time() * 1000)
    out = [
        {
            "url": f"reddit:{symbol.upper()}:{ts}:{i}",
            "symbol": symbol.upper(),
            "source": "reddit",
            "title": t,
            "score": signed,
            "label": label,
            "published_at": "",
            "scored_at": ts,
        }
        for i, (t, (label, signed)) in enumerate(zip(titles, scored))
    ]
    await insert_sentiment(out)
    log.info("Scored %d Reddit posts for %s", len(out), symbol)
    return len(out)


async def symbol_sentiment(symbol: str, days: int = 7) -> dict:
    """
    Aggregate recent sentiment for a symbol into one signal.
    Recency-weighted mean of signed scores; symbol-specific articles weighted 2x over
    general-market ones.
    """
    rows = await get_symbol_sentiment(symbol, days)
    if not rows:
        return {"score": 0.0, "label": "neutral", "n": 0, "as_of": None}

    now = time.time() * 1000
    num = den = 0.0
    for r in rows:
        age_days = max(0.0, (now - r["scored_at"]) / 86_400_000)
        recency = 0.5 ** (age_days / 3.0)  # 3-day half-life
        w = recency * (2.0 if r["symbol"] == symbol.upper() else 1.0)
        num += w * r["score"]
        den += w
    score = round(num / den, 4) if den else 0.0
    label = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"
    return {
        "score": score,
        "label": label,
        "n": len(rows),
        "as_of": max(r["scored_at"] for r in rows),
    }


if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    async def _demo():
        from backend.db import init_db

        await init_db()  # ensure news_sentiment table exists
        # Quick correctness check on known-polarity finance sentences.
        tests = [
            "Company beats earnings expectations and raises full-year guidance",
            "Shares plunge after the firm slashes outlook and warns of layoffs",
            "The board will meet next Tuesday to review the quarterly report",
        ]
        print(f"Sentiment backend: {backend()}")
        for txt, (lab, sc) in zip(tests, await ascore_texts(tests)):
            print(f"  [{lab:<8} {sc:+.3f}] {txt}")
        n = await score_news_cache()
        print(f"\nScored {n} cached news articles.")

    asyncio.run(_demo())
