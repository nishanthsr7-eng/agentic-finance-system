"""
FLUX Prediction — News Sentiment via an LLM (torch-free backend for Layer 2c)
============================================================================
A drop-in replacement for the FinBERT/CryptoBERT scorer in ``sentiment.py``
that asks the configured chat model to classify headlines instead.

Why this exists: FinBERT needs torch, and torch is ~445 MB resident — five
times the rest of the backend put together. That fits nowhere free. The
hosted LLM is already configured for chat and insights, so reusing it for
sentiment keeps Layer 2c alive on a 512 MB host.

The two backends are not identical and the difference is worth stating:

  * FinBERT returns calibrated per-class probabilities, so the signed score is
    a genuine P(pos) - P(neg) and sits anywhere in [-1, 1].
  * An LLM returns a judgement. We ask for a score directly, which gives a
    coarser, more clustered distribution — models favour round numbers.

Downstream this only feeds an aggregate mean, which is robust to that. But
the model is doing something different, not the same thing more cheaply.

Public API mirrors the transformers path so the caller can swap freely:

    await score_texts_llm(texts, asset_type)      -> [(label, signed), ...]
"""

from __future__ import annotations

import json
import logging

from .. import llm

log = logging.getLogger("flux.prediction.sentiment_llm")

# Headlines per request. Large enough that a scoring pass is a handful of calls
# rather than hundreds, small enough that the model reliably returns one entry
# per input and the response stays inside the token budget.
BATCH = 20

_SYSTEM = (
    "You are a financial news sentiment classifier. You score headlines by their "
    "likely effect on the price of the asset they concern, not by whether the news "
    "is pleasant. A profit warning is negative; an earnings beat is positive; a "
    "routine announcement is neutral. You reply with JSON and nothing else."
)

_PROMPT = """Score each headline from -1.0 (strongly bearish) to 1.0 (strongly bullish).
Use 0.0 for routine or genuinely ambiguous news. These are {domain} headlines.

Reply with a JSON array of objects, one per headline, in the same order:
[{{"i": 0, "score": -0.6}}, {{"i": 1, "score": 0.2}}]

Return exactly {n} objects. No prose, no code fences.

Headlines:
{items}"""

_DOMAIN = {"crypto": "cryptocurrency", "equity": "stock market"}


def _label(signed: float) -> str:
    """Bucket a signed score the way _normalise() does for the transformers path."""
    if signed >= 0.15:
        return "positive"
    if signed <= -0.15:
        return "negative"
    return "neutral"


def _parse(raw: str, n: int) -> list[tuple[str, float]]:
    """
    Parse the model's reply into n (label, signed) pairs.

    Anything the model got wrong — short array, missing index, score out of
    range, prose instead of JSON — degrades to neutral for the affected items
    rather than failing the batch. A sentiment signal that is occasionally
    neutral is recoverable; one that raises takes the ingestion job down with it.
    """
    out: list[tuple[str, float]] = [("neutral", 0.0)] * n
    text = llm.strip_fences(raw).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        log.warning("LLM sentiment: no JSON array in reply (%.80s)", text)
        return out
    try:
        items = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        log.warning("LLM sentiment: unparseable JSON (%s)", exc)
        return out
    if not isinstance(items, list):
        return out

    for pos, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        # "i" is what we asked for, but models drift to positional output; fall
        # back to the array position when the key is missing or not an int.
        idx = item.get("i", pos)
        if not isinstance(idx, int) or not 0 <= idx < n:
            idx = pos
            if not 0 <= idx < n:
                continue
        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        score = max(-1.0, min(1.0, score))
        out[idx] = (_label(score), round(score, 4))
    return out


async def score_texts_llm(texts: list[str], asset_type: str = "equity") -> list[tuple[str, float]]:
    """Score headlines with the configured chat model. Never raises."""
    if not texts:
        return []

    domain = _DOMAIN.get(asset_type, _DOMAIN["equity"])
    out: list[tuple[str, float]] = []

    for start in range(0, len(texts), BATCH):
        chunk = texts[start : start + BATCH]
        # Headlines can contain newlines; collapsing them keeps the numbering
        # unambiguous, and truncation keeps one long article summary from
        # crowding out the rest of the batch.
        listing = "\n".join(f"{i}. {' '.join(t.split())[:280]}" for i, t in enumerate(chunk))
        try:
            reply = await llm.chat(
                [
                    {"role": "system", "content": _SYSTEM},
                    {
                        "role": "user",
                        "content": _PROMPT.format(domain=domain, n=len(chunk), items=listing),
                    },
                ],
                timeout=45.0,
                temperature=0.0,  # classification, not generation
            )
            out.extend(_parse(reply, len(chunk)))
        except Exception as exc:
            log.warning("LLM sentiment batch failed (%s) — %d headlines neutral", exc, len(chunk))
            out.extend([("neutral", 0.0)] * len(chunk))

    return out


def available() -> bool:
    """True if a chat model is configured to score with."""
    from ..config import settings

    return bool(settings.LLM_API_KEY) or llm.provider() == "ollama"
