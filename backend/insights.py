"""
FLUX — Pre-computed AI Insights
---------------------------------
Uses Ollama to generate structured market analysis for the top N movers
after each ingestion cycle. Results stored in SQLite + embedded in ChromaDB.

Insight JSON schema (from Ollama):
{
  "sentiment":  "bullish" | "bearish" | "neutral",
  "confidence": <int 0-100>,
  "signal":     "BUY" | "SELL" | "HOLD",
  "summary":    "<one-sentence analysis>",
  "key_level":  "<price level string>",
  "catalyst":   "<main driver string>"
}
"""

import json
import logging
import time

import httpx

from . import llm
from .db import insert_insight

log = logging.getLogger("flux.insights")

_http: httpx.AsyncClient | None = None


def _client() -> httpx.AsyncClient:
    global _http
    if _http is None or _http.is_closed:
        _http = httpx.AsyncClient(timeout=45.0)
    return _http


_PROMPT_TMPL = """\
You are a senior quantitative analyst. Analyse this asset and respond ONLY with valid JSON — no markdown, no code fences, no explanation.

Asset:      {name} ({symbol})
Price:      {price}
24h Change: {change:+.2f}%
Type:       {asset_type}

Confidence scoring guide (use the full range — do NOT default to 75):
  90-95: Very strong trend, high volume confirmation, clear catalyst
  75-89: Moderate trend with supporting signals
  55-74: Mixed signals, moderate conviction
  40-54: Weak or contradictory signals
  25-39: High uncertainty, possible reversal

JSON schema to use (exact keys):
{{
  "sentiment":  "bullish" | "bearish" | "neutral",
  "confidence": <integer 25-95, calibrated using the guide above>,
  "signal":     "BUY" (strong uptrend) | "SELL" (downtrend or overbought) | "HOLD" (unclear or wait),
  "summary":    "<one sentence market analysis, specific to this asset>",
  "key_level":  "<important price level as a number>",
  "catalyst":   "<primary driver of this move, 5-10 words>"
}}"""

_MARKET_SUMMARY_TMPL = """\
You are a senior market strategist. Summarise today's market conditions in 2-3 sentences based on the following data:

Top movers: {movers_str}

Response format: plain text, max 3 sentences, professional tone."""


async def _ollama_call(prompt: str) -> str:
    """
    Single-prompt completion via the configured provider (hosted or Ollama).
    Raises RuntimeError on failure — LLMError subclasses it, so the existing
    callers' error handling is unchanged.
    """
    return await llm.chat([{"role": "user", "content": prompt}], timeout=60.0)


def _strip_fences(text: str) -> str:
    """Remove markdown code fences that models sometimes add."""
    return llm.strip_fences(text)


async def generate_asset_insight(asset: dict) -> dict | None:
    """
    Generate a structured insight for one asset.
    Returns the insight dict (for DB + ChromaDB) or None on failure.
    """
    prompt = _PROMPT_TMPL.format(
        name=asset.get("name", asset["symbol"]),
        symbol=asset["symbol"],
        price=asset.get("price", 0),
        change=float(asset.get("change_pct", 0)),
        asset_type=asset.get("asset_type", "unknown"),
    )
    try:
        raw = await _ollama_call(prompt)
        data = json.loads(_strip_fences(raw))
    except RuntimeError as exc:
        log.warning("Ollama unavailable for %s: %s", asset["symbol"], exc)
        return None
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning("Insight parse failed for %s: %s | raw=%s", asset["symbol"], exc, raw[:120])
        return None

    ts = int(time.time() * 1000)
    sentiment = str(data.get("sentiment", "neutral")).lower()
    confidence = int(data.get("confidence", 0))
    signal = str(data.get("signal", "HOLD")).upper()
    summary = str(data.get("summary", ""))
    key_level = str(data.get("key_level", ""))
    catalyst = str(data.get("catalyst", ""))

    content = (
        f"{asset.get('name', asset['symbol'])} ({asset['symbol']}) — "
        f"LLM commentary (not a model signal): {signal} | {sentiment.capitalize()}, "
        f"LLM conviction {confidence}/100. "
        f"{summary} Key level: {key_level}. Catalyst: {catalyst}."
    )

    return {
        "symbol": asset["symbol"],
        "insight_type": "price_analysis",
        "content": content,
        "sentiment": sentiment,
        "confidence": confidence,
        "generated_at": ts,
        # extra fields for callers (not stored in base DB row)
        "_signal": signal,
        "_key_level": key_level,
        "_catalyst": catalyst,
        "_summary": summary,
    }


async def generate_market_summary(assets: list[dict]) -> str | None:
    """Generate a short natural-language market summary from top movers."""
    top = sorted(assets, key=lambda a: abs(a.get("change_pct", 0)), reverse=True)[:5]
    movers_str = ", ".join(
        f"{a.get('name', a['symbol'])} {a.get('change_pct', 0):+.2f}%" for a in top
    )
    try:
        return await _ollama_call(_MARKET_SUMMARY_TMPL.format(movers_str=movers_str))
    except RuntimeError as exc:
        log.warning("Market summary failed: %s", exc)
        return None


async def run_insight_cycle(
    assets: list[dict],
    max_assets: int = 6,
) -> list[dict]:
    """
    Generate insights for the top N movers.
    Stores each in SQLite and embeds in ChromaDB.
    Returns list of generated insight dicts.
    """
    from .rag import embed_insight  # lazy to avoid circular on startup

    # Pick top movers by absolute 24h change
    ranked = sorted(assets, key=lambda a: abs(a.get("change_pct", 0)), reverse=True)
    targets = ranked[:max_assets]

    generated: list[dict] = []
    for asset in targets:
        insight = await generate_asset_insight(asset)
        if insight is None:
            continue
        db_row = {
            "symbol": insight["symbol"],
            "insight_type": insight["insight_type"],
            "content": insight["content"],
            "sentiment": insight["sentiment"],
            "confidence": insight["confidence"],
            "generated_at": insight["generated_at"],
        }
        await insert_insight(db_row)
        embed_insight(insight)
        generated.append(insight)
        log.info(
            "Insight: %s → %s (%s%% conf)",
            insight["symbol"],
            insight["sentiment"],
            insight["confidence"],
        )

    return generated
