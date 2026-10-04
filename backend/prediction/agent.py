"""
FLUX Prediction — LLM Fusion & Verifier Agent (Layer 3, Phase 7–8)
==================================================================
The ML models predict; the LLM does NOT. Its job is to (a) explain the calibrated signal in
plain language, and (b) RED-TEAM it against fresh news + RAG context — and, when the narrative
contradicts the model, to DOWNGRADE confidence or VETO the trade.

The honesty contract (non-negotiable, enforced in code, not by trusting the LLM):
  • The LLM may LOWER confidence or VETO. It may NEVER raise confidence above the calibrated
    number — `final_confidence = min(model_confidence, llm_confidence)`.
  • A veto forces act=False and size→0, regardless of what the model wanted.
This keeps the calibrated ML probability as the ceiling and the LLM as an auditable safety layer.

Phase 8: `verify_portfolio` focuses the (slower) LLM pass on the TOP-K names of the deployable
long-only book — ranked by the Phase-7 conviction score (edge × meta) — instead of the whole
universe, and the daily flywheel cycle (serve.py) runs it after logging each fresh batch.

Public API:
    await verify_prediction(symbol, prediction=None, persist=True)      -> dict | None
    await verify_portfolio(predictions=None, top_k=5, persist=True)     -> list[dict]

Degrades gracefully: if Ollama is down, returns the model prediction unchanged with
`verifier="unavailable"` so the pipeline never hard-depends on the LLM.
"""

from __future__ import annotations

import json
import logging
import time

from ..db import get_symbol_sentiment, insert_insight
from ..insights import _ollama_call, _strip_fences  # reuse the existing Ollama plumbing
from .predict import ACT_THRESHOLD, predict

log = logging.getLogger("flux.prediction.agent")


_VERIFIER_PROMPT = """\
You are a skeptical risk manager reviewing a QUANT MODEL's trade signal. The model is already
calibrated — your job is NOT to predict price. Your job is to red-team the signal against the
news and context below, then decide whether to TRUST, DOWNGRADE, or VETO it.

Hard rules:
- You may LOWER the confidence or VETO. You may NEVER raise confidence above the model's {model_confidence}.
- VETO (veto=true) only when fresh news genuinely contradicts the model (e.g. a guidance cut,
  fraud, surprise downgrade, regulatory action) — not for vague unease.

MODEL SIGNAL for {symbol}:
- direction: {direction}  (calibrated P(correct) = {model_confidence}%)
- meta-model act gate: {meta_prob:.2f}  (acts when >= {act_threshold})
- {band_pct}% conformal band: [{conf_low}, {conf_high}]  (point target {pred_price})
- market regime (HMM): {regime}
- aggregate news sentiment: {sentiment:+.2f}

RECENT NEWS HEADLINES for {symbol}:
{news_block}

ADDITIONAL CONTEXT (RAG):
{rag_block}

Respond ONLY with valid JSON — no markdown, no code fences:
{{
  "agree":             true | false,
  "confidence":        <integer 0-{model_confidence}>,
  "rationale":         "<2-3 sentence plain-English read of the signal vs the news>",
  "risks":             "<the single biggest risk to this trade>",
  "key_levels":        "<notable price level(s) as a short string>",
  "contradicts_model": true | false,
  "veto":              true | false
}}"""


def _news_block(symbol: str, rows: list[dict], k: int = 5) -> str:
    titles = [r.get("title", "").strip() for r in rows if r.get("title")]
    titles = [t for t in titles][:k]
    return "\n".join(f"- {t}" for t in titles) if titles else "- (no recent headlines)"


async def verify_prediction(
    symbol: str, prediction: dict | None = None, persist: bool = True
) -> dict | None:
    """
    Run the calibrated model prediction through the LLM verifier. Returns the prediction
    augmented with the LLM verdict and the ENFORCED final confidence / act / veto.
    """
    symbol = symbol.upper()
    p = prediction or await predict(symbol)
    if not p:
        return None

    model_conf = int(p["confidence"])

    # Gather context: per-symbol scored headlines + RAG over the knowledge base.
    try:
        sent_rows = await get_symbol_sentiment(symbol)
    except Exception:
        sent_rows = []
    try:
        from ..rag import rag_query

        rag_ctx, _n = rag_query(f"{symbol} stock outlook catalysts risks earnings guidance")
    except Exception:
        rag_ctx = ""

    prompt = _VERIFIER_PROMPT.format(
        symbol=symbol,
        direction=p["direction"],
        model_confidence=model_conf,
        meta_prob=p["meta_prob"],
        act_threshold=ACT_THRESHOLD,
        band_pct=p.get("band_pct", 80),
        conf_low=p.get("conf_low"),
        conf_high=p.get("conf_high"),
        pred_price=p.get("pred_price"),
        regime=p.get("regime", "trend"),
        sentiment=p.get("sentiment", 0.0),
        news_block=_news_block(symbol, sent_rows),
        rag_block=(rag_ctx[:1500] if rag_ctx else "- (no additional context)"),
    )

    # ── Call the LLM; degrade gracefully if it's unavailable ──────────────────────
    verdict = {"verifier": "ok"}
    try:
        raw = await _ollama_call(prompt)
        data = json.loads(_strip_fences(raw))
    except RuntimeError as exc:  # Ollama down / not running
        log.warning("verifier unavailable for %s: %s", symbol, exc)
        return {
            **p,
            "verifier": "unavailable",
            "model_confidence": model_conf,
            "final_confidence": model_conf,
            "final_act": bool(p["act"]),
            "veto": False,
            "rationale": None,
        }
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning("verifier parse failed for %s: %s | raw=%s", symbol, exc, raw[:120])
        return {
            **p,
            "verifier": "parse_error",
            "model_confidence": model_conf,
            "final_confidence": model_conf,
            "final_act": bool(p["act"]),
            "veto": False,
            "rationale": None,
        }

    # ── Enforce the honesty contract (LLM can only lower / veto) ──────────────────
    llm_conf = int(data.get("confidence", model_conf))
    veto = bool(data.get("veto", False)) or bool(data.get("contradicts_model", False))
    final_conf = max(0, min(model_conf, llm_conf))  # never above the calibrated ceiling
    if veto:
        final_conf = min(final_conf, model_conf // 2)  # vetoed signals are heavily downgraded
    final_act = bool(p["act"]) and not veto
    final_kelly = 0.0 if veto else p.get("kelly_frac", 0.0)

    rationale = str(data.get("rationale", "")).strip()
    risks = str(data.get("risks", "")).strip()
    content = (
        f"{symbol} — model says {p['direction']} @ {model_conf}% (regime {p.get('regime')}). "
        f"Verifier: {'VETO' if veto else ('agree' if data.get('agree') else 'downgrade')} → "
        f"final {final_conf}%. {rationale} Risk: {risks}"
    )

    verdict.update(
        {
            **p,
            "model_confidence": model_conf,
            "llm_confidence": llm_conf,
            "final_confidence": final_conf,
            "final_act": final_act,
            "final_kelly_frac": round(final_kelly, 4),
            "veto": veto,
            "agree": bool(data.get("agree", False)),
            "contradicts_model": bool(data.get("contradicts_model", False)),
            "rationale": rationale,
            "risks": risks,
            "key_levels": str(data.get("key_levels", "")).strip(),
        }
    )

    if persist:
        try:
            await insert_insight(
                {
                    "symbol": symbol,
                    "insight_type": "prediction",
                    "content": content,
                    "sentiment": "bearish" if p["direction"] == "DOWN" else "bullish",
                    "confidence": final_conf,
                    "generated_at": int(time.time() * 1000),
                }
            )
        except Exception as exc:
            log.warning("verifier persist failed for %s: %s", symbol, exc)

    log.info(
        "Verifier %s: model %d%% -> final %d%% (veto=%s)", symbol, model_conf, final_conf, veto
    )
    return verdict


# ── Portfolio-level red-team (Phase 8) ────────────────────────────────────────────
def _conviction(p: dict) -> float:
    """Phase-7 cross-sectional conviction score: signed edge × meta-prob (the leaderboard rank)."""
    meta = p.get("meta_prob")
    return (float(p.get("prob_up", 0.5)) - 0.5) * float(meta if meta is not None else 0.0)


async def verify_portfolio(
    predictions: list[dict] | None = None, top_k: int = 5, persist: bool = True
) -> list[dict]:
    """
    Red-team the TOP-K portfolio names through the Layer-3 verifier (Phase 8).

    The deployable book is long-only, edge-ranked (Phase 7), so the names that actually carry
    capital are the highest-conviction longs. We rank the live leaderboard by ``edge × meta``,
    take the top-k, and run each through ``verify_prediction`` — which can only DOWNGRADE or VETO
    (never raise confidence) and persists an auditable rationale. This focuses the (slower) LLM
    pass on the handful of names that matter instead of the whole universe.

    Pass an already-computed ``predictions`` batch (e.g. from the daily cycle) to avoid recomputing;
    otherwise it runs ``predict_all`` itself. Returns the list of verdicts (highest conviction first).
    Degrades gracefully per name: an LLM outage yields ``verifier="unavailable"`` with confidence
    left at the model ceiling, never raised.
    """
    if predictions is None:
        from .predict import predict_all

        predictions = await predict_all()
    ranked = sorted((p for p in predictions if p), key=_conviction, reverse=True)

    verdicts: list[dict] = []
    for p in ranked[: max(0, top_k)]:
        try:
            v = await verify_prediction(p["symbol"], prediction=p, persist=persist)
        except Exception as exc:  # one bad name never sinks the batch
            log.warning("verify_portfolio %s failed: %s", p.get("symbol"), exc)
            continue
        if v:
            # Defensive: the contract is enforced inside verify_prediction, but assert it here too
            # so a future regression in the verdict path can never silently raise a live number.
            mc = int(v.get("model_confidence", v.get("confidence", 0)))
            v["final_confidence"] = min(int(v.get("final_confidence", mc)), mc)
            verdicts.append(v)

    n_veto = sum(1 for v in verdicts if v.get("veto"))
    n_down = sum(
        1 for v in verdicts if int(v.get("final_confidence", 0)) < int(v.get("model_confidence", 0))
    )
    log.info(
        "Portfolio verify: %d/%d names reviewed | %d veto | %d downgraded",
        len(verdicts),
        min(top_k, len(ranked)),
        n_veto,
        n_down,
    )
    return verdicts


if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    async def _demo():
        from backend.db import init_db

        await init_db()
        v = await verify_prediction("AAPL")
        if not v:
            print("no prediction")
            return
        print(
            f"model={v.get('model_confidence', v['confidence'])}% "
            f"llm={v.get('llm_confidence')}% final={v.get('final_confidence')}% "
            f"veto={v.get('veto')} verifier={v.get('verifier')}"
        )
        print("rationale:", v.get("rationale"))

    asyncio.run(_demo())
