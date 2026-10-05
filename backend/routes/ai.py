"""AI insights, RAG Q&A, asset intel and chat."""

import asyncio
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from .. import llm
from ..auth import require_admin, require_user
from ..cache import cache
from ..db import (
    get_latest_insights,
)
from ..rag import rag_query
from .common import _AI_LIMIT, _cached_assets, _insight_job, _now_iso, log

router = APIRouter()

# Shown to users instead of the provider's error text, which can include HTTP bodies
# and setup hints. The full error still goes to the log.
_AI_DOWN = "The AI service is unavailable right now. Please try again shortly."


# ── §B: AI Insights Endpoints ─────────────────────────────────────────────────


@router.get("/ai/insights")
async def ai_insights_list(
    limit: int = Query(20, ge=1, le=100),
    symbol: str = Query("", description="Filter by symbol"),
):
    """Latest pre-computed AI insights from SQLite."""
    rows = await get_latest_insights(
        limit=limit,
        symbol=symbol.upper() if symbol else None,
    )
    return {"insights": rows, "count": len(rows), "timestamp": _now_iso()}


@router.post("/ai/insights/refresh", dependencies=[Depends(require_admin)])
async def ai_insights_refresh():
    """Trigger an immediate AI insight generation cycle (non-blocking)."""
    all_assets = _cached_assets()
    if not all_assets:
        raise HTTPException(503, "No asset data in cache yet — wait for the first ingestion cycle")

    asyncio.create_task(_insight_job())
    return {
        "status": "refresh_started",
        "assets": len(all_assets),
        "timestamp": _now_iso(),
    }


# ── §C: RAG Query Endpoint ────────────────────────────────────────────────────


class RagQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    include_context: bool = False


@router.post("/ai/rag/query", dependencies=[Depends(require_user), _AI_LIMIT])
async def ai_rag_query(body: RagQueryRequest):
    """
    RAG-enhanced financial Q&A.
    Retrieves relevant context from ChromaDB, injects it into the Ollama
    system prompt, and returns a grounded answer.
    """
    context, n_chunks = rag_query(body.query)

    system_content = (
        "You are AURA, a professional AI financial advisor. "
        "Answer with data-driven precision. Keep responses under 150 words.\n\n"
    )
    if context:
        system_content += f"=== LIVE MARKET CONTEXT (retrieved) ===\n{context}\n=== END ==="

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": body.query},
    ]
    try:
        answer = await llm.chat(messages, timeout=45.0)
    except llm.LLMError as exc:
        log.error("RAG chat failed: %s", exc)
        raise HTTPException(503, _AI_DOWN) from exc
    except Exception as exc:
        log.error("RAG chat failed: %s", exc)
        raise HTTPException(502, "AI unavailable") from exc

    resp: dict = {
        "answer": answer,
        "rag_available": bool(context),
        "context_chunks": n_chunks,
        "timestamp": _now_iso(),
    }
    if body.include_context:
        resp["context"] = context
    return resp


# ── AI Chat (Ollama) ──────────────────────────────────────────────────────────


class _ChatMsg(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., max_length=20_000)


class ChatRequest(BaseModel):
    messages: list[_ChatMsg] = Field(..., min_length=1, max_length=50)


@router.get("/ai/intel/{symbol}", dependencies=[Depends(require_user), _AI_LIMIT])
async def ai_intel(symbol: str):
    """
    Returns structured AI analysis for an asset.
    Injects live price/change/mcap context and instructs Ollama to reply in JSON.
    Response: { consensus, confidence, vol_profile, report, catalysts[] }
    """
    # Resolve live asset data from cache (best-effort)
    sym_upper = symbol.upper()
    asset_data: dict | None = None
    for cache_key in ("crypto", "stocks"):
        pool = cache.get(cache_key)
        if pool:
            for a in pool:
                if (
                    a.get("sub", "").upper() == sym_upper
                    or a.get("symbol", "").upper() == sym_upper
                ):
                    asset_data = a
                    break
        if asset_data:
            break

    if asset_data:
        price = asset_data.get("price", 0)
        change_pct = asset_data.get("change_pct", 0)
        name = asset_data.get("name", symbol)
        mcap = asset_data.get("market_cap", 0)
        direction = "bullish" if change_pct >= 0 else "bearish"
        context_line = (
            f"{name} ({sym_upper}) is trading at ${price:,.2f}, "
            f"{'+' if change_pct >= 0 else ''}{change_pct:.2f}% in the last 24h "
            + (f"with a market cap of ${mcap / 1e9:.1f}B. " if mcap else ". ")
            + f"Sentiment direction: {direction}."
        )
    else:
        name = symbol
        context_line = f"Asset: {symbol}. No live price data available."

    prompt = (
        f"You are a professional market analyst. Analyze the following asset and respond ONLY with valid JSON — "
        f"no markdown, no explanation, no code fences.\n\n"
        f"Market data: {context_line}\n\n"
        f"Respond with exactly this JSON schema:\n"
        f'{{"consensus": "<Bullish|Bearish|Neutral> (<percent>%)", '
        f'"confidence": <integer 0-100>, '
        f'"vol_profile": "<short phrase describing volume/flow>", '
        f'"report": "<2-3 sentence market analysis>", '
        f'"catalysts": ["<tag1>", "<tag2>", "<tag3>"]}}'
    )

    try:
        raw = llm.strip_fences(await llm.chat([{"role": "user", "content": prompt}], timeout=30.0))

        import json as _json

        intel = _json.loads(raw)
        intel.setdefault("symbol", symbol)
        intel.setdefault("name", name)
        intel.setdefault("available", True)
        return intel
    except Exception as e:
        # Honest degraded response — we do NOT fabricate a buy/sell signal when the
        # model is offline or returns junk. The frontend renders an "Unavailable"
        # state from available:false; live price/headlines remain real.
        log.warning("ai_intel unavailable for %s: %s", symbol, e)
        return {
            "symbol": symbol,
            "name": name,
            "available": False,
            "consensus": None,
            "confidence": None,
            "vol_profile": None,
            "report": "AI analysis is temporarily unavailable.",
            "catalysts": [],
        }


@router.post("/ai/chat/stream", dependencies=[Depends(require_user), _AI_LIMIT])
async def ai_chat_stream(body: ChatRequest):
    """
    Streaming version of /ai/chat — returns Server-Sent Events.
    Each SSE data line is JSON: {"content": "<token>"}.
    Final line is: data: [DONE]
    """
    import json as _json

    from fastapi.responses import StreamingResponse

    messages = [{"role": m.role, "content": m.content} for m in body.messages]

    async def generate():
        try:
            async for token in llm.stream_chat(messages, timeout=60.0):
                yield f"data: {_json.dumps({'content': token})}\n\n"
            # The provider's own terminator is consumed by the token generator,
            # so the sentinel the client waits on is emitted here.
            yield "data: [DONE]\n\n"
        except Exception as e:
            log.error("Streaming chat error: %s", e)
            yield f"data: {_json.dumps({'error': _AI_DOWN})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/ai/chat", dependencies=[Depends(require_user), _AI_LIMIT])
async def ai_chat(body: ChatRequest):
    """
    Forward a chat request to the configured LLM provider and return the reply.

    Provider is resolved in backend/llm.py: a hosted OpenAI-compatible endpoint
    when LLM_API_KEY is set, otherwise the local Ollama instance (OLLAMA_MODEL).
    """
    try:
        content = await llm.chat(
            [{"role": m.role, "content": m.content} for m in body.messages],
            timeout=30.0,
        )
        return {"content": content}
    except llm.LLMError as e:
        log.error("AI chat failed: %s", e)
        raise HTTPException(503, _AI_DOWN) from e
    except Exception as e:
        log.error("AI chat failed: %s", e)
        raise HTTPException(502, "AI unavailable") from e
