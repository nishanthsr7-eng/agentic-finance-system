"""
FLUX — LLM provider layer
=========================

One chat interface, two transports:

  * ``ollama``  — the local dev path (http://localhost:11434/api/chat)
  * ``openai``  — any OpenAI-compatible /chat/completions endpoint, which is
                  what every free hosted tier speaks: Groq, OpenRouter,
                  Together, DeepInfra, Gemini's compat shim, vLLM, LM Studio.

Deployment can't run Ollama (no free tier will host a local LLM), so the hosted
build points at a free cloud endpoint while local dev keeps working unchanged.

Selection is automatic: set ``LLM_API_KEY`` and the openai transport is used,
otherwise it falls back to Ollama. ``LLM_PROVIDER`` forces one explicitly.

    # local (nothing to set — Ollama as before)
    OLLAMA_URL=http://localhost:11434
    OLLAMA_MODEL=aura

    # hosted, e.g. Groq
    LLM_BASE_URL=https://api.groq.com/openai/v1
    LLM_API_KEY=gsk_...
    LLM_MODEL=openai/gpt-oss-120b

Both transports expose the same three calls, so the route handlers don't care
which one is live:

    await chat(messages)          -> str
    stream_chat(messages)         -> async generator of token strings
    await health()                -> (ok: bool, model: str)
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx

from .config import settings

log = logging.getLogger("flux.llm")


class LLMError(RuntimeError):
    """Raised when the configured provider is unreachable or errors out."""


def _describe(exc: Exception) -> str:
    """
    Readable one-liner for an exception.

    httpx's timeout classes stringify to "", which would otherwise surface to
    the client as a bare "Ollama error: " — the class name is the whole signal
    for those, and a cold model load hitting the request timeout is by far the
    most common local failure.
    """
    msg = str(exc).strip()
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


# ── Provider resolution ──────────────────────────────────────────────────────


def provider() -> str:
    """Active transport: "openai" or "ollama"."""
    p = (settings.LLM_PROVIDER or "").strip().lower()
    if p in ("openai", "ollama"):
        return p
    # auto: a key means a hosted OpenAI-compatible endpoint is configured
    return "openai" if settings.LLM_API_KEY else "ollama"


def model_name() -> str:
    """The model id the active provider will be asked for."""
    if provider() == "openai":
        return settings.LLM_MODEL or "openai/gpt-oss-120b"
    return settings.OLLAMA_MODEL


def _base_url() -> str:
    return (settings.LLM_BASE_URL or "https://api.groq.com/openai/v1").rstrip("/")


def _headers() -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if settings.LLM_API_KEY:
        h["Authorization"] = f"Bearer {settings.LLM_API_KEY}"
    return h


_client: httpx.AsyncClient | None = None


def _http() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=60.0)
    return _client


async def aclose() -> None:
    """Close the shared client (called from the app shutdown hook)."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


# ── Chat (non-streaming) ─────────────────────────────────────────────────────


async def chat(
    messages: list[dict[str, str]],
    *,
    timeout: float = 45.0,
    temperature: float | None = None,
) -> str:
    """
    Send a chat completion and return the assistant text.

    ``messages`` is the OpenAI/Ollama shared shape: [{"role", "content"}, ...].
    Raises LLMError on any transport or provider failure.
    """
    if provider() == "openai":
        return await _openai_chat(messages, timeout=timeout, temperature=temperature)
    return await _ollama_chat(messages, timeout=timeout)


async def _openai_chat(
    messages: list[dict[str, str]],
    *,
    timeout: float,
    temperature: float | None,
) -> str:
    payload: dict = {
        "model": model_name(),
        "messages": messages,
        "stream": False,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    try:
        r = await _http().post(
            f"{_base_url()}/chat/completions",
            json=payload,
            headers=_headers(),
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"].get("content") or "").strip()
    except httpx.HTTPStatusError as exc:
        # Surface the provider's own message — rate limits and bad model ids are
        # the two failures worth reading verbatim while wiring a new key up.
        detail = exc.response.text[:300]
        raise LLMError(f"LLM HTTP {exc.response.status_code}: {detail}") from exc
    except (KeyError, IndexError) as exc:
        raise LLMError("LLM returned an unexpected response shape") from exc
    except Exception as exc:
        raise LLMError(f"LLM error: {_describe(exc)}") from exc


async def _ollama_chat(messages: list[dict[str, str]], *, timeout: float) -> str:
    payload = {
        "model": settings.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }
    try:
        r = await _http().post(
            f"{settings.OLLAMA_URL}/api/chat",
            json=payload,
            timeout=timeout,
        )
        r.raise_for_status()
        return (r.json().get("message", {}).get("content") or "").strip()
    except httpx.ConnectError as exc:
        raise LLMError("Ollama is not running — start with: ollama serve") from exc
    except Exception as exc:
        raise LLMError(f"Ollama error: {_describe(exc)}") from exc


# ── Chat (streaming) ─────────────────────────────────────────────────────────


async def stream_chat(
    messages: list[dict[str, str]],
    *,
    timeout: float = 60.0,
) -> AsyncIterator[str]:
    """
    Yield assistant tokens as they arrive. Raises LLMError before the first
    token if the provider is unreachable; mid-stream errors end the stream.
    """
    if provider() == "openai":
        async for tok in _openai_stream(messages, timeout=timeout):
            yield tok
    else:
        async for tok in _ollama_stream(messages, timeout=timeout):
            yield tok


async def _openai_stream(
    messages: list[dict[str, str]],
    *,
    timeout: float,
) -> AsyncIterator[str]:
    payload = {"model": model_name(), "messages": messages, "stream": True}
    try:
        async with _http().stream(
            "POST",
            f"{_base_url()}/chat/completions",
            json=payload,
            headers=_headers(),
            timeout=timeout,
        ) as r:
            if r.status_code >= 400:
                body = (await r.aread()).decode(errors="replace")[:300]
                raise LLMError(f"LLM HTTP {r.status_code}: {body}")
            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    token = delta.get("content") or ""
                    if token:
                        yield token
                except Exception:
                    # A malformed or keep-alive frame is not fatal to the stream.
                    continue
    except LLMError:
        raise
    except httpx.ConnectError as exc:
        raise LLMError("LLM endpoint unreachable") from exc


async def _ollama_stream(
    messages: list[dict[str, str]],
    *,
    timeout: float,
) -> AsyncIterator[str]:
    payload = {"model": settings.OLLAMA_MODEL, "messages": messages, "stream": True}
    try:
        async with _http().stream(
            "POST",
            f"{settings.OLLAMA_URL}/api/chat",
            json=payload,
            timeout=timeout,
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content") or ""
                    if token:
                        yield token
                    if chunk.get("done"):
                        return
                except Exception:
                    continue
    except httpx.ConnectError as exc:
        raise LLMError("Ollama is not running — start with: ollama serve") from exc


# ── Health ───────────────────────────────────────────────────────────────────


async def health() -> tuple[bool, str]:
    """
    Cheap liveness probe for /health. Never raises.

    Ollama gets a real /api/tags ping. For hosted providers a round-trip per
    health call would burn free-tier quota, so a configured key counts as
    available — a broken key surfaces on the first real chat instead.
    """
    if provider() == "openai":
        return bool(settings.LLM_API_KEY), model_name()
    try:
        async with httpx.AsyncClient(timeout=1.5) as c:
            r = await c.get(f"{settings.OLLAMA_URL}/api/tags")
            return r.status_code == 200, settings.OLLAMA_MODEL
    except Exception:
        return False, settings.OLLAMA_MODEL


def strip_fences(text: str) -> str:
    """Remove markdown code fences that models wrap JSON replies in."""
    t = (text or "").strip()
    if t.startswith("```"):
        parts = t.split("```")
        t = parts[1] if len(parts) > 1 else t
        if t.startswith("json"):
            t = t[4:]
    return t.strip()
