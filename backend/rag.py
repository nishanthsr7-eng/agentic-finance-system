"""
FLUX — ChromaDB RAG Pipeline
-----------------------------
Collections:
  market_snapshots  — price narratives (updated every ingestion cycle)
  financial_news    — news article embeddings
  ai_insights       — pre-computed insight embeddings

All ChromaDB operations are wrapped in try/except so the rest of the
backend degrades gracefully if ChromaDB is not installed or fails.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any

from .config import settings

log = logging.getLogger("flux.rag")


def _resolve_chroma_dir() -> Path:
    """Kept outside the project root (see backend.db._resolve_db_path) so the
    dev static server can't serve the vector store. Override with CHROMA_PATH."""
    if settings.CHROMA_PATH:
        return Path(settings.CHROMA_PATH)
    base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / ".local" / "share")) / "flux"
    base.mkdir(parents=True, exist_ok=True)
    return base / "flux_chroma"


_CHROMA_DIR = _resolve_chroma_dir()

# Suppress noisy ONNX TensorRT/CUDA probe warnings — falls back to CPU cleanly
import os as _os

_os.environ.setdefault("ORT_LOGGING_LEVEL", "3")  # ERROR only

# ── Module-level lazy singletons ──────────────────────────────────────────────
_chroma_client: Any = None
_collections: dict = {}
_embedding_fn: Any = None
_rag_available: bool = False


def _init_embedding_fn() -> Any:
    """
    Try embedding functions in priority order:
    1. ChromaDB DefaultEmbeddingFunction (onnxruntime, ~80 MB ONNX model)
    2. None — RAG degrades to keyword-only retrieval
    """
    global _embedding_fn
    if _embedding_fn is not None:
        return _embedding_fn
    try:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        _embedding_fn = DefaultEmbeddingFunction()
        log.info("ChromaDB: using DefaultEmbeddingFunction (all-MiniLM-L6-v2 ONNX)")
    except Exception as exc:
        log.warning("ChromaDB embedding init failed (%s) — RAG disabled", exc)
        _embedding_fn = None
    return _embedding_fn


def init_chroma() -> bool:
    """
    Initialise ChromaDB persistent client and create/open all collections.
    Returns True if successful, False if ChromaDB unavailable.
    """
    global _chroma_client, _rag_available
    if not settings.RAG_ENABLED:
        # One env var takes ChromaDB, onnxruntime and the ~80 MB MiniLM model
        # out of the process entirely. Every rag.* helper already no-ops when
        # _rag_available is False, so the app keeps working without a redeploy.
        log.info("RAG disabled (RAG_ENABLED=false) — skipping ChromaDB init")
        _rag_available = False
        return False
    try:
        import chromadb

        _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
        ef = _init_embedding_fn()
        for name in ("market_snapshots", "financial_news", "ai_insights"):
            _collections[name] = _chroma_client.get_or_create_collection(
                name=name,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
        _rag_available = True
        log.info("ChromaDB initialised at %s (3 collections)", _CHROMA_DIR)
        return True
    except ImportError:
        log.warning("chromadb not installed — RAG pipeline disabled. Run: pip install chromadb")
        _rag_available = False
        return False
    except Exception as exc:
        log.error("ChromaDB init failed: %s", exc)
        _rag_available = False
        return False


def is_available() -> bool:
    return _rag_available


def _col(name: str):
    return _collections.get(name)


# ── Embedding helpers ─────────────────────────────────────────────────────────


def _price_narrative(snap: dict) -> str:
    """Convert a snapshot dict to an embeddable text sentence."""
    direction = "up" if (snap.get("change_pct") or 0) >= 0 else "down"
    mcap = snap.get("market_cap", 0) or 0
    cap_str = f" Market cap: ${mcap / 1e9:.1f}B." if mcap > 0 else ""
    return (
        f"{snap.get('name', snap['symbol'])} ({snap['symbol']}) "
        f"is trading at {snap['price']:.4f}, "
        f"{abs(snap.get('change_pct', 0)):.2f}% {direction} in 24h.{cap_str} "
        f"Asset type: {snap.get('asset_type', 'unknown')}."
    )


def embed_market_snapshots(snapshots: list[dict]) -> None:
    """Upsert latest price snapshots into the market_snapshots collection."""
    if not _rag_available or not snapshots:
        return
    try:
        col = _col("market_snapshots")
        if col is None:
            return
        ts = int(time.time())
        docs, ids, metas = [], [], []
        for s in snapshots:
            sym = s.get("symbol", "")
            if not sym:
                continue
            doc_id = f"{sym}_{ts}"
            docs.append(_price_narrative(s))
            ids.append(doc_id)
            metas.append(
                {
                    "symbol": sym,
                    "price": float(s.get("price", 0)),
                    "change_pct": float(s.get("change_pct", 0)),
                    "ts": ts,
                }
            )
        col.upsert(documents=docs, ids=ids, metadatas=metas)
        log.debug("Embedded %d market snapshots in ChromaDB", len(docs))
    except Exception as exc:
        log.warning("embed_market_snapshots failed: %s", exc)


def embed_news(articles: list[dict]) -> None:
    """Upsert news articles into the financial_news collection."""
    if not _rag_available or not articles:
        return
    try:
        col = _col("financial_news")
        if col is None:
            return
        docs, ids, metas = [], [], []
        for a in articles:
            url = (a.get("url") or "")[:255]
            if not url:
                continue
            text = f"{a.get('title', '')}. {a.get('summary', '')}".strip()
            if not text:
                continue
            docs.append(text)
            ids.append(url)
            metas.append(
                {
                    "source": a.get("source", ""),
                    "published_at": a.get("published_at", ""),
                    "url": url,
                }
            )
        if docs:
            col.upsert(documents=docs, ids=ids, metadatas=metas)
            log.debug("Embedded %d news articles in ChromaDB", len(docs))
    except Exception as exc:
        log.warning("embed_news failed: %s", exc)


def embed_insight(insight: dict) -> None:
    """Upsert a single AI insight into the ai_insights collection."""
    if not _rag_available:
        return
    try:
        col = _col("ai_insights")
        if col is None:
            return
        doc_id = f"insight_{insight['symbol']}_{insight['generated_at']}"
        col.upsert(
            documents=[insight["content"]],
            ids=[doc_id],
            metadatas=[
                {
                    "symbol": insight["symbol"],
                    "sentiment": insight.get("sentiment", ""),
                    "insight_type": insight.get("insight_type", ""),
                    "generated_at": int(insight.get("generated_at", 0)),
                }
            ],
        )
    except Exception as exc:
        log.warning("embed_insight failed: %s", exc)


# ── Query ─────────────────────────────────────────────────────────────────────


def rag_query(query: str, n_results: int = 6) -> tuple[str, int]:
    """
    Query all three collections and return (context_string, chunk_count).
    Returns ("", 0) when RAG is unavailable or no results found.
    """
    if not _rag_available:
        return "", 0

    contexts: list[str] = []
    per_col = max(2, n_results // 3)

    for col_name in ("ai_insights", "market_snapshots", "financial_news"):
        col = _col(col_name)
        if col is None:
            continue
        try:
            results = col.query(query_texts=[query], n_results=per_col)
            docs = results.get("documents", [[]])[0]
            contexts.extend(docs)
        except Exception as exc:
            log.debug("RAG query on %s skipped: %s", col_name, exc)

    if not contexts:
        return "", 0

    merged = "\n\n".join(f"[Context {i + 1}] {doc}" for i, doc in enumerate(contexts[:n_results]))
    return merged, len(contexts[:n_results])


def chroma_stats() -> dict:
    """Return collection sizes for the /health endpoint."""
    if not _rag_available:
        return {"available": False}
    stats: dict = {"available": True}
    for name, col in _collections.items():
        try:
            stats[name] = col.count()
        except Exception:
            stats[name] = -1
    return stats
