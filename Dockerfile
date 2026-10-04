# FLUX backend — container image.
#
# Two builds from one file, selected by the FULL build arg:
#
#   slim (default)  backend/requirements-slim.lock (exact pins of -slim.txt), no torch. ~248 MB resident,
#                   which fits Render's free 512 MB tier. FinBERT sentiment is
#                   replaced by SENTIMENT_BACKEND=llm; the LSTM magnitude head
#                   and the Chronos baseline are unavailable.
#   full            backend/requirements.txt plus CPU torch. ~748 MB resident,
#                   so it needs a host with 1 GB+.
#
# Hugging Face Spaces was the original target — it gave 16 GB free — but Docker
# Spaces moved behind PRO in July 2026, and no remaining card-free tier fits the
# full set. Hence the split.
#
#   docker build -t flux-api .                       # slim
#   docker build --build-arg FULL=1 -t flux-api .    # full
#   docker run --rm -p 8080:8080 -e PORT=8080 --env-file .env flux-api

FROM python:3.11-slim

# Some scientific wheels (hmmlearn, arch) still build from source on slim.
# curl is kept for the healthcheck; the rest is dropped from the layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# The container runs as uid 1000 rather than root. Everything the app writes
# at runtime must be owned by that user, so create it before installing.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── Memory budget ────────────────────────────────────────────────────────
# All of this exists because the free tier gives 512 MB and 0.1 CPU.
#
# The *_NUM_THREADS vars: numpy/OpenBLAS, xgboost and onnxruntime each size
# their thread pools from the host's core count. Render reports many cores
# while giving 0.1 of one, so the defaults allocate a dozen per-thread arenas
# and buffers that can never run in parallel anyway — pure resident memory.
#
# MALLOC_ARENA_MAX: glibc otherwise creates up to 8 * ncores malloc arenas and
# is slow to return them to the OS. Capping it at 2 is the cheapest RSS win
# available to a threaded Python process in a small container.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    TOKENIZERS_PARALLELISM=false \
    ANONYMIZED_TELEMETRY=False

WORKDIR $HOME/app

# ── Dependencies ─────────────────────────────────────────────────────────────
# FULL=1 adds torch and the requirements.txt extras; the default is the slim set.
ARG FULL=0

# CPU-only torch first, on its own index, and only for the full build. The
# default PyPI wheel bundles CUDA and is ~2.5 GB; the CPU build is ~200 MB and
# no free tier has a GPU anyway. Installing it up front means the torch>=2.2
# pin in requirements.txt is already satisfied and pip won't pull the CUDA wheel.
# The version spec must stay quoted: RUN uses a shell, which would otherwise
# read `torch>=2.2` as a redirect and write an empty file named "=2.2".
RUN if [ "$FULL" = "1" ]; then \
        pip install --no-cache-dir --user \
            --index-url https://download.pytorch.org/whl/cpu \
            "torch>=2.2"; \
    fi

COPY --chown=user backend/requirements.txt      ./backend/requirements.txt
COPY --chown=user backend/requirements-slim.lock ./backend/requirements-slim.lock
RUN if [ "$FULL" = "1" ]; then \
        pip install --no-cache-dir --user -r backend/requirements.txt; \
    else \
        pip install --no-cache-dir --user -r backend/requirements-slim.lock; \
    fi

# ── Embedding model ──────────────────────────────────────────────────
# Bake ChromaDB's all-MiniLM-L6-v2 ONNX embedder into the image.
#
# Chroma fetches it lazily, on the first embed call, into ~/.cache/chroma — an
# 80 MB download plus a tar extraction plus building the first InferenceSession,
# all inside a process that is already close to the 512 MB ceiling. That spike
# is the most likely OOM on this plan, and it recurs on every boot because the
# free tier's disk is ephemeral while the image is not.
#
# Doing it here pays the cost once, at build time, on the builder's larger box;
# at runtime the file is simply already present. It also takes a network
# dependency off the request path — an S3 blip can no longer fail an embed.
#
# It has to be a real embed rather than just a constructor, because the
# download is triggered from __call__. `|| true` keeps a transient build-time
# network failure from breaking the image: the runtime path still works, it
# just falls back to downloading on first use, exactly as it does today.
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as D; D()(['warm up the onnx embedder'])" || true

# ── Application ──────────────────────────────────────────────────────────────
COPY --chown=user backend ./backend

# Writable runtime paths. The defaults in config.py resolve to a per-user app
# data dir; pinning them here keeps SQLite, Chroma and the model caches inside
# the container's writable layer instead of somewhere read-only.
ENV DB_PATH=$HOME/app/data/flux_market.db \
    CHROMA_PATH=$HOME/app/data/chroma \
    HF_HOME=$HOME/app/data/hf \
    XDG_CACHE_HOME=$HOME/app/data/cache

RUN mkdir -p $HOME/app/data/chroma $HOME/app/data/hf $HOME/app/data/cache

# Hosts inject the port to listen on as $PORT and ignore EXPOSE; 8080 is the
# common default and the right fallback for a plain `docker run`.
ENV PORT=8080
EXPOSE 8080

# The platform health-checks the service itself, so this only matters locally.
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

# Shell form on purpose: $PORT has to be expanded at runtime, and the exec form
# would pass the literal string "$PORT" to uvicorn.
CMD exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}
