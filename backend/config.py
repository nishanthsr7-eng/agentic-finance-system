"""
FLUX Backend — Configuration
Reads API keys and tunables from the project-root .env file.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Market Data ───────────────────────────────────────────
    COINGECKO_API_KEY: str = ""
    FINNHUB_API_KEY: str = ""
    ALPHA_VANTAGE_API_KEY: str = ""
    NEWSAPI_KEY: str = ""
    FRED_API_KEY: str = ""  # macro series (term spread, rates, CPI) — optional

    # ── Reddit (retail-flow sentiment) ────────────────────────
    REDDIT_CLIENT_ID: str = ""
    REDDIT_CLIENT_SECRET: str = ""
    REDDIT_USER_AGENT: str = "flux-market/0.1"

    # ── Alpaca (paper forward-test only — NEVER live trading) ─
    ALPACA_API_KEY: str = ""
    ALPACA_SECRET_KEY: str = ""
    ALPACA_PAPER: bool = True  # hard default: paper endpoint only

    # ── Cache TTLs (seconds) ──────────────────────────────────
    CRYPTO_TTL: int = 30  # matches client sweep interval
    STOCKS_TTL: int = 60  # Finnhub free tier: 60 req/min
    NEWS_TTL: int = 300  # news refreshes every 5 min

    # ── Ollama (local dev LLM) ───────────────────────────────
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "aura"

    # ── Hosted LLM (deployment) ──────────────────────────────
    # No free host will run Ollama for us, so deployed builds point at an
    # OpenAI-compatible endpoint instead (Groq, OpenRouter, Gemini's compat
    # shim, …). Setting LLM_API_KEY is what switches backend/llm.py over;
    # leave it empty and everything keeps using local Ollama.
    # LLM_PROVIDER: "" = auto (key present → openai), or force "openai"/"ollama".
    LLM_PROVIDER: str = ""
    LLM_BASE_URL: str = "https://api.groq.com/openai/v1"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "openai/gpt-oss-120b"

    # ── News sentiment backend (Layer 2c) ────────────────────
    # "auto"         — FinBERT/CryptoBERT if torch is installed, else the LLM
    # "transformers" — force the local models (needs torch, ~445 MB resident)
    # "llm"          — force the chat model, which is what fits a 512 MB host
    SENTIMENT_BACKEND: str = "auto"

    # ── CORS ─────────────────────────────────────────────────
    # Dev frontend origins only — widen explicitly via .env for deployment,
    # never back to "*" (wildcard + Authorization headers is a footgun).
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # ── Auth ─────────────────────────────────────────────────
    # AUTH_REQUIRED=false → unauthenticated requests act as the demo user
    # (id=1). Keep true for anything reachable by others.
    AUTH_REQUIRED: bool = True
    # Empty → a random secret is generated and persisted in the app-data dir.
    AUTH_SECRET: str = ""
    # Sent as `X-Admin-Token` to unlock maintenance routes (ingestion trigger,
    # /predict/run, /cache/flush, insight refresh, ?fresh=true). Empty → those
    # routes always answer 403. The scheduler calls the jobs directly, not over
    # HTTP, so cron is unaffected either way.
    ADMIN_TOKEN: str = ""

    # ── Persistence ───────────────────────────────────────────
    # Empty → auto-resolve to the per-user app-data dir (%LOCALAPPDATA%/flux on
    # Windows, ~/.local/share/flux elsewhere). Never default these into the
    # project root: the dev static server serves that directory publicly.
    DB_PATH: str = ""
    # "mysql" keeps predictions, prediction_outcomes, calibration_buckets and
    # ohlcv_history in MySQL/TiDB so they survive a restart; the other market
    # tables stay in SQLite either way. "sqlite" = the original behaviour.
    MARKET_STORE: str = "sqlite"
    CHROMA_PATH: str = ""

    # ── MySQL (seeded "ultimate" dataset for all pages) ───────
    MYSQL_HOST: str = "127.0.0.1"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""
    MYSQL_DB: str = "flux"
    # Managed/free MySQL tiers (TiDB Serverless, Aiven, PlanetScale) refuse
    # plaintext connections. Local MySQL doesn't want TLS at all, so this stays
    # off by default and deployment turns it on.
    MYSQL_SSL: bool = False
    # Optional CA bundle path. Empty → the system trust store, which is what
    # every managed provider's public cert chains to.
    MYSQL_SSL_CA: str = ""
    # Reuse up to this many open connections instead of a fresh TLS handshake
    # per query. 0 = no pool (connect per call, the original behaviour).
    MYSQL_POOL_SIZE: int = 0
    # A pooled connection idle longer than this is pinged (and reconnected if
    # the server dropped it) before reuse. TiDB Serverless closes idle ones.
    MYSQL_POOL_PING_AFTER_S: int = 30

    # ── Ingestion ─────────────────────────────────────────────
    # Every interval below is honoured by ingestion.build_scheduler. They used
    # to be hardcoded there, which quietly ignored INGESTION_INTERVAL_MIN and
    # burned the NewsAPI free quota (100 req/day) in a few hours.
    INGESTION_ENABLED: bool = True
    INGESTION_INTERVAL_MIN: int = 5  # crypto + stocks cycle
    OHLCV_INTERVAL_MIN: int = 30  # 30-day daily bars
    NEWS_INTERVAL_MIN: int = 15  # NewsAPI headlines
    INSIGHT_MAX_ASSETS: int = 6  # top movers to analyse per cycle
    # The insight cycle is chained onto the market cycle rather than scheduled
    # independently, so it always reads a cache that was just refreshed. This
    # is the minimum gap between two chained runs, not a trigger of its own.
    INSIGHT_INTERVAL_MIN: int = 15
    SNAPSHOT_RETENTION_DAYS: int = 7  # prune older price_snapshots
    INGESTION_LOG_RETENTION_DAYS: int = 30  # prune older ingestion_log rows (0 = keep all)

    # ── Memory budget (512 MB hosts) ──────────────────────────
    # The prediction cycle, the drift retrain and the options snapshot are the
    # heaviest things this process does. Running them a few minutes after boot
    # is convenient in development and fatal on a 512 MB box, where it turns a
    # single OOM into a restart loop. Deployment sets this false and relies on
    # the cron triggers plus POST /ingestion/trigger/{job}.
    HEAVY_JOBS_ON_STARTUP: bool = True
    # One switch to take ChromaDB (and its ~80 MB ONNX embedder) out of the
    # process entirely, without a redeploy, if the host still runs out of room.
    RAG_ENABLED: bool = True

    # ── Demo account ──────────────────────────────────────────
    # Rebuild the demo user (id 1) daily at 00:05 UTC and once after boot if
    # that hasn't happened today, so its data always ends on today's date.
    DEMO_RESEED: bool = True

    model_config = {
        "env_file": str(Path(__file__).parent.parent / ".env"),
        "extra": "ignore",
    }


settings = Settings()
