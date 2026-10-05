# Architecture

A technical overview of how FLUX is structured and how data flows through the
system.

---

## 1. High-Level View

```
┌──────────────┐   HTTP/SSE    ┌────────────────────────────────────────────┐
│  Frontend    │ ◄───────────► │  FastAPI Backend (backend/main.py)         │
│  HTML/CSS/JS │   :3000→:8000 │  routes · CORS · security headers          │
└──────────────┘               └───────┬───────────────┬───────────┬────────┘
                                        │               │           │
                              ┌─────────▼──┐   ┌────────▼──────┐ ┌──▼───────────┐
                              │ APScheduler│   │ Persistence   │ │ AI / LLM     │
                              │ ingestion  │   │ SQLite · MySQL│ │ Groq / Ollama│
                              │ + predict  │   │ ChromaDB      │ │ + RAG        │
                              └─────┬──────┘   └───────────────┘ └──────────────┘
                                    │
                    ┌───────────────▼────────────────┐
                    │ Upstream data providers         │
                    │ CoinGecko·Finnhub·yfinance·FRED  │
                    │ NewsAPI·Binance·Deribit·SEC·HF   │
                    └──────────────────────────────────┘
```

---

## 2. Frontend

- Static HTML pages styled by a design-token CSS system (`css/tokens.css` is the
  single source of theme truth) and driven by vanilla ES-module JavaScript.
- Served by `live-server` on port 3000 in development, with an `--ignorePattern`
  that prevents the dev server from exposing `.db`, `.log`, `backend/`, and
  `scripts/`.
- Talks to the backend on port 8000 over HTTP and Server-Sent Events (for chat
  streaming).

## 3. Backend (FastAPI)

`backend/main.py` wires the application:

- **Routers** - `backend/routes/` (market data, predictions, ingestion,
  AI, backtest), seeded dataset
  (`user_api.py`), paper trading (`trading_api.py`), payments
  (`payments_api.py`), and auth (`auth.py`).
- **Middleware** - strict CORS allow-list (origins from `CORS_ORIGINS`; headers
  limited to `Authorization`, `Content-Type`, `X-Admin-Token`) and security headers
  (`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`,
  `Cache-Control`) on every response.
- **Lifecycle** - a `lifespan` handler: on startup it applies pending MySQL
  migrations (`backend/migrations/`, tracked in a `schema_migrations` table),
  initialises SQLite and ChromaDB, and starts the APScheduler jobs; on shutdown
  it stops the scheduler and closes HTTP clients.
- **Protection** - a built-in rate limiter on login and the LLM routes, auth on
  every personal or expensive route, and `X-Admin-Token` on maintenance routes.
- **Caching** - a small in-memory TTL cache (`cache.py`) fronts every upstream
  provider to respect free-tier rate limits.

### Key modules

| Module | Responsibility |
|---|---|
| `config.py` | Pydantic settings sourced from `.env` |
| `routes/` | `market`, `predict`, `ingestion`, `ai`, `backtest` routers + shared `common.py` |
| `llm.py` | One client for Groq (OpenAI-compatible) or local Ollama |
| `demo_seed.py` | Rolling demo dataset, rebuilt daily so dates stay current |
| `cache.py` | In-memory TTL cache |
| `ingestion.py` | APScheduler jobs: prices, OHLCV, news |
| `insights.py` | LLM market-insight generation cycle |
| `rag.py` | ChromaDB embedding + retrieval |
| `db.py` | Async SQLite (market, predictions, outcomes) |
| `mysql_db.py` | MySQL access for the seeded dataset |
| `auth.py` | Login / register / sessions |
| `trading_api.py` | Paper trading, watchlist, alerts |
| `payments_api.py` | Payments writes |
| `prediction/` | The machine-learning prediction agent |

## 4. Persistence

- **SQLite (`flux_market.db`)** - live price snapshots, 30-day and long-history
  OHLCV, news cache, predictions, prediction outcomes, calibration buckets, and
  verifier verdicts. Stored in the per-user app-data directory by default so the
  static server never serves it. On Render the disk is wiped on every deploy.
- **MySQL (TiDB Serverless in production)** - users, accounts, transactions,
  holdings, the paper-trading wallet, rewards. Seeded by `scripts/seed_mysql.py`;
  the demo user's data is rebuilt daily by `demo_seed.py`. Connections are pooled
  when `MYSQL_POOL_SIZE` > 0 (default 0 = one connection per query).
- **`MARKET_STORE`** - `sqlite` (default) keeps predictions and price history in
  SQLite; `mysql` moves them to MySQL so they survive a redeploy.
- **ChromaDB** - a vector store holding embedded market snapshots and news for
  Retrieval-Augmented Generation. Ships with the ONNX `all-MiniLM-L6-v2`
  embedding model.

## 5. Scheduling

APScheduler (`AsyncIOScheduler`) runs the background work:

Every interval is a setting, not a constant. The defaults suit development;
deployment widens them to stay inside the free API quotas.

| Job | Setting | Default |
|---|---|---|
| Crypto + stock price snapshots | `INGESTION_INTERVAL_MIN` | 5 min |
| Daily OHLCV refresh | `OHLCV_INTERVAL_MIN` | 30 min |
| News ingestion | `NEWS_INTERVAL_MIN` | 15 min |
| AI insight cycle | `INSIGHT_INTERVAL_MIN` | 15 min (chained) |
| Prune old ingestion logs | cron | 00:00 UTC |
| Demo dataset reseed (`DEMO_RESEED`) | cron + once 4 min after boot | 00:05 UTC |
| Long-history OHLCV append | cron | 00:10 UTC |
| Options IV snapshot | cron | 00:20 UTC |
| Prediction cycle | cron | 00:30 UTC |
| Drift retrain | cron | Sun 02:00 UTC |

That is 10 jobs; Render's log shows "APScheduler started — 10 jobs registered".

The insight cycle has no trigger of its own. It is **chained onto the market
cycle**: the market job ingests, then awaits the insight job, throttled to
`INSIGHT_INTERVAL_MIN`. The insight job reads the in-memory cache that
ingestion writes, so two independent triggers raced - on a cold start the
insight run fired against an empty cache and produced nothing. Chaining also
means the two never run concurrently, which matters on a small host.

The prediction cycle, the options snapshot and the drift retrain also get a
one-off run shortly after startup when `HEAVY_JOBS_ON_STARTUP` is true (the
default, and convenient in development). Constrained hosts set it false; their
cron triggers still fire, and `POST /ingestion/trigger/predictions` runs the
cycle on demand.

Each job records last-run metadata exposed via `/ingestion/status` and
`/health`.

## 6. AI and RAG Layer

- **LLM** - `llm.py` talks to any OpenAI-compatible endpoint when `LLM_API_KEY`
  is set (production: Groq, `openai/gpt-oss-120b`), otherwise to a local Ollama
  model (`aura`). It powers market insights, chat, news sentiment and the
  prediction verifier. When it is unreachable the pages say so instead of
  inventing output.
- **RAG** - `rag.py` embeds market data and news into ChromaDB; on a question,
  it retrieves relevant chunks and injects them into the LLM system prompt for a
  grounded answer.

## 7. Prediction Agent (`backend/prediction/`)

A three-layer hybrid documented fully in
[docs/AGENT_TRAINING.md](AGENT_TRAINING.md):

- **Layer 1 - leak-safe feature store** (`features.py`, `crypto_features.py`,
  `equity_features.py`, `sentiment_features.py`, `fred.py`, `options.py`, plus
  loaders in `datasources/`).
- **Layer 2 - signal engine** (`labeling.py`, `cv.py`, `train.py`, `regime.py`,
  `ensemble.py`, `conformal.py`, `garch.py`, `magnitude.py`, `sizing.py`,
  `portfolio.py`, `flux_x.py`; calibration lives in `train.py`).
- **Layer 3 - LLM verifier** (`agent.py`) - a one-way veto/downgrade safety
  layer over the calibrated signal.
- **Serving + flywheel** - `predict.py`, `serve.py`, `backtest.py`, `drift.py`,
  `paper.py` handle inference, logging, resolution, drift-retrain, and paper
  forward-testing.

## 8. Integration

- `mcp/flux-finance-mcp.js` exposes FLUX data through the Model Context Protocol
  for MCP-compatible clients.

## 9. Request Lifecycle (example: live crypto quotes)

1. Browser calls `GET /market/quotes/crypto`.
2. Backend checks the in-memory cache (30s TTL).
3. On a miss, it fetches CoinGecko `/coins/markets`, normalises the payload, and
   caches it.
4. The APScheduler ingestion job independently persists snapshots to SQLite and
   embeds them into ChromaDB for RAG.
5. The response returns `{ assets, source, cached, timestamp }`.
