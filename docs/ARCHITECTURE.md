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
                              │ ingestion  │   │ SQLite · MySQL│ │ Ollama (aura)│
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

- **Routers** — `backend/routes/` (market data, predictions, ingestion,
  AI, backtest), seeded dataset
  (`user_api.py`), paper trading (`trading_api.py`), payments
  (`payments_api.py`), and auth (`auth.py`).
- **Middleware** — strict CORS allow-list and security headers
  (`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`,
  `Cache-Control`) on every response.
- **Lifecycle** — a `lifespan` handler: on startup it ensures all schemas, initialises SQLite and
  ChromaDB, and starts the APScheduler jobs; on shutdown it stops the scheduler
  and closes HTTP clients.
- **Caching** — a small in-memory TTL cache (`cache.py`) fronts every upstream
  provider to respect free-tier rate limits.

### Key modules

| Module | Responsibility |
|---|---|
| `config.py` | Pydantic settings sourced from `.env` |
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

- **SQLite (`flux_market.db`)** — live price snapshots, 30-day and long-history
  OHLCV, news cache, predictions, prediction outcomes, calibration buckets, and
  verifier verdicts. Stored in the per-user app-data directory by default so the
  static server never serves it.
- **MySQL (`flux`)** — the seeded "ultimate" application dataset that backs the
  user-facing pages (accounts, transactions, holdings, rewards). Seeded by
  `seed_mysql.py`.
- **ChromaDB** — a vector store holding embedded market snapshots and news for
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
| Prediction cycle, options snapshot, drift retrain | cron | 00:30 / 00:20 / Sun 02:00 UTC |

The insight cycle has no trigger of its own. It is **chained onto the market
cycle**: the market job ingests, then awaits the insight job, throttled to
`INSIGHT_INTERVAL_MIN`. The insight job reads the in-memory cache that
ingestion writes, so two independent triggers raced — on a cold start the
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

- **Ollama** serves the local `aura` model used for market insights, the chat
  endpoints, and the prediction verifier. The system degrades honestly when
  Ollama is offline.
- **RAG** — `rag.py` embeds market data and news into ChromaDB; on a question,
  it retrieves relevant chunks and injects them into the LLM system prompt for a
  grounded answer.

## 7. Prediction Agent (`backend/prediction/`)

A three-layer hybrid documented fully in
[docs/AGENT_TRAINING.md](AGENT_TRAINING.md):

- **Layer 1 — leak-safe feature store** (`features.py`, `crypto_features.py`,
  `equity_features.py`, `sentiment_features.py`, `fred.py`, `options.py`, plus
  loaders in `datasources/`).
- **Layer 2 — signal engine** (`labeling.py`, `cv.py`, `train.py`, `regime.py`,
  `ensemble.py`, `conformal.py`, `garch.py`, `magnitude.py`, `calibration`,
  `sizing.py`, `portfolio.py`, `flux_x.py`).
- **Layer 3 — LLM verifier** (`agent.py`) — a one-way veto/downgrade safety
  layer over the calibrated signal.
- **Serving + flywheel** — `predict.py`, `serve.py`, `backtest.py`, `drift.py`,
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
