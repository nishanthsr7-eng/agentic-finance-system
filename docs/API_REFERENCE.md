# API Reference

HTTP endpoints exposed by the FastAPI backend (`backend/main.py` and the
included routers). Base URL in development: `http://localhost:8000`.

Interactive docs are available at `http://localhost:8000/docs` (Swagger) and
`/redoc` while the server is running.

**Access levels.** 🔒 = needs a logged-in user (`Authorization: Bearer <token>`).
🛠 = maintenance route, needs `X-Admin-Token` matching the `ADMIN_TOKEN` env
var; with `ADMIN_TOKEN` unset these always answer 403.

---

## System

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness plus cache, scheduler, Chroma, ingestion, Ollama, and key status |
| POST | `/cache/flush` 🛠 | Invalidate the crypto/stock quote caches |

## Live Market Data

| Method | Path | Description |
|---|---|---|
| GET | `/market/quotes/crypto` | Top-15 cryptocurrencies (CoinGecko, cached 30s) |
| GET | `/market/quotes/stocks` | 15 curated equities (Finnhub, cached 60s) |
| GET | `/market/summary` | Highlight strip: top crypto + biggest stock movers |
| GET | `/market/candles/{asset}?tf=1D\|7D\|1M\|1Y` | OHLCV candles (btc, eth, spy); unknown asset → 404 |
| GET | `/market/indicators/{symbol}` | Volume(14d), RSI(14), MACD(12,26,9) |
| GET | `/market/news?q=` | Finance headlines; `q` filters by asset |

## Persisted Market Data (SQLite)

| Method | Path | Description |
|---|---|---|
| GET | `/data/snapshots?limit=` | Latest price snapshots |
| GET | `/data/history/{symbol}?limit=` | Price history for a symbol |
| GET | `/data/ohlcv/{symbol}?days=` | 30-day daily OHLCV |
| GET | `/data/news?limit=` | Recent cached news articles |

## Ingestion Control

| Method | Path | Description |
|---|---|---|
| GET | `/ingestion/status` | Scheduler status and per-job last-run metadata |
| POST | `/ingestion/trigger/{job}` 🛠 | Trigger a job: `crypto`, `stocks`, `ohlcv`, `news`, `market`, `insights`, `predictions` |

## Prediction Agent

| Method | Path | Description |
|---|---|---|
| GET | `/predict/{symbol}?fresh=false` | Latest calibrated prediction (stored, or `fresh=true` 🛠 to recompute) |
| GET | `/predict/{symbol}/history?limit=` | Past predictions joined with realised outcomes |
| GET | `/predict/{symbol}/forecast?lookback=` | Chart bundle: close series + prediction point + conformal band |
| POST | `/predict/{symbol}/verify` 🔒 | Run the LLM verifier (downgrade/veto only) |
| GET | `/predict/{symbol}/verdict` | Latest persisted verifier verdict for a symbol |
| GET | `/predict/verdicts?limit=` | Latest verifier verdict per symbol |
| GET | `/predict/leaderboard?limit=` | Latest prediction per symbol, ranked by calibrated confidence |
| GET | `/predict/calibration?model=live` | Realised hit-rate per confidence bucket (reliability curve) |
| POST | `/predict/run?symbol=` 🛠 | Generate + persist predictions now; resolve matured ones |

## AI Insights, Chat, RAG

| Method | Path | Description |
|---|---|---|
| GET | `/ai/insights?limit=&symbol=` | Latest pre-computed AI insights |
| POST | `/ai/insights/refresh` 🛠 | Trigger an immediate insight cycle |
| GET | `/ai/intel/{symbol}` 🔒 | Structured per-asset analysis (JSON) |
| POST | `/ai/chat` 🔒 | Non-streaming chat via Ollama |
| POST | `/ai/chat/stream` 🔒 | Streaming chat (Server-Sent Events) |
| POST | `/ai/rag/query` 🔒 | RAG-grounded financial Q&A |

## Portfolio and Backtesting

| Method | Path | Description |
|---|---|---|
| GET | `/portfolio/value` | Mark-to-market valuation in INR (auth required) |
| POST | `/backtest` 🔒 | SMA-crossover backtest via yfinance |

## Routers (included)

| Router | Source | Surface |
|---|---|---|
| Seeded dataset | `user_api.py` | `/db/*` data for user-facing pages |
| Paper trading | `trading_api.py` | wallet, trades, watchlist, alerts |
| Payments | `payments_api.py` | transactions, recurring, rewards |
| Auth | `auth.py` | login, register, session/me |

---

## Response Conventions

- Market endpoints return `{ assets|candles|..., source, cached, timestamp }`.
- Errors use standard HTTP status codes: `400` invalid input, `404` not found,
  `502` upstream provider failure, `503` a required key/service is unavailable.
- Timestamps are ISO-8601 UTC.
- AI endpoints return an explicit `available: false` / `unavailable` state
  rather than fabricating output when the LLM is offline.
