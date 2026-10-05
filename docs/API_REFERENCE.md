# API Reference

HTTP endpoints exposed by the FastAPI backend (`backend/main.py`, the
routers in `backend/routes/` and the other included routers). Base URL in development: `http://localhost:8000`.

Interactive docs are available at `http://localhost:8000/docs` (Swagger) and
`/redoc` while the server is running.

**Access levels.** 🔒 = needs a logged-in user (`Authorization: Bearer <token>`).
🛠 = maintenance route, needs `X-Admin-Token` matching the `ADMIN_TOKEN` env
var; with `ADMIN_TOKEN` unset these always answer 403.

---

## System

| Method | Path | Description |
|---|---|---|
| GET, HEAD | `/health` | Liveness: `{"status":"ok"}`. With `X-Admin-Token`, also cache, scheduler, Chroma, ingestion, LLM provider, and which API keys are set (never the keys) |
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
| GET | `/ingestion/status` | Scheduler status and per-job last-run metadata (admin: `X-Admin-Token`) |
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
| POST | `/ai/chat` 🔒 | Non-streaming chat via the configured LLM (Groq in production, Ollama locally); roles `system`, `user`, `assistant` |
| POST | `/ai/chat/stream` 🔒 | Streaming chat (Server-Sent Events) |
| POST | `/ai/rag/query` 🔒 | RAG-grounded financial Q&A |

## Portfolio and Backtesting

| Method | Path | Description |
|---|---|---|
| GET | `/portfolio/value` 🔒 | Mark-to-market valuation in INR |
| POST | `/backtest` 🔒 | SMA-crossover backtest via yfinance; `404` with a reason when there is too little data |

## Auth (`backend/auth.py`)

| Method | Path | Description |
|---|---|---|
| POST | `/auth/register` | Create an account, returns a bearer token |
| POST | `/auth/login` | Log in, returns a bearer token (rate-limited) |
| GET | `/auth/me` 🔒 | The signed-in user |

## User data, trading and payments (`/db/*`, MySQL)

Every personal route reads the user from the bearer token; there is no
`?user_id=` override.

| Method | Path | Source | Description |
|---|---|---|---|
| GET | `/db/health` | `user_api.py` | `{"ok": true\|false}`; database name and row counts only with `X-Admin-Token` |
| GET | `/db/bootstrap` 🔒 | `user_api.py` | Transactions, accounts, portfolio, recurring, contacts, security and rewards in one call (page-load hydration) |
| GET | `/db/user`, `/db/accounts`, `/db/portfolio`, `/db/contacts`, `/db/security`, `/db/rewards` 🔒 | `user_api.py` | The pieces of the bootstrap payload, one per route |
| GET | `/db/faqs`, `/db/careers`, `/db/team` | `user_api.py` | Static page content |
| GET | `/db/market/catalog`, `/db/market/snapshots`, `/db/market/insights`, `/db/market/news`, `/db/market/predictions` | `user_api.py` | Market data mirrored in MySQL |
| GET, POST | `/db/transactions` 🔒 | GET `user_api.py`, POST `payments_api.py` | Ledger (`?limit`, `?category`); POST records a payment |
| GET, POST | `/db/recurring` 🔒 | GET `user_api.py`, POST `payments_api.py` | Recurring payments |
| POST | `/db/accounts/activate`, `/db/rewards/claim`, `/db/security/toggle` 🔒 | `payments_api.py` | Payments-page actions |
| GET | `/db/wallet` 🔒 | `trading_api.py` | Paper-trading cash and holdings |
| GET, POST | `/db/trades` 🔒 | `trading_api.py` | Trade history; POST fills at the server's own quote and locks the wallet row |
| GET, POST, DELETE | `/db/watchlist`, `/db/watchlist/{symbol}` 🔒 | `trading_api.py` | Watchlist |
| GET, POST, DELETE | `/db/alerts`, `/db/alerts/{id}`, `POST /db/alerts/{id}/triggered` 🔒 | `trading_api.py` | Price alerts |

---

## Response Conventions

- Market endpoints return `{ assets|candles|..., source, cached, timestamp }`.
- Errors use standard HTTP status codes: `400` invalid input, `401` not logged
  in, `403` admin token missing, `404` not found, `422` failed validation, `429`
  rate-limited (with `Retry-After`), `502` upstream provider failure, `503` a
  required key/service is unavailable.
- Error messages never include provider responses or exception text; those go
  to the server log.
- Timestamps are ISO-8601 UTC.
- AI endpoints return an explicit `available: false` / `unavailable` state
  rather than fabricating output when the LLM is offline.
