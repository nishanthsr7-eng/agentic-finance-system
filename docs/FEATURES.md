# Features

A complete catalogue of FLUX capabilities, grouped by domain.

---

## 1. Live Market Data

- **Crypto quotes** - top 15 cryptocurrencies by market cap via CoinGecko,
  including 7-day sparklines, 24h change, and market capitalisation.
- **Equity quotes** - 15 curated large-cap stocks via Finnhub, fetched
  concurrently with sector metadata and logos.
- **OHLCV candles** - interactive candlestick data (1D / 7D / 1M / 1Y) for
  BTC, ETH and SPY via yfinance.
- **Technical indicators** - server-computed Volume (14d avg), RSI(14) with
  Wilder smoothing, and MACD(12,26,9) for any supported symbol.
- **In-memory caching** - per-asset TTL caches (crypto 30s, stocks 60s, news
  300s) reduce upstream calls and respect free-tier rate limits.

## 2. Background Ingestion Engine

Powered by APScheduler (`AsyncIOScheduler`):

- **Price snapshots** - crypto + stock snapshots persisted to SQLite every 5
  minutes.
- **Daily OHLCV** - 30-day daily bars refreshed every 30 minutes.
- **News ingestion** - finance/crypto headlines from NewsAPI every 15 minutes,
  filtered to reputable finance domains.
- **Status + manual triggers** - every job reports last-run metadata; jobs can
  be triggered on demand via the API.

## 3. AI Insights and RAG

- **Bulk insight generation** - pre-computed, LLM-authored insights for top
  movers each ingestion cycle.
- **Retrieval-Augmented Generation** - market snapshots and news are embedded
  into ChromaDB; user questions retrieve relevant context that is injected into
  the LLM prompt for grounded answers.
- **Streaming chat** - Server-Sent-Events chat endpoint backed by the configured
  LLM (Groq in production, Ollama locally).
- **Per-asset intel** - structured JSON analysis (consensus, confidence,
  volume profile, catalysts) with an honest "unavailable" state when the model
  is offline.

## 4. Prediction Agent (FLUX-X)

What is served:

- **Calibrated direction classifier** - one pooled XGBoost model over 29 crypto
  and US-stock symbols and 42 causal price, technical and market features, with
  isotonic probability calibration.
- **Triple-barrier + meta-labeling** - labels reflect tradeable moves;
  a second model decides whether to *act* on the primary call.
- **Leak-safe methodology** - fractional differentiation, purged + embargoed
  walk-forward cross-validation, sample-uniqueness weighting.
- **Conformal prediction bands** - 80% and 90% return ranges scaled by
  GARCH(1,1) volatility, covering 80.0% / 90.0% out of fold.
- **Regime detection** - Gaussian HMM (trend / chop / risk-off) shown on the
  Advisor and used to scale position size.
- **News tilt** - live headline sentiment nudges confidence (LLM in
  production, FinBERT locally).
- **Fractional-Kelly sizing** - ¼-Kelly from the calibrated probability.
- **LLM verifier** - a one-way safety layer that can downgrade or veto a signal
  against fresh news but can never raise confidence above the calibrated number.
- **Self-correcting flywheel** - predictions are stored, graded at t+5,
  feed the live track record, and a weekly drift check can retrain.

Built and tested, but switched off because they did not beat the plain model out
of sample: per-asset-class experts, FRED macro, crypto-native and fundamentals
feature blocks, sentiment as a feature, and the regime-conditional ensemble.

See [docs/AGENT_TRAINING.md](AGENT_TRAINING.md) for the full algorithm and each
gate's result.

## 5. Portfolio and Trading

- **Mark-to-market valuation** - joins user holdings with live yfinance prices
  and the live USD/INR rate; returns per-holding value, cost basis, weight, and
  unrealised P&L.
- **Paper trading** - virtual wallet and trades (paper only; never live
  execution). The server fills every order at its own quote and locks the wallet
  row, so the client can't pick the price or double-spend.
- **Watchlist and alerts** - track symbols and configure price alerts.
- **SMA-crossover backtester** - runs a configurable moving-average crossover
  strategy via yfinance and reports total return, Sharpe, max drawdown, win
  rate, and the equity curve.

## 6. Payments and Wallet

- Transactions, recurring payments, rewards, account switching, and credit
  limits, persisted to MySQL.

## 7. Accounts and Security

- **Authentication** - register / login with hashed passwords and signed bearer
  tokens; login and sign-up are rate-limited. A shared demo account is one click
  from the login page, and its data rolls forward every day.
- **Hardened HTTP responses** - `X-Content-Type-Options`, `X-Frame-Options`,
  `Referrer-Policy`, and `Cache-Control` headers on every API response; a
  strict, non-wildcard CORS allow-list; error messages never echo provider or
  exception text.

## 8. Developer and Integration Surface

- **Model Context Protocol server** (`mcp/flux-finance-mcp.js`) exposing FLUX
  data to MCP-compatible clients.
- **Reproducible training** - versioned training/eval "gate" scripts under
  `scripts/experiments/` that must pass measurable out-of-sample checks before a
  component ships.
