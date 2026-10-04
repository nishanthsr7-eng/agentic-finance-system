# Features

A complete catalogue of FLUX capabilities, grouped by domain.

---

## 1. Live Market Data

- **Crypto quotes** — top 15 cryptocurrencies by market cap via CoinGecko,
  including 7-day sparklines, 24h change, and market capitalisation.
- **Equity quotes** — 15 curated large-cap stocks via Finnhub, fetched
  concurrently with sector metadata and logos.
- **OHLCV candles** — interactive candlestick data (1D / 7D / 1M / 1Y) for
  BTC, ETH and SPY via yfinance.
- **Technical indicators** — server-computed Volume (14d avg), RSI(14) with
  Wilder smoothing, and MACD(12,26,9) for any supported symbol.
- **In-memory caching** — per-asset TTL caches (crypto 30s, stocks 60s, news
  300s) reduce upstream calls and respect free-tier rate limits.

## 2. Background Ingestion Engine

Powered by APScheduler (`AsyncIOScheduler`):

- **Price snapshots** — crypto + stock snapshots persisted to SQLite every 5
  minutes.
- **Daily OHLCV** — 30-day daily bars refreshed every 30 minutes.
- **News ingestion** — finance/crypto headlines from NewsAPI every 15 minutes,
  filtered to reputable finance domains.
- **Status + manual triggers** — every job reports last-run metadata; jobs can
  be triggered on demand via the API.

## 3. AI Insights and RAG

- **Bulk insight generation** — pre-computed, LLM-authored insights for top
  movers each ingestion cycle.
- **Retrieval-Augmented Generation** — market snapshots and news are embedded
  into ChromaDB; user questions retrieve relevant context that is injected into
  the LLM prompt for grounded answers.
- **Streaming chat** — Server-Sent-Events chat endpoint backed by Ollama.
- **Per-asset intel** — structured JSON analysis (consensus, confidence,
  volume profile, catalysts) with an honest "unavailable" state when the model
  is offline.

## 4. Prediction Agent (FLUX-X)

A regime-conditional, asset-class-specialised machine-learning engine:

- **Calibrated direction classifier** — pooled cross-sectional XGBoost over
  stationary features, with isotonic probability calibration.
- **Triple-barrier + meta-labeling** — labels reflect tradeable moves;
  a second model decides whether to *act* on the primary call.
- **Leak-safe methodology** — fractional differentiation, purged + embargoed
  walk-forward cross-validation, sample-uniqueness weighting.
- **Conformal prediction bands** — distribution-free, guaranteed-coverage
  intervals, shaped by GARCH(1,1) volatility.
- **Regime detection** — Gaussian HMM (trend / chop / risk-off) used as a
  feature, an expert selector, and a sizing gate.
- **Orthogonal data blocks** — crypto funding/open-interest/DVOL/on-chain;
  equity macro spreads, options IV/skew, and fundamentals.
- **Multi-source sentiment** — FinBERT (equity) and CryptoBERT (crypto) routed
  by asset class, plus GDELT tone.
- **Cross-sectional portfolio** — rank, market/sector-neutralise, vol-target,
  and cost-aware fractional-Kelly sizing.
- **LLM verifier** — a one-way safety layer that can downgrade or veto a signal
  against fresh news/RAG but can never raise confidence above the calibrated
  number.
- **Self-correcting flywheel** — predictions are logged, resolved against
  outcomes, recalibrated, and drift-triggered retraining is wired in.

See [docs/AGENT_TRAINING.md](AGENT_TRAINING.md) for the full algorithm.

## 5. Portfolio and Trading

- **Mark-to-market valuation** — joins user holdings with live yfinance prices
  and the live USD/INR rate; returns per-holding value, cost basis, weight, and
  unrealised P&L.
- **Paper trading** — wallet, trades, and order simulation (paper only; never
  live execution).
- **Watchlist and alerts** — track symbols and configure price alerts.
- **SMA-crossover backtester** — runs a configurable moving-average crossover
  strategy via yfinance and reports total return, Sharpe, max drawdown, win
  rate, and the equity curve.

## 6. Payments and Wallet

- Transactions, recurring payments, rewards, account switching, and credit
  limits, persisted to MySQL.

## 7. Accounts and Security

- **Authentication** — register / login / session management with hashed
  credentials.
- **Privacy Shield** — privacy-and-security controls surface.
- **Hardened HTTP responses** — `X-Content-Type-Options`, `X-Frame-Options`,
  `Referrer-Policy`, and `Cache-Control` headers on every API response; a
  strict, non-wildcard CORS allow-list.

## 8. Developer and Integration Surface

- **Model Context Protocol server** (`mcp/flux-finance-mcp.js`) exposing FLUX
  data to MCP-compatible clients.
- **Reproducible training** — versioned training/eval "gate" scripts under
  `scripts/` that must pass measurable out-of-sample checks before a component
  ships.
