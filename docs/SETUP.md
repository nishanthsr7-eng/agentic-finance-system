# Setup Guide

Complete instructions for installing and running FLUX locally on Windows,
macOS, or Linux.

---

## 1. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10 or newer | Backend and prediction agent |
| Node.js | 18 or newer | Frontend dev server (`live-server`) |
| MySQL | 8.0 or newer, or a free TiDB Serverless cluster | User data for the app pages |
| Git | any | Cloning and version control |

Optional:
- An LLM for the AI features: a free Groq key (`LLM_API_KEY`), or
  [Ollama](https://ollama.com) running locally.
- A C/C++ build toolchain if a wheel for `xgboost` or `hmmlearn` isn't
  available for your platform.

---

## 2. Clone and Install

### 2.1 Frontend

```bash
npm install
```

This installs `live-server`, the only frontend dependency.

### 2.2 Backend

Create and activate a virtual environment, then install Python dependencies:

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

pip install -r backend/requirements-slim.txt   # what production runs, no PyTorch
# or, for FinBERT sentiment, the Chronos baseline and the LSTM magnitude head:
pip install -r backend/requirements.txt        # adds PyTorch (~1 GB)
```

---

## 3. Configure Secrets (`.env`)

Copy the template, `cp .env.example .env` (Windows: `copy .env.example .env`),
and fill in what you have. The backend reads it through `backend/config.py`
(Pydantic settings). See [docs/API_KEYS.md](API_KEYS.md) for every setting. The
minimum is:

```dotenv
# ── Market data ───────────────────────────────
FINNHUB_API_KEY=your_key_here
COINGECKO_API_KEY=your_key_here     # optional, works keyless at a lower limit
NEWSAPI_KEY=your_key_here

# ── LLM (pick one) ────────────────────────────
LLM_API_KEY=your_groq_key           # Groq; leave empty to use Ollama
# OLLAMA_URL=http://localhost:11434
# OLLAMA_MODEL=aura

# ── MySQL ─────────────────────────────────────
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DB=flux
# MYSQL_SSL=true                    # needed for TiDB Serverless
```

> Every market key degrades gracefully: if a key is absent, the dependent
> feature is disabled rather than crashing the app. The full list of settings,
> including the optional ones, is in [docs/API_KEYS.md](API_KEYS.md).

**Never commit `.env`.** It is excluded by `.gitignore`.

---

## 4. Database Setup

### 4.1 SQLite (automatic)

The live market database (`flux_market.db`) is created automatically on first
backend startup. By default it is stored in the per-user app-data directory
(`%LOCALAPPDATA%\flux` on Windows, `~/.local/share/flux` elsewhere) so the dev
static server never exposes it publicly. Override with `DB_PATH` in `.env`.

### 4.2 MySQL (seeded application dataset)

The app pages read from MySQL. One command creates the database named in
`MYSQL_DB`, every table (including the `backend/migrations/` steps) and the
seed data with the demo user:

```bash
python scripts/seed_mysql.py
```

The API also applies any pending migrations on startup.

The demo user's transactions are rebuilt every day at 00:05 UTC (and once
shortly after startup) so its dates stay current; other users are untouched.

---

## 5. LLM (optional)

The insights, chat, RAG, news sentiment and prediction verifier call an LLM.
With `LLM_API_KEY` set they use Groq (as production does) and you can skip
this section. Otherwise they call a local Ollama model named `aura`:

```bash
# Install Ollama from https://ollama.com, then:
ollama serve

# The custom model is defined in ai_engine/models/Modelfile
ollama create aura -f ai_engine/models/Modelfile
```

If no LLM is reachable, the app degrades honestly - AI panels show an
"unavailable" state and live prices/headlines continue to work.

---

## 6. Run the Application

Open two terminals from the project root.

**Terminal 1 - Backend API (port 8000):**

```bash
uvicorn backend.main:app --reload --port 8000
```

On Windows you can also double-click `scripts\start-backend.bat`.

**Terminal 2 - Frontend (port 3000):**

```bash
npm run dev
```

Visit <http://localhost:3000> and click "Use demo account". Verify the backend
at <http://localhost:8000/health> - the response reports cache, scheduler,
Chroma, ingestion, and LLM status.

The trained models ship in `backend/prediction/models/`, so forecasts work
without training. Locally `HEAVY_JOBS_ON_STARTUP=true` runs the prediction cycle
shortly after startup; after that it runs daily at 00:30 UTC.

---

## 7. Optional - Retrain the Prediction Agent

Only needed to change the model. Download the training data and retrain:

```bash
# 1. Download datasets (see docs/DATASETS.md for all sources)
python scripts/dl_binance.py
python scripts/dl_deribit.py
python scripts/dl_coinmetrics.py

# 2. Backfill years of daily OHLCV into the ohlcv_history table
python scripts/backfill_history.py

# 3. Train + calibrate + persist the models
python backend/prediction/train.py
```

The full pipeline, gates, and algorithms are documented in
[docs/AGENT_TRAINING.md](AGENT_TRAINING.md).

---

## 8. Running Tests

```bash
pytest backend/prediction backend/tests
ruff check .
ruff format --check .
```

`backend/prediction/tests/` covers features, labeling, cross-validation
leakage, calibration, the ensemble and portfolio construction.
`backend/tests/` covers the API: auth and admin guards, rate limits, trade
pricing, the demo seed, migrations and error handling. CI runs the same
commands plus a Docker build and the static-site build.

---

## 9. Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `FINNHUB_API_KEY not configured` (503) | Add the key to `.env` and restart Uvicorn. |
| Stocks panel empty | Markets closed, or Finnhub free-tier rate limit (60 req/min). |
| "The AI service is unavailable" (503) | Set `LLM_API_KEY`, or run `ollama serve` with the `aura` model. The server log has the exact error. |
| MySQL connection refused | Confirm the server is running and `.env` credentials match. |
| Advisor says "No forecast for X yet" | The prediction cycle hasn't run yet. Wait for it, or call `POST /predict/run` with `X-Admin-Token`. |
| Frontend serves `.db`/`.log` files | The `live-server` `--ignorePattern` excludes them; do not move the DB into the project root. |
