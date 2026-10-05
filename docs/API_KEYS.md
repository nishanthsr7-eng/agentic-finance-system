# API Keys and Settings

Every key and setting FLUX reads, where to get it, and what breaks without it.
Values go into `.env` in the project root locally (read by `backend/config.py`)
or into the Render environment in production. Every market key is optional: a
missing key turns its feature off instead of crashing the app.

> Security: never commit `.env`, never paste real keys into documentation or
> source, and rotate any key that has been exposed.

---

## Used by the running app

| Variable | Used for | Without it | Get it |
|---|---|---|---|
| `FINNHUB_API_KEY` | Stock quotes, earnings dates (the earnings gate) | Stock panels return 503 | <https://finnhub.io/register>, free (60 req/min) |
| `COINGECKO_API_KEY` | Top-15 crypto quotes, market caps, sparklines | Works keyless at a lower rate limit | <https://www.coingecko.com/en/api>, Demo plan (sent as `x-cg-demo-api-key`) |
| `NEWSAPI_KEY` | Headline ticker, per-asset news, news for sentiment | News panels return 503 | <https://newsapi.org/register>, free developer tier |
| `ALPHA_VANTAGE_API_KEY` | Extra news-sentiment source | Source skipped | <https://www.alphavantage.co/support/#api-key>, free (25 req/day) |
| `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` | Subreddit sentiment | Source skipped | <https://www.reddit.com/prefs/apps> → "script" app |
| `LLM_API_KEY` | Insights, chat, news sentiment, the verifier (production: Groq) | Falls back to local Ollama | <https://console.groq.com/keys>, free |
| `LLM_BASE_URL`, `LLM_MODEL` | OpenAI-compatible endpoint and model | Defaults to Groq and `openai/gpt-oss-120b` | - |
| `OLLAMA_URL`, `OLLAMA_MODEL` | Local LLM when `LLM_API_KEY` is empty | AI panels show "unavailable" | Defaults `http://localhost:11434`, `aura` |
| `SENTIMENT_BACKEND` | `auto` (FinBERT/CryptoBERT if PyTorch is installed, else LLM), `transformers`, or `llm` | `auto` | Production pins `llm` (PyTorch doesn't fit in 512 MB) |

## Security settings

| Variable | Used for | Default |
|---|---|---|
| `AUTH_SECRET` | Signs login tokens. Set a long random value in production | Empty: generated once and saved to `auth_secret.key` in the app-data directory; on Render that file is lost on every deploy, logging everyone out |
| `AUTH_REQUIRED` | `false` treats anonymous requests as the demo user (local only) | `true` |
| `ADMIN_TOKEN` | `X-Admin-Token` for maintenance routes and `/db/health` details. Not in `render.yaml`; add it in the Render dashboard | Empty: those routes always answer 403 |
| `CORS_ORIGINS` | Frontend origins allowed to call the API | `localhost:3000` and `127.0.0.1:3000` |

## Databases and storage

| Variable | Used for | Default |
|---|---|---|
| `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DB` | User data (TiDB Serverless in production) | `127.0.0.1:3306`, db `flux` |
| `MYSQL_SSL`, `MYSQL_SSL_CA` | TLS to MySQL (required by TiDB) | Off |
| `MYSQL_POOL_SIZE` | Reuse connections; `0` opens one per query | `0` |
| `MARKET_STORE` | `sqlite` or `mysql` for predictions and price history | `sqlite` |
| `DB_PATH`, `CHROMA_PATH` | SQLite and ChromaDB locations | Per-user app-data directory |
| `DEMO_RESEED` | Rebuild the demo user's data daily | `true` |

Scheduler intervals (`INGESTION_INTERVAL_MIN`, `OHLCV_INTERVAL_MIN`,
`NEWS_INTERVAL_MIN`, `INSIGHT_INTERVAL_MIN`), `INGESTION_ENABLED`,
`HEAVY_JOBS_ON_STARTUP` and `RAG_ENABLED` are described in
[docs/ARCHITECTURE.md](ARCHITECTURE.md) and [docs/DEPLOYMENT.md](DEPLOYMENT.md).

---

## Used only for training and research

| Variable | Used for | Get it |
|---|---|---|
| `FRED_API_KEY` | FRED macro series for training and the gate-1 experiments (off in the served model) | <https://fredaccount.stlouisfed.org/apikeys>, free |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | Read by the Kaggle CLI to download the Huge Stock Market dataset | <https://www.kaggle.com/settings> → "Create New Token" |
| `HF_TOKEN` | Optional; read by Hugging Face tools. FinBERT, CryptoBERT and Financial PhraseBank are public | <https://huggingface.co/settings/tokens> |

## Paper trading (never live)

| Variable | Used for |
|---|---|
| `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` | Optional forward test on an Alpaca **paper** account (`backend/prediction/paper.py`). `ALPACA_PAPER` defaults to `true` and there is no live-trading code path. Not set in the deployment |

---

## Minimum configurations

| Goal | Set |
|---|---|
| Local app with live prices | `FINNHUB_API_KEY`, `NEWSAPI_KEY` (CoinGecko works keyless) + MySQL |
| AI features | the above + `LLM_API_KEY`, or Ollama running locally |
| Production (Render) | the above + `AUTH_SECRET`, `ADMIN_TOKEN`, `CORS_ORIGINS`, `MYSQL_SSL`, `SENTIMENT_BACKEND=llm` |
| Retraining | + `FRED_API_KEY`, plus the datasets in [docs/DATASETS.md](DATASETS.md) |
