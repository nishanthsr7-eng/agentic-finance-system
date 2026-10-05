# Pages

Every page in the FLUX frontend and the features it provides. Pages live in the
project root (`index.html`) and the `pages/` directory. Styling is driven by the
design-token system in `css/`, and behaviour by the ES modules in `js/`.

---

## Landing - `index.html`

The marketing entry point and product overview.

- Hero section, feature highlights and navigation into the app.
- A phone preview of the app with three screens (practice wallet, holdings and
  a practice buy order), switched by the Forecast / Check / Practise control.
- A compact footer shared by every public page: a demo call-to-action, then
  About, Security, FAQs, Contact, Privacy and Terms.
- Static page; driven by `js/main.js`.

## Dashboard - `pages/dashboard.html`

The signed-in home page.

- Daily briefing on the top mover, portfolio value, asset allocation, savings
  score, income/spend cards and a cashflow chart, from the user's own data
  (`GET /db/bootstrap`, one call for everything).
- Live crypto and stock quotes for the briefing strip
  (`/market/quotes/crypto`, `/market/quotes/stocks`).
- First-time users get an account-setup overlay (`js/account-setup.js`); the
  demo account skips it.
- Driven by `js/dashboard.js`, `js/flux-data.js`.

## Marketplace - `pages/marketplace.html`

Quotes and paper trading.

- Live crypto and stock quotes with sparklines, a headline ticker and a
  screener (`/market/quotes/*`, `/market/summary`, `/market/news`).
- Per-asset drawer: AI intel (`/ai/intel/{symbol}`) and a terminal chart with
  RSI / MACD / volume (`/market/indicators/{symbol}`).
- Order ticket: market or limit, BUY or SELL, filled at the server's quote;
  SELL is capped at the holding (`/db/trades`, `/db/wallet`).
- Watchlist and price alerts (`/db/watchlist`, `/db/alerts`).
- Driven by `js/marketplace.js`, `js/market-live.js`.

## Analysis - `pages/analysis.html`

Charts, spending and strategy testing.

- Candlestick chart for BTC, ETH or SPY (1D / 7D / 1M / 1Y) via
  `GET /market/candles/{asset}`.
- Cashflow heatmap of the last 30 days, the Live Ledger of the user's paper
  trades (`/db/trades`), asset allocation (`/portfolio/value`) and LLM market
  commentary (`/ai/insights`).
- SMA-crossover backtester (`POST /backtest`) reporting total return, Sharpe,
  max drawdown, win rate and the equity curve; the demo account runs a default
  backtest on load.
- Driven by `js/analysis.js`.

## Smart Advisor - `pages/advisor.html`

The window into the prediction agent.

- Chart of recent prices (live candles for BTC/ETH, 60 daily candles for the
  rest) with the 5-day forecast drawn as a cone from the current price out to
  the likely range on the target date (`GET /predict/{symbol}/forecast`).
  Hovering the forecast zone shows the likely range for that day.
- A summary above the chart: the call ("Likely up", "Likely down", or "Flat"
  when the expected move is under 1%), the likely range, confidence, market
  regime and the news check (`/predict/{symbol}/verdict`; "Check again"
  re-runs `/predict/{symbol}/verify`), plus suggested size and track record
  (`/predict/{symbol}/history`).
- "All forecasts" table for every symbol (`/predict/leaderboard`,
  `/predict/verdicts`). The selected symbol uses the same live range as the
  summary.
- Advisor chat (`/ai/chat/stream`).
- Driven by `js/advisor.js` and `css/advisor.css`.

## Payments - `pages/payments.html`

Simulated money movement.

- Transactions, recurring payments, contacts, rewards, account switching and a
  spending meter (`backend/payments_api.py`).
- Driven by `js/payments.js`.

## Login / Sign up - `pages/login.html`, `pages/signin.html`

Log in, "Use demo account", or register (`backend/auth.py`). After login the
page prefetches `/db/bootstrap` so the dashboard opens filled. Password reset
is not available; the page says so.

## Info pages

`about`, `brand`, `security`, `shield` (Privacy Shield), `careers`, `contact`,
`faq`, `terms`, `privacy` and `cookies` in `pages/`, plus `404.html`. They complete the product site with placeholder
copy for a fictional company. Careers and FAQ are filled from MySQL
(`/db/careers`, `/db/faqs`); the contact form does not send anything.

---

## Shared Frontend Modules

| Module | Responsibility |
|---|---|
| `js/flux-config.js` | API base URL: `localhost:8000` locally, the Render URL in production |
| `js/flux-data.js` | API client; loads the user's data via `/db/bootstrap` |
| `js/seed.js` | Offline fallback data, shown with a "Showing sample data" badge when the API is unreachable |
| `js/account-setup.js` | First-login setup overlay |
| `js/main.js` | Landing-page behaviour |
| `js/market-live.js` | Live quote polling and rendering (Marketplace) |
| `css/tokens.css` | Design tokens (the single source of theme truth) |
| `css/base.css`, `css/layout-extensions.css` | Base layout and structure |
