"""Market quotes, candles, indicators, news, persisted data and portfolio value."""

import asyncio
from typing import Any

import httpx
import yfinance as yf
from fastapi import APIRouter, Depends, HTTPException, Query

from ..auth import require_admin, require_user
from ..cache import cache
from ..config import settings
from ..db import (
    get_latest_snapshots,
    get_ohlcv,
    get_recent_news,
    get_symbol_history,
)
from .common import _now_iso, _wrap, get_client, log

router = APIRouter()


# ── Asset Catalogues ──────────────────────────────────────────────────────────

# CoinGecko IDs — ranked by market cap
CRYPTO_IDS = [
    "bitcoin",
    "ethereum",
    "tether",
    "binancecoin",
    "solana",
    "ripple",
    "dogecoin",
    "cardano",
    "avalanche-2",
    "polkadot",
    "chainlink",
    "uniswap",
    "litecoin",
    "shiba-inu",
    "tron",
]

# Stocks: symbols + static metadata (sector rarely changes)
STOCK_META: dict[str, dict] = {
    "AAPL": {"name": "Apple Inc.", "sector": "Technology"},
    "MSFT": {"name": "Microsoft Corp.", "sector": "Technology"},
    "NVDA": {"name": "NVIDIA Corp.", "sector": "Semiconductors"},
    "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology"},
    "AMZN": {"name": "Amazon.com Inc.", "sector": "Consumer"},
    "TSLA": {"name": "Tesla Inc.", "sector": "Automotive"},
    "META": {"name": "Meta Platforms", "sector": "Technology"},
    "NFLX": {"name": "Netflix Inc.", "sector": "Media"},
    "JPM": {"name": "JPMorgan Chase", "sector": "Financials"},
    "AMD": {"name": "Advanced Micro Dev.", "sector": "Semiconductors"},
    "TSM": {"name": "Taiwan Semiconductor", "sector": "Semiconductors"},
    "ORCL": {"name": "Oracle Corp.", "sector": "Technology"},
    "CRM": {"name": "Salesforce Inc.", "sector": "Software"},
    "INTC": {"name": "Intel Corp.", "sector": "Semiconductors"},
    "BABA": {"name": "Alibaba Group", "sector": "Consumer"},
}

STOCK_SYMBOLS = list(STOCK_META.keys())

# Financial Modeling Prep image CDN — purpose-built stock logos, no auth required
_FMP = "https://financialmodelingprep.com/image-stock"
STOCK_LOGOS: dict[str, str] = {
    sym: f"{_FMP}/{sym}.png"
    for sym in [
        "AAPL",
        "MSFT",
        "NVDA",
        "GOOGL",
        "AMZN",
        "TSLA",
        "META",
        "NFLX",
        "JPM",
        "AMD",
        "TSM",
        "ORCL",
        "CRM",
        "INTC",
        "BABA",
    ]
}


# ── Crypto Fetcher ────────────────────────────────────────────────────────────


async def _fetch_crypto_live() -> list[dict]:
    """Fetch top coins from CoinGecko /coins/markets."""
    params: dict[str, Any] = {
        "vs_currency": "usd",
        "ids": ",".join(CRYPTO_IDS),
        "order": "market_cap_desc",
        "per_page": len(CRYPTO_IDS),
        "page": 1,
        "sparkline": "true",
        "price_change_percentage": "24h",
    }
    headers = {}
    if settings.COINGECKO_API_KEY:
        # Demo key uses x-cg-demo-api-key header
        headers["x-cg-demo-api-key"] = settings.COINGECKO_API_KEY

    r = await get_client().get(
        "https://api.coingecko.com/api/v3/coins/markets",
        params=params,
        headers=headers,
    )
    r.raise_for_status()
    coins = r.json()

    assets = []
    for coin in coins:
        symbol_short = (coin.get("symbol") or "").upper()
        assets.append(
            {
                "symbol": f"BINANCE:{symbol_short}USDT",
                "name": coin.get("name", symbol_short),
                "sub": symbol_short,
                "price": float(coin.get("current_price") or 0),
                "change_pct": float(coin.get("price_change_percentage_24h") or 0),
                "market_cap": float(coin.get("market_cap") or 0),
                "icon": coin.get("image", ""),
                "sector": "",
                "sparkline_7d": (coin.get("sparkline_in_7d") or {}).get("price", []),
            }
        )
    log.info("CoinGecko: fetched %d coins", len(assets))
    return assets


# ── Stocks Fetcher ────────────────────────────────────────────────────────────


async def _fetch_single_stock(client: httpx.AsyncClient, symbol: str) -> dict | None:
    """Fetch a single Finnhub quote. Returns None on error."""
    try:
        r = await client.get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": symbol, "token": settings.FINNHUB_API_KEY},
        )
        if r.status_code != 200:
            return None
        q = r.json()
        price = float(q.get("c") or 0)
        if price == 0:
            return None  # market closed / bad symbol — skip
        prev = float(q.get("pc") or price)
        chg = ((price - prev) / prev * 100) if prev else 0
        meta = STOCK_META.get(symbol, {"name": symbol, "sector": "—"})
        return {
            "symbol": symbol,
            "name": meta["name"],
            "sub": symbol,
            "price": price,
            "change_pct": round(chg, 3),
            "market_cap": 0,
            "icon": STOCK_LOGOS.get(symbol, ""),
            "sector": meta["sector"],
        }
    except Exception as exc:
        log.warning("Finnhub %s failed: %s", symbol, exc)
        return None


async def _fetch_stocks_live() -> list[dict]:
    """Fetch all stock quotes concurrently from Finnhub."""
    client = get_client()
    tasks = [_fetch_single_stock(client, sym) for sym in STOCK_SYMBOLS]
    results = await asyncio.gather(*tasks)
    assets = [r for r in results if r is not None]
    log.info("Finnhub: fetched %d/%d stocks", len(assets), len(STOCK_SYMBOLS))
    return assets


@router.get("/market/summary")
async def market_summary():
    """
    Returns 4 highlight assets for the marketplace strip cards:
    top-2 cryptos by market cap, plus the 2 stocks with the biggest absolute 24h move.
    Backed by the same caches as /market/quotes/* — no extra upstream calls.
    """
    crypto = cache.get("crypto")
    if crypto is None:
        try:
            crypto = await _fetch_crypto_live()
            cache.set("crypto", crypto, ttl=settings.CRYPTO_TTL)
        except Exception:
            crypto = []

    stocks = cache.get("stocks")
    if stocks is None and settings.FINNHUB_API_KEY:
        try:
            stocks = await _fetch_stocks_live()
            if stocks:
                cache.set("stocks", stocks, ttl=settings.STOCKS_TTL)
        except Exception:
            stocks = []
    stocks = stocks or []

    by_cap = sorted(crypto, key=lambda a: a.get("market_cap", 0), reverse=True)
    top_stocks = sorted(stocks, key=lambda a: abs(a.get("change_pct", 0)), reverse=True)[:2]
    # Always four cards: when stock quotes are unavailable, the next cryptos fill the gap.
    top_crypto = by_cap[: 4 - len(top_stocks)]

    def _strip(a: dict, category: str) -> dict:
        return {
            "symbol": a.get("symbol"),
            "name": a.get("name"),
            "sub": a.get("sub"),
            "price": a.get("price"),
            "change_pct": a.get("change_pct"),
            "icon": a.get("icon"),
            "category": category,
        }

    highlights = [_strip(a, "crypto") for a in top_crypto] + [
        _strip(a, "stocks") for a in top_stocks
    ]
    return {"highlights": highlights, "timestamp": _now_iso()}


@router.post("/cache/flush", dependencies=[Depends(require_admin)])
async def cache_flush():
    """Invalidate all market quote caches so the next request fetches fresh data."""
    for key in ("crypto", "stocks"):
        cache.invalidate(key)
    return {"flushed": ["crypto", "stocks"]}


# ── §A: Persisted Market Data Endpoints ──────────────────────────────────────


@router.get("/data/snapshots")
async def data_snapshots(limit: int = Query(50, ge=1, le=500)):
    """Latest N price snapshots from SQLite (all symbols, newest first)."""
    rows = await get_latest_snapshots(limit)
    return {"snapshots": rows, "count": len(rows), "timestamp": _now_iso()}


@router.get("/data/history/{symbol}")
async def data_history(
    symbol: str,
    limit: int = Query(100, ge=1, le=1000),
):
    """Price history for a specific symbol from SQLite."""
    rows = await get_symbol_history(symbol.upper(), limit)
    return {"symbol": symbol.upper(), "history": rows, "count": len(rows), "timestamp": _now_iso()}


@router.get("/data/ohlcv/{symbol}")
async def data_ohlcv(
    symbol: str,
    days: int = Query(30, ge=1, le=365),
):
    """30-day daily OHLCV from SQLite (persisted by scheduler)."""
    rows = await get_ohlcv(symbol.upper(), days)
    return {"symbol": symbol.upper(), "ohlcv": rows, "count": len(rows), "timestamp": _now_iso()}


@router.get("/data/news")
async def data_news_cached(limit: int = Query(20, ge=1, le=100)):
    """Recent news articles from SQLite."""
    rows = await get_recent_news(limit)
    return {"articles": rows, "count": len(rows), "timestamp": _now_iso()}


@router.get("/market/quotes/crypto")
async def quotes_crypto():
    """
    Returns top-15 cryptocurrencies from CoinGecko.
    Cached for CRYPTO_TTL seconds (default 30s).
    """
    cached_data = cache.get("crypto")
    if cached_data is not None:
        return _wrap(cached_data, "coingecko_cache", cached=True)

    try:
        assets = await _fetch_crypto_live()
    except httpx.HTTPStatusError as e:
        log.error("CoinGecko HTTP error: %s", e)
        raise HTTPException(502, f"CoinGecko error: {e.response.status_code}") from e
    except Exception as e:
        log.error("CoinGecko fetch failed: %s", e)
        raise HTTPException(502, "CoinGecko unreachable") from e

    cache.set("crypto", assets, ttl=settings.CRYPTO_TTL)
    return _wrap(assets, "coingecko_live", cached=False)


@router.get("/market/quotes/stocks")
async def quotes_stocks():
    """
    Returns 15 curated equities from Finnhub.
    Cached for STOCKS_TTL seconds (default 60s).
    """
    cached_data = cache.get("stocks")
    if cached_data is not None:
        return _wrap(cached_data, "finnhub_cache", cached=True)

    if not settings.FINNHUB_API_KEY:
        raise HTTPException(503, "FINNHUB_API_KEY not configured")

    try:
        assets = await _fetch_stocks_live()
    except Exception as e:
        log.error("Finnhub fetch failed: %s", e)
        raise HTTPException(502, "Finnhub unreachable") from e

    # Don't cache an empty sweep (bad key, rate limit), so the next request retries.
    if assets:
        cache.set("stocks", assets, ttl=settings.STOCKS_TTL)
    return _wrap(assets, "finnhub_live", cached=False)


# ── Portfolio Mark-to-Market Valuation ───────────────────────────────────────


# yfinance ticker per holding: NSE equities/ETFs trade as <SYM>.NS in INR;
# crypto trades as <SYM>-USD and needs the USDINR rate applied.
def _holding_yf_symbol(h: dict) -> str:
    if h.get("asset_type") == "crypto":
        return f"{h['symbol']}-USD"
    if h.get("currency") == "USD":  # US-listed, e.g. SPY
        return h["symbol"]
    return f"{h['symbol']}.NS"


def _holding_needs_fx(h: dict) -> bool:
    """Crypto and US-listed holdings are quoted in USD and converted to INR."""
    return h.get("asset_type") == "crypto" or h.get("currency") == "USD"


def _fetch_last_prices_sync(symbols: list[str]) -> dict[str, float]:
    """Blocking yfinance batch quote — last close per symbol. Symbols that
    fail to resolve are simply absent from the result."""
    out: dict[str, float] = {}
    if not symbols:
        return out
    df = yf.download(
        symbols,
        period="5d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    for sym in symbols:
        try:
            closes = (df[sym]["Close"] if len(symbols) > 1 else df["Close"]).dropna()
            if len(closes):
                out[sym] = float(closes.iloc[-1])
        except Exception:  # noqa: BLE001 — unresolvable symbol → skip
            continue
    return out


@router.get("/portfolio/value")
async def portfolio_value(user_id: int = Depends(require_user)):
    """Mark-to-market portfolio valuation in INR.

    Joins the user's holdings with live last prices (yfinance batch) and the
    live USDINR rate for crypto. Returns per-holding market value, cost basis
    and unrealised P&L. Cached for 60s per user.
    """
    cache_key = f"pf_value_{user_id}"
    cached = cache.get(cache_key)
    if cached is not None:
        return {**cached, "cached": True}

    from .. import mysql_db as M

    holdings = M.query("SELECT * FROM portfolio_holdings WHERE user_id=%s", (user_id,))
    if not holdings:
        return {
            "holdings": [],
            "total_value_inr": 0,
            "total_cost_inr": 0,
            "pnl_inr": 0,
            "pnl_pct": 0,
            "fx_usdinr": None,
            "priced": 0,
            "unpriced": [],
            "cached": False,
            "timestamp": _now_iso(),
        }

    loop = asyncio.get_event_loop()
    yf_symbols = sorted({_holding_yf_symbol(h) for h in holdings})
    needs_fx = any(_holding_needs_fx(h) for h in holdings)
    fetch_list = yf_symbols + (["USDINR=X"] if needs_fx else [])
    prices = await loop.run_in_executor(None, _fetch_last_prices_sync, fetch_list)

    fx = prices.get("USDINR=X")
    if needs_fx and not fx:
        # Without FX, pricing crypto in INR would be silently wrong — keep the
        # response honest by leaving those holdings unpriced instead.
        log.warning("portfolio_value: USDINR rate unavailable; USD holdings unpriced")

    rows, unpriced = [], []
    total_value = total_cost = 0.0
    for h in holdings:
        qty = float(h["quantity"])
        cost = qty * float(h["avg_price"])  # cost basis stored in INR
        yf_sym = _holding_yf_symbol(h)
        last = prices.get(yf_sym)
        is_usd = _holding_needs_fx(h)
        rate = fx if is_usd else 1.0
        if last is not None and rate:
            value = qty * last * rate
            total_value += value
            total_cost += cost
            rows.append(
                {
                    "symbol": h["symbol"],
                    "name": h["name"],
                    "asset_type": h["asset_type"],
                    "quantity": qty,
                    "avg_price": float(h["avg_price"]),
                    "last_price_inr": round(last * rate, 2),
                    "value_inr": round(value, 2),
                    "cost_inr": round(cost, 2),
                    "pnl_pct": round((value - cost) / cost * 100, 2) if cost else 0,
                }
            )
        else:
            unpriced.append(h["symbol"])

    for r in rows:
        r["weight_pct"] = round(r["value_inr"] / total_value * 100, 2) if total_value else 0

    result = {
        "holdings": rows,
        "total_value_inr": round(total_value, 2),
        "total_cost_inr": round(total_cost, 2),
        "pnl_inr": round(total_value - total_cost, 2),
        "pnl_pct": round((total_value - total_cost) / total_cost * 100, 2) if total_cost else 0,
        "fx_usdinr": round(fx, 4) if fx else None,
        "priced": len(rows),
        "unpriced": unpriced,
        "timestamp": _now_iso(),
    }
    cache.set(cache_key, result, ttl=60)
    return {**result, "cached": False}


# ── Candles Endpoint (yfinance) ───────────────────────────────────────────────

# Maps frontend asset key → yfinance ticker symbol
_CANDLE_SYMBOL_MAP: dict[str, str] = {
    "btc": "BTC-USD",
    "eth": "ETH-USD",
    "spy": "SPY",
}

# Maps frontend timeframe → (period, interval) for yfinance
_TF_PARAMS: dict[str, tuple[str, str]] = {
    "1D": ("1d", "5m"),
    "7D": ("7d", "1h"),
    "1M": ("1mo", "1d"),
    "1Y": ("1y", "1wk"),
}

# Candle cache TTLs (seconds)
_CANDLE_TTL: dict[str, int] = {
    "1D": 60,
    "7D": 300,
    "1M": 600,
    "1Y": 3600,
}


def _fetch_candles_sync(yf_symbol: str, period: str, interval: str) -> list[dict]:
    """Blocking yfinance download — must be called via run_in_executor."""
    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        return []
    candles = []
    for ts, row in df.iterrows():
        candles.append(
            {
                "t": int(ts.timestamp() * 1000),
                "o": float(f"{float(row['Open']):.6g}"),
                "h": float(f"{float(row['High']):.6g}"),
                "l": float(f"{float(row['Low']):.6g}"),
                "c": float(f"{float(row['Close']):.6g}"),
                "v": int(row.get("Volume", 0) or 0),
            }
        )
    return candles


@router.get("/market/candles/{asset}")
async def market_candles(
    asset: str,
    tf: str = Query("1D", pattern="^(1D|7D|1M|1Y)$"),
):
    """
    Returns OHLCV candles for a given asset and timeframe.

    asset: btc | eth | spy
    tf:    1D | 7D | 1M | 1Y
    """
    asset_lower = asset.lower()
    yf_symbol = _CANDLE_SYMBOL_MAP.get(asset_lower)
    if yf_symbol is None:
        raise HTTPException(404, f"Unknown asset '{asset}'. Supported: {list(_CANDLE_SYMBOL_MAP)}")

    cache_key = f"candles:{asset_lower}:{tf}"
    cached = cache.get(cache_key)
    if cached is not None:
        return {
            "symbol": asset_lower,
            "tf": tf,
            "candles": cached,
            "cached": True,
            "timestamp": _now_iso(),
        }

    period, interval = _TF_PARAMS[tf]
    try:
        loop = asyncio.get_event_loop()
        candles = await loop.run_in_executor(None, _fetch_candles_sync, yf_symbol, period, interval)
    except Exception as e:
        log.error("yfinance %s %s/%s failed: %s", yf_symbol, period, interval, e)
        raise HTTPException(502, "Price history unavailable") from e

    if not candles:
        raise HTTPException(404, f"No data returned for {asset_lower} / {tf}")

    ttl = _CANDLE_TTL.get(tf, 300)
    cache.set(cache_key, candles, ttl=ttl)
    log.info("yfinance %s %s/%s: %d candles", yf_symbol, period, interval, len(candles))
    return {
        "symbol": asset_lower,
        "tf": tf,
        "candles": candles,
        "cached": False,
        "timestamp": _now_iso(),
    }


# ── Indicators Endpoint ───────────────────────────────────────────────────────


def _normalize_to_yf(symbol: str) -> str:
    """Map any frontend ticker string to a yfinance symbol."""
    s = symbol.upper().strip()
    # Strip exchange prefixes
    for pfx in ("BINANCE:", "NASDAQ:", "NYSE:", "AMEX:"):
        if s.startswith(pfx):
            s = s[len(pfx) :]
            break
    # Crypto XYZUSDT / XYZUSD → XYZ-USD
    if s.endswith("USDT"):
        return s[:-4] + "-USD"
    if s.endswith("USD") and len(s) > 3 and not s[:-3].isdigit():
        return s[:-3] + "-USD"
    # Stock tickers pass through unchanged
    return s


def _compute_indicators_sync(yf_symbol: str) -> dict:
    """Blocking: fetch 60 days of daily data and compute Vol, RSI(14), MACD."""

    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(period="60d", interval="1d", auto_adjust=True)
    if df.empty or len(df) < 27:
        return {}

    closes = df["Close"].tolist()
    volumes = df["Volume"].tolist()

    # Volume: mean of last 14 days
    vol = int(sum(volumes[-14:]) / 14) if volumes else 0

    # RSI(14) — Wilder smoothing
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains[:14]) / 14
    avg_loss = sum(losses[:14]) / 14
    for i in range(14, len(gains)):
        avg_gain = (avg_gain * 13 + gains[i]) / 14
        avg_loss = (avg_loss * 13 + losses[i]) / 14
    rsi = 100.0 if avg_loss == 0 else round(100 - 100 / (1 + avg_gain / avg_loss), 2)

    # MACD(12, 26, 9)
    def ema(prices: list, period: int) -> list:
        k = 2 / (period + 1)
        result = [prices[0]]
        for p in prices[1:]:
            result.append(p * k + result[-1] * (1 - k))
        return result

    ema12 = ema(closes, 12)
    ema26 = ema(closes, 26)
    macd_line = [ema12[i] - ema26[i] for i in range(len(closes))]
    signal = ema(macd_line, 9)
    macd_val = round(macd_line[-1], 4)
    signal_val = round(signal[-1], 4)

    return {
        "volume": vol,
        "rsi": rsi,
        "macd": macd_val,
        "macd_signal": signal_val,
        "macd_hist": round(macd_val - signal_val, 4),
    }


@router.get("/market/indicators/{symbol}")
async def market_indicators(symbol: str):
    """
    Returns Vol (14d avg), RSI(14), and MACD(12,26,9) for a given symbol.
    symbol: any frontend ticker — e.g. BINANCE:BTCUSDT, NVDA, AAPL, BTC-USD
    """
    yf_symbol = _normalize_to_yf(symbol)
    cache_key = f"indicators:{yf_symbol}"
    cached = cache.get(cache_key)
    if cached is not None:
        return {"symbol": yf_symbol, "indicators": cached, "cached": True, "timestamp": _now_iso()}

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _compute_indicators_sync, yf_symbol)
    except Exception as e:
        log.error("indicators %s failed: %s", yf_symbol, e)
        raise HTTPException(502, "Indicators unavailable") from e

    if not result:
        raise HTTPException(404, f"Insufficient data for {yf_symbol}")

    cache.set(cache_key, result, ttl=300)
    log.info(
        "indicators %s: RSI=%.1f MACD=%.4f", yf_symbol, result.get("rsi", 0), result.get("macd", 0)
    )
    return {"symbol": yf_symbol, "indicators": result, "cached": False, "timestamp": _now_iso()}


# ── News Fetcher ──────────────────────────────────────────────────────────────

# Finance/crypto domains — keeps general headlines on-topic
_FINANCE_DOMAINS = (
    "coindesk.com,cointelegraph.com,reuters.com,bloomberg.com,"
    "cnbc.com,marketwatch.com,wsj.com,ft.com,investing.com,"
    "cryptonews.com,decrypt.co,theblock.co,financialtimes.com"
)

# General market query used for the scrolling headline ticker
_GENERAL_QUERY = (
    'bitcoin OR ethereum OR cryptocurrency OR "stock market" OR '
    '"Federal Reserve" OR "interest rate" OR "S&P 500" OR NASDAQ'
)

# Maps common asset names/symbols to tighter search terms
_ASSET_QUERY_MAP: dict[str, str] = {
    "bitcoin": "Bitcoin BTC",
    "ethereum": "Ethereum ETH",
    "solana": "Solana SOL",
    "bnb": "Binance BNB",
    "ripple": "Ripple XRP",
    "dogecoin": "Dogecoin DOGE",
    "cardano": "Cardano ADA",
    "nvidia": "NVIDIA NVDA",
    "apple": "Apple AAPL",
    "microsoft": "Microsoft MSFT",
    "tesla": "Tesla TSLA",
    "amazon": "Amazon AMZN",
    "meta": "Meta Platforms META",
    "alphabet": "Alphabet Google GOOGL",
}


async def _fetch_news_live(query: str, page_size: int = 15) -> list[dict]:
    """Fetch articles from NewsAPI /v2/everything."""
    params: dict[str, Any] = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": settings.NEWSAPI_KEY,
        "domains": _FINANCE_DOMAINS,
    }
    r = await get_client().get(
        "https://newsapi.org/v2/everything",
        params=params,
    )
    r.raise_for_status()
    data = r.json()

    articles = []
    for item in data.get("articles", []):
        title = (item.get("title") or "").strip()
        if not title or title == "[Removed]":
            continue
        articles.append(
            {
                "title": title,
                "source": item.get("source", {}).get("name", ""),
                "url": item.get("url", ""),
                "published_at": item.get("publishedAt", ""),
                "summary": (item.get("description") or "")[:200].strip(),
            }
        )
    log.info("NewsAPI '%s': fetched %d articles", query[:40], len(articles))
    return articles


@router.get("/market/news")
async def market_news(q: str = ""):
    """
    Returns finance headlines from NewsAPI.

    - No  ?q  →  general market headlines (scrolling ticker)
    - ?q=bitcoin  →  asset-specific news for the intel drawer

    Cached per query string for NEWS_TTL seconds (default 300s).
    """
    if not settings.NEWSAPI_KEY:
        raise HTTPException(503, "NEWSAPI_KEY not configured")

    # Normalise: strip BINANCE: prefix, lower-case
    term = q.strip().lower().removeprefix("binance:").removesuffix("usdt")
    cache_key = f"news:{term or '__general__'}"

    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return {"articles": cached_data, "query": term, "cached": True, "timestamp": _now_iso()}

    # Resolve the best search query for this term
    if not term:
        search_query = _GENERAL_QUERY
        page_size = 20
    else:
        search_query = _ASSET_QUERY_MAP.get(term, q.strip())
        page_size = 8

    try:
        articles = await _fetch_news_live(search_query, page_size)
    except httpx.HTTPStatusError as e:
        log.error("NewsAPI HTTP error: %s", e)
        raise HTTPException(502, f"NewsAPI error: {e.response.status_code}") from e
    except Exception as e:
        log.error("NewsAPI fetch failed: %s", e)
        raise HTTPException(502, "NewsAPI unreachable") from e

    cache.set(cache_key, articles, ttl=settings.NEWS_TTL)
    return {"articles": articles, "query": term, "cached": False, "timestamp": _now_iso()}
