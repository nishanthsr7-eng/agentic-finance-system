"""SMA-crossover backtest over yfinance history."""

import asyncio
import functools
from datetime import date

import yfinance as yf
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..auth import require_user
from .common import log

router = APIRouter()


class BacktestDataError(ValueError):
    """Not enough price data for the request. The message is ours and safe to show."""


# ── Backtesting (yfinance SMA crossover) ─────────────────────────────────────


class BacktestRequest(BaseModel):
    symbol: str = Field(..., pattern=r"^[A-Za-z0-9^.\-=]{1,15}$")  # e.g. "BTC-USD", "AAPL", "^GSPC"
    start: date  # "YYYY-MM-DD"
    end: date  # "YYYY-MM-DD"
    short_sma: int = Field(20, ge=2, le=400)
    long_sma: int = Field(50, ge=2, le=400)
    initial_capital: float = Field(100000.0, gt=0, le=1e9)


# The demo account auto-runs the same default backtest on every Analysis
# visit; cache by arguments (end date = today, so entries roll daily). Errors
# raise and are not cached. Results are small (≤20 trades + the curve).
@functools.lru_cache(maxsize=32)
def _run_backtest_sync(
    symbol: str, start: str, end: str, short_sma: int, long_sma: int, initial_capital: float
) -> dict:
    import pandas as pd

    # Map friendly tickers
    _map = {"BTC": "BTC-USD", "ETH": "ETH-USD"}
    yf_sym = _map.get(symbol.upper(), symbol)

    # yfinance shares state across threads; a download that overlaps another
    # one (the Analysis page starts several) can come back empty or cut short.
    # Retry once before giving up.
    for _attempt in range(2):
        df = yf.download(
            yf_sym, start=start, end=end, auto_adjust=True, progress=False, threads=False
        )
        if len(df) > long_sma:
            break
    if df.empty:
        raise BacktestDataError(f"No data for {yf_sym} in range {start}–{end}")
    if len(df) <= long_sma:
        raise BacktestDataError(f"Only {len(df)} days of {yf_sym} data; SMA({long_sma}) needs more")

    # yfinance returns MultiIndex columns (field, ticker) even for a single
    # symbol — flatten so df["Close"]/row["Close"] etc. are plain scalars.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].squeeze()
    df["sma_s"] = close.rolling(short_sma).mean()
    df["sma_l"] = close.rolling(long_sma).mean()
    df = df.dropna()

    # Signal: 1 = long, 0 = out
    df["signal"] = (df["sma_s"] > df["sma_l"]).astype(int)
    df["pos_chg"] = df["signal"].diff()

    capital = initial_capital
    shares = 0.0
    trades = []
    equity_curve = []

    for ts, row in df.iterrows():
        price = float(row["Close"])
        if row["pos_chg"] == 1 and capital > 0:  # buy
            shares = capital / price
            capital = 0.0
            trades.append(
                {
                    "date": str(ts.date()),
                    "action": "BUY",
                    "price": round(price, 4),
                    "shares": round(shares, 6),
                }
            )
        elif row["pos_chg"] == -1 and shares > 0:  # sell
            capital = shares * price
            pnl = (
                capital - initial_capital
                if not trades
                else capital - (trades[-1]["price"] * shares)
            )
            trades.append(
                {
                    "date": str(ts.date()),
                    "action": "SELL",
                    "price": round(price, 4),
                    "shares": round(shares, 6),
                    "pnl": round(pnl, 2),
                }
            )
            shares = 0.0

        total_value = capital + shares * price
        equity_curve.append({"t": int(ts.timestamp() * 1000), "v": round(total_value, 2)})

    # Final portfolio value
    last_price = float(df["Close"].iloc[-1])
    final_value = capital + shares * last_price
    total_return = (final_value - initial_capital) / initial_capital * 100

    # Daily returns for Sharpe
    eq_vals = [p["v"] for p in equity_curve]
    if len(eq_vals) > 1:
        daily_rets = [
            (eq_vals[i] - eq_vals[i - 1]) / eq_vals[i - 1] for i in range(1, len(eq_vals))
        ]
        avg_r = sum(daily_rets) / len(daily_rets)
        std_r = (sum((r - avg_r) ** 2 for r in daily_rets) / len(daily_rets)) ** 0.5
        sharpe = round((avg_r / std_r * (252**0.5)) if std_r > 0 else 0, 3)
        # Max drawdown
        peak = eq_vals[0]
        max_dd = 0.0
        for v in eq_vals:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100
            if dd > max_dd:
                max_dd = dd
    else:
        sharpe = 0
        max_dd = 0.0

    sell_trades = [t for t in trades if t["action"] == "SELL"]
    win_rate = round(
        sum(1 for t in sell_trades if t.get("pnl", 0) > 0) / max(len(sell_trades), 1) * 100, 1
    )

    return {
        "symbol": yf_sym,
        "start": start,
        "end": end,
        "short_sma": short_sma,
        "long_sma": long_sma,
        "initial_capital": initial_capital,
        "final_value": round(final_value, 2),
        "metrics": {
            "total_return": round(total_return, 2),
            "sharpe": sharpe,
            "max_drawdown": round(max_dd, 2),
            "win_rate": win_rate,
            "total_trades": len(trades),
        },
        "trades": trades[-20:],  # last 20 to keep payload small
        "equity_curve": equity_curve,
    }


@router.post("/backtest", dependencies=[Depends(require_user)])
async def run_backtest(body: BacktestRequest):
    """
    Run an SMA crossover backtest via yfinance.

    Body: { symbol, start, end, short_sma, long_sma, initial_capital }
    Returns: { metrics, equity_curve, trades }
    """
    if body.short_sma >= body.long_sma:
        raise HTTPException(400, "short_sma must be less than long_sma")
    if body.start >= body.end:
        raise HTTPException(400, "start must be before end")
    if (body.end - body.start).days > 3653:
        raise HTTPException(400, "Date range is limited to 10 years")
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            _run_backtest_sync,
            body.symbol,
            body.start.isoformat(),
            body.end.isoformat(),
            body.short_sma,
            body.long_sma,
            body.initial_capital,
        )
        return result
    except BacktestDataError as e:
        raise HTTPException(404, str(e)) from e
    except Exception as e:
        log.error("Backtest failed: %s", e)
        raise HTTPException(502, "Backtest failed. Try another symbol or date range.") from e
