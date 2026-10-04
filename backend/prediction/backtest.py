"""
FLUX Prediction — Cost-Aware Walk-Forward Backtest
==================================================
Turns the leak-free OOF signals into a tradeable strategy and measures whether it actually
beats buy-and-hold on a RISK-ADJUSTED basis (the metric that matters — raw accuracy vs the
market's up-drift is the wrong yardstick).

Strategy rules:
  • Use only OUT-OF-FOLD predictions (purged walk-forward) → no look-ahead.
  • Act only when the meta-model says so (meta_prob >= ACT_THRESHOLD).
  • Direction = primary call (long on UP, short on DOWN).
  • Size = fractional Kelly from the calibrated meta-probability.
  • Non-overlapping trades per symbol (enter when flat, hold to the barrier/timeout).
  • Subtract realistic round-trip cost on every trade.
Benchmark: passive buy-and-hold over the SAME events (always long, no trading cost) — a
deliberately hard, honest bar.

Run:  python -m backend.prediction.backtest
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.prediction.predict import ACT_THRESHOLD  # noqa: E402
from backend.prediction.regime import REGIME_SCALE, decode_regimes  # noqa: E402
from backend.prediction.train import generate_oof_signals, load_dataset  # noqa: E402

COST = 0.0005  # 5 bps round-trip cost + slippage, charged per active trade
TRADING_DAYS = 252


# ── Metrics ────────────────────────────────────────────────────────────────────
def _equity(returns: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 + returns)


def _max_drawdown(eq: np.ndarray) -> float:
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min()) if len(eq) else 0.0


def _sharpe(returns: np.ndarray, trades_per_year: float) -> float:
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return 0.0
    return float(returns.mean() / returns.std(ddof=1) * np.sqrt(trades_per_year))


def _profit_factor(returns: np.ndarray) -> float:
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    return float(gains / losses) if losses > 0 else float("inf")


def _summarize_symbol(returns: np.ndarray, exits: np.ndarray) -> dict | None:
    """Per-symbol metrics from that symbol's own non-overlapping trade stream."""
    if len(returns) < 5:
        return None
    eq = _equity(returns)
    span_years = max(
        (pd.to_datetime(exits.max()) - pd.to_datetime(exits.min())).days / 365.25, 0.25
    )
    tpy = len(returns) / span_years
    return {
        "n": len(returns),
        "total_return": float(eq[-1] - 1.0),
        "sharpe": _sharpe(returns, tpy),
        "max_dd": _max_drawdown(eq),
        "win_rate": float((returns > 0).mean()),
        "profit_factor": _profit_factor(returns),
    }


# ── Per-symbol simulation ───────────────────────────────────────────────────────
def simulate_symbol(g: pd.DataFrame, thr: float = ACT_THRESHOLD, use_regime: bool = False):
    """Non-overlapping trades for one symbol. Full notional so it's comparable to buy-hold.
    With use_regime, position size is scaled by the market regime (0 in risk_off → stand aside)."""
    strat, hold, acted = [], [], []
    next_s = next_h = None
    for row in g.sort_values("date").itertuples():
        if next_h is None or row.date >= next_h:  # buy & hold: always long, no cost
            hold.append((row.t1, row.ret))
            next_h = row.t1
        if (next_s is None or row.date >= next_s) and row.meta >= thr:
            scale = REGIME_SCALE.get(row.regime, 1.0) if use_regime else 1.0
            if scale <= 0:
                continue  # stand aside in risk_off
            side = 1 if row.prim >= 0.5 else -1  # long UP / short DOWN
            pnl = scale * (side * row.ret) - scale * COST
            strat.append((row.t1, pnl))
            acted.append((row.meta, side * row.ret > 0))
            next_s = row.t1
    return strat, hold, acted


def _run_threshold(ev: pd.DataFrame, thr: float, use_regime: bool = False):
    """Aggregate strategy metrics across symbols for a given act threshold."""
    strat_m, all_acted, n_trades = [], [], 0
    for _sym, g in ev.groupby("sym"):
        strat, _hold, acted = simulate_symbol(g, thr, use_regime)
        all_acted += acted
        n_trades += len(strat)
        s = (
            _summarize_symbol(np.array([t[1] for t in strat]), np.array([t[0] for t in strat]))
            if strat
            else None
        )
        if s:
            strat_m.append(s)
    return _aggregate(strat_m), n_trades, all_acted


def _aggregate(per_symbol: list[dict]) -> dict:
    """Equal-weight aggregate across symbols (median return = robust to crypto outliers)."""
    if not per_symbol:
        return {"symbols": 0}
    arr = lambda k: np.array([m[k] for m in per_symbol])
    return {
        "symbols": len(per_symbol),
        "trades": int(arr("n").sum()),
        "sharpe": float(np.mean(arr("sharpe"))),
        "total_return_med": float(np.median(arr("total_return"))),
        "max_dd": float(np.mean(arr("max_dd"))),
        "win_rate": float(np.mean(arr("win_rate"))),
        "profit_factor": float(np.median(arr("profit_factor"))),
    }


async def run():
    t0 = time.time()
    print("Loading dataset + generating leak-free OOF signals...")
    X, y, w, t1, feat_cols, fd_orders, data = await load_dataset()
    prim, meta, iso, _mrep = generate_oof_signals(X, y, w, t1, feat_cols)

    ev = pd.DataFrame(
        {
            "sym": data["_sym"].values,
            "date": pd.to_datetime(data["_date"].values),
            "t1": pd.to_datetime(data["_t1"].values),
            "ret": data["_ret"].values,
            "prim": prim,
            "meta": meta,
        }
    ).dropna(subset=["meta"])

    # Attach causal market regime (ffill covers crypto weekends).
    reg = await decode_regimes()
    reg_by_date = reg["regime"].reindex(pd.date_range(reg.index.min(), ev["date"].max())).ffill()
    ev["regime"] = ev["date"].map(reg_by_date).fillna("trend")
    print(
        f"OOF events available for backtest: {len(ev):,} "
        f"({ev['date'].min().date()} -> {ev['date'].max().date()})"
    )
    print(f"event regime mix: {ev['regime'].value_counts().to_dict()}\n")

    # Buy & hold benchmark (once)
    hold_m, n_hold_trades = [], 0
    for _sym, g in ev.groupby("sym"):
        _s, hold, _a = simulate_symbol(g, thr=2.0)  # thr=2 → never trades strat leg
        n_hold_trades += len(hold)
        h = (
            _summarize_symbol(np.array([t[1] for t in hold]), np.array([t[0] for t in hold]))
            if hold
            else None
        )
        if h:
            hold_m.append(h)
    H = _aggregate(hold_m)

    # Strategy across a sweep of selectivity thresholds — UNGATED vs REGIME-GATED
    def sweep_table(use_regime):
        print(
            f"  {'act_thr':>8}{'exposure':>10}{'avgSharpe':>11}{'medTotal':>11}{'avgMaxDD':>10}{'win':>6}{'trades':>8}"
        )
        print("-" * 84)
        res = {}
        for thr in sorted({0.50, 0.55, 0.60, 0.65, 0.70, ACT_THRESHOLD}):
            S, n_trades, acted = _run_threshold(ev, thr, use_regime)
            res[thr] = (S, n_trades, acted)
            if S["symbols"]:
                exp = n_trades / max(n_hold_trades, 1)
                print(
                    f"  {thr:>8.2f}{exp:>10.1%}{S['sharpe']:>11.2f}{S['total_return_med']:>+11.1%}"
                    f"{S['max_dd']:>10.1%}{S['win_rate']:>6.0%}{n_trades:>8}"
                )
        return res

    print("=" * 84)
    print("  STRATEGY — UNGATED")
    print("-" * 84)
    ungated = sweep_table(use_regime=False)
    print("\n  STRATEGY — REGIME-GATED (size x regime: trend 1.0 / chop 0.6 / risk_off 0.25)")
    print("-" * 84)
    gated = sweep_table(use_regime=True)
    print("-" * 84)
    print(
        f"  {'BUY&HOLD':>8}{'100.0%':>10}{H['sharpe']:>11.2f}{H['total_return_med']:>+11.1%}"
        f"{H['max_dd']:>10.1%}{H['win_rate']:>6.0%}{n_hold_trades:>8}"
    )
    print("=" * 84)

    # Headline: compare ungated vs gated vs buy-hold at the live act threshold (ACT_THRESHOLD)
    U, G = ungated[ACT_THRESHOLD][0], gated[ACT_THRESHOLD][0]
    print(f"  At act_thr={ACT_THRESHOLD}:")
    print(
        f"    ungated    Sharpe {U['sharpe']:.2f} | maxDD {U['max_dd']:.1%} | exposure {ungated[ACT_THRESHOLD][1] / max(n_hold_trades, 1):.0%}"
    )
    print(
        f"    regime-gated Sharpe {G['sharpe']:.2f} | maxDD {G['max_dd']:.1%} | exposure {gated[ACT_THRESHOLD][1] / max(n_hold_trades, 1):.0%}"
    )
    print(f"    buy & hold Sharpe {H['sharpe']:.2f} | maxDD {H['max_dd']:.1%}")
    sw = G["sharpe"] > H["sharpe"]
    dw = abs(G["max_dd"]) < abs(H["max_dd"])
    print(
        f"    VERDICT (gated vs buy-hold): Sharpe {'WIN' if sw else 'loss'} | drawdown {'WIN' if dw else 'loss'}"
    )
    all_acted = ungated[ACT_THRESHOLD][2]  # buckets from the broad set

    # ── Per-confidence-bucket calibration (realized hit-rate by meta decile) ───
    print("\n  Calibration buckets (meta_prob decile -> realized win-rate):")
    acted = pd.DataFrame(all_acted, columns=["meta", "win"])
    rows = []
    if len(acted):
        acted["bucket"] = (acted["meta"] * 10).clip(0, 9).astype(int)
        ts = int(time.time() * 1000)
        for b, g in acted.groupby("bucket"):
            stated = (b + 0.5) / 10
            hit = float(g["win"].mean())
            print(f"     bucket {b} (~{stated:.0%}): realized {hit:.0%}  (n={len(g)})")
            rows.append(
                {
                    "model": "ensemble",
                    "bucket": int(b),
                    "stated_conf": stated,
                    "realized_hit": hit,
                    "n": int(len(g)),
                    "updated_at": ts,
                }
            )
        try:
            from backend.db import init_db, upsert_calibration

            await init_db()
            await upsert_calibration(rows)
            print("   -> saved to calibration_buckets")
        except Exception as exc:
            print(f"   (calibration save skipped: {exc})")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
