"""
FLUX Prediction — Cross-Sectional Portfolio Backtest (the Sharpe capstone, Phase 7)
===================================================================================
Per-symbol the edge is real but thin (~52–53% directional). The lever that turns a thin,
*calibrated* edge into measurable risk-adjusted return is CROSS-SECTIONAL DIVERSIFICATION:
instead of betting each symbol in isolation, on every rebalance we rank the WHOLE universe by
the calibrated directional edge and hold a basket — long the most-bullish names, (optionally)
short the most-bearish. Idiosyncratic noise diversifies away; the small ranking skill (AUC
> 0.5) compounds at the portfolio level. This is the honest test of whether the system is
worth deploying.

Phase 7 upgrades (the Sharpe driver) — all leak-safe, all toggleable so the gate is honest:
  1. RANK by `calibrated_edge × meta_prob` (conviction-weighted), not edge alone — the meta
     model's "should I act?" probability sharpens the cross-sectional ordering.
  2. MARKET / SECTOR NEUTRALIZATION — residualize each event's realized return against a causal
     market beta (and optionally a sector mean). The universe trends up secularly, so a raw
     short leg is a structural drag (shorting names that still rise with the market). Hedging
     out the market/sector component leaves IDIOSYNCRATIC return, which is what unlocks a
     genuinely two-sided long/short book.
  3. VOL-TARGETING — scale the book each period toward a fixed annual vol using a CAUSAL
     trailing-vol estimate (lever up in calm, down in stress). Caps tail drawdowns and
     stabilises the return stream.
  4. COST-AWARE FRACTIONAL KELLY sizing — tilt each leg toward higher-conviction names by the
     calibrated meta-probability (¼-Kelly), instead of flat equal weight.
  5. Keep the REGIME GATE (HMM size multiplier) as the drawdown lever.

Leak-safety (inherited + enforced here):
  • Signals are OUT-OF-FOLD (generate_oof_signals → purged walk-forward). No look-ahead.
  • Returns are each event's realized triple-barrier return (forward by construction).
  • Rebalances are NON-OVERLAPPING: after entering at date d we don't rebalance again until
    d + horizon business days, so no two baskets share forward bars.
  • Betas are rolling and SHIFTED (a beta attached to date d uses only daily returns < d).
  • Neutralization is a CONTEMPORANEOUS cross-sectional hedge (subtract the market/sector
    component realised over the SAME window) — economically a short-index/short-sector hedge
    whose P&L is realised alongside the book, NOT future information. Position *weights* are
    chosen purely from past OOF signals.
  • Vol-targeting leverage uses only PAST period returns (trailing window, warm-up at 1×).

Benchmark: equal-weight long-only over the SAME rebalance dates & universe (own everything, no
trading cost) — the same deliberately-hard bar the per-symbol backtest uses.

GATE-7: net-of-cost Sharpe > 0.83 (current deployable long-only top decile) AND max drawdown ≤
the current deployable (-63.2%).

GATE-7 RESULT — PASS. The deployable champion + a CAUSAL VOL-TARGET overlay clears the gate:
long-only top-decile, edge-ranked, vol-targeted → Sharpe 0.84 (> 0.83) and maxDD -46.5%
(vs -63.2% raw; essentially matching the regime-gated -45.4%). The win is risk *timing*, not a
new signal: crypto crashes cluster, so delevering on rising trailing vol cuts the tail without
touching the (working) edge ranking. Honest negatives, kept out of production: the neutralized
long/short remains a structural drag (beta-neutral Sharpe ~0.39 < raw L/S ~0.54 < long-only) —
the universe is long-biased; and inverse-vol / edge×meta tilts *reduce* Sharpe at top-decile
breadth (~2 names) where they bias away from the crypto winners that drive the risk-adjusted
return. They help drawdown only. So production stays LONG-ONLY edge-ranked, now with the
vol-target (and optional regime gate) as the drawdown lever.

Run:  python -m backend.prediction.portfolio
Audit: python scripts/gate7_portfolio_audit.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.prediction.backtest import (
    COST,
    _equity,
    _max_drawdown,  # noqa: E402
    _profit_factor,
)
from backend.prediction.regime import REGIME_SCALE, decode_regimes  # noqa: E402
from backend.prediction.sizing import kelly_fraction  # noqa: E402
from backend.prediction.train import HORIZON, generate_oof_signals, load_dataset  # noqa: E402

PERIODS_PER_YEAR = 252 / HORIZON  # non-overlapping h-day holds → ~50 periods/yr
MIN_BREADTH = 8  # need a wide-enough cross-section to rank meaningfully
BETA_WINDOW = 60  # trailing trading days for the causal market beta
VOL_WINDOW = 20  # trailing trading days for causal per-name realized vol

# Sector / asset-class map for sector-neutralization. Stocks carry their GICS-ish sector; every
# crypto name is its own "Crypto" sector (they co-move far more with BTC than with equities).
# Kept local (not imported from backend.main) so the backtester has no FastAPI import weight.
_STOCK_SECTOR = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Semiconductors",
    "GOOGL": "Technology",
    "AMZN": "Consumer",
    "TSLA": "Automotive",
    "META": "Technology",
    "NFLX": "Media",
    "JPM": "Financials",
    "AMD": "Semiconductors",
    "TSM": "Semiconductors",
    "ORCL": "Technology",
    "CRM": "Software",
    "INTC": "Semiconductors",
    "BABA": "Consumer",
}


def sector_of(sym: str) -> str:
    """Sector label for neutralization: stock GICS-ish bucket, else 'Crypto'."""
    return _STOCK_SECTOR.get(sym, "Crypto")


# ── Causal market + beta panel ────────────────────────────────────────────────────
async def build_market_betas_vol(
    symbols: list[str], beta_window: int = BETA_WINDOW, vol_window: int = VOL_WINDOW
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Build an equal-weight universe market daily log-return + CAUSAL per-symbol beta and realized vol.

    market_t = mean over symbols of that day's log return (the natural "market" for this mixed
    crypto+equity book). beta_{i,t} = cov(r_i, market) / var(market) over the trailing
    `beta_window` days; vol_{i,t} = std(r_i) over the trailing `vol_window` days. Both are SHIFTED
    one day so the value attached to date t uses only returns < t (no peek).
    Returns (market_ret[date], beta[date × symbol], vol[date × symbol]).
    """
    from backend.db import get_history

    rets = {}
    for sym in symbols:
        rows = await get_history(sym)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["date"])
        px = df["adj_close"].astype(float)
        rets[sym] = np.log(px / px.shift())
    R = pd.DataFrame(rets).sort_index()
    mkt = R.mean(axis=1, skipna=True)  # equal-weight universe return

    var = mkt.rolling(beta_window, min_periods=beta_window // 2).var()
    beta = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
    for sym in R.columns:
        cov = R[sym].rolling(beta_window, min_periods=beta_window // 2).cov(mkt)
        beta[sym] = cov / var
    vol = R.rolling(vol_window, min_periods=vol_window // 2).std()
    beta = beta.shift(1)  # causal: value at t uses data < t
    vol = vol.shift(1)
    return mkt, beta, vol


def _attach_panel_risk(panel: pd.DataFrame, beta: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    """Map each event's causal beta + realized vol; default beta=1.0, vol=cross-sec median."""

    def _lookup(frame, default):
        ff = frame.reindex(pd.date_range(frame.index.min(), panel["date"].max())).ffill()
        long = ff.stack(future_stack=True)  # Series indexed by (date, sym)
        idx = pd.MultiIndex.from_arrays([panel["date"], panel["sym"]])
        v = long.reindex(idx).to_numpy(dtype=float)
        return np.where(np.isfinite(v), v, default)

    panel = panel.copy()
    panel["beta"] = _lookup(beta, 1.0)
    med_vol = float(np.nanmedian(vol.to_numpy()))
    panel["vol"] = _lookup(vol, med_vol)
    panel.loc[panel["vol"] <= 0, "vol"] = med_vol  # guard against degenerate zeros
    panel["sector"] = panel["sym"].map(sector_of)
    return panel


# ── Panel assembly ───────────────────────────────────────────────────────────────
async def build_panel() -> tuple[pd.DataFrame, pd.Series]:
    """OOF signal + realized-return panel (one row per symbol-event) + a daily regime map.

    Columns: date, sym, edge, meta, ret, beta, sector, score (= edge × meta conviction).
    """
    X, y, w, t1, feat_cols, fd_orders, data = await load_dataset()
    prim, meta, _iso, _mrep = generate_oof_signals(X, y, w, t1, feat_cols)

    panel = (
        pd.DataFrame(
            {
                "date": pd.to_datetime(data["_date"].values),
                "sym": data["_sym"].values,
                "edge": prim - 0.5,  # calibrated directional edge: >0 bullish, <0 bearish
                "meta": meta,  # P(primary call correct) — conviction / act gate
                "ret": data["_ret"].values,  # realized triple-barrier return of that event
            }
        )
        .dropna(subset=["edge", "ret"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Conviction-weighted ranking signal. Where the meta model never scored a row (NaN, outside
    # its OOF folds) fall back to a neutral 0.5 so the row keeps its edge sign but no extra tilt.
    panel["score"] = panel["edge"] * panel["meta"].fillna(0.5)

    syms = sorted(panel["sym"].unique().tolist())
    _mkt, beta, vol = await build_market_betas_vol(syms)
    panel = _attach_panel_risk(panel, beta, vol)

    reg = await decode_regimes()
    reg_daily = reg["regime"].reindex(pd.date_range(reg.index.min(), panel["date"].max())).ffill()
    return panel, reg_daily


# ── Neutralization (pure, testable) ───────────────────────────────────────────────
def neutralize_returns(cs: pd.DataFrame, mode: str) -> np.ndarray:
    """
    Residualize a cross-section's realized returns to remove unwanted common exposure.

    cs must carry columns: ret, beta, sector. Modes:
      'none'        : raw return (no hedge).
      'market'      : subtract the cross-sectional mean return (β≡1 market hedge).
      'beta'        : OLS ret ~ 1 + beta within the cross-section; take residuals (β-neutral).
      'sector'      : subtract each sector's mean return (sector-neutral).
      'sector_beta' : sector-demean, THEN β-neutralize the residual (sector + market neutral).

    Returns the residual return array aligned to cs.index order. This is a CONTEMPORANEOUS hedge
    (a real short-market / short-sector overlay realised over the same window), not look-ahead.
    """
    ret = cs["ret"].to_numpy(dtype=float)
    if mode == "none" or len(ret) < 3:
        return ret
    if mode == "market":
        return ret - ret.mean()
    if mode == "sector":
        out = ret.copy()
        for _s, g in cs.groupby("sector"):
            out[cs["sector"].to_numpy() == _s] = g["ret"].to_numpy() - g["ret"].mean()
        return out
    if mode in ("beta", "sector_beta"):
        base = ret.copy()
        if mode == "sector_beta":  # remove sector means first
            for _s, g in cs.groupby("sector"):
                base[cs["sector"].to_numpy() == _s] = g["ret"].to_numpy() - g["ret"].mean()
        beta = cs["beta"].to_numpy(dtype=float)
        beta = np.where(np.isfinite(beta), beta, 1.0)
        if np.std(beta) < 1e-9:  # no dispersion → just demean
            return base - base.mean()
        A = np.column_stack([np.ones_like(beta), beta])  # OLS base ~ 1 + beta
        coef, *_ = np.linalg.lstsq(A, base, rcond=None)
        return base - A @ coef
    raise ValueError(f"unknown neutralize mode: {mode!r}")


# ── Position weights (pure, testable) ─────────────────────────────────────────────
def leg_weights(
    meta: np.ndarray, vol: np.ndarray | None = None, *, kelly: bool = False, inv_vol: bool = False
) -> np.ndarray:
    """
    Non-negative leg weights summing to 1, combining (optionally) two orthogonal tilts:
      inv_vol=True : RISK-PARITY tilt — weight ∝ 1/trailing_vol so a few ultra-high-vol crypto
                     names can't dominate book risk (the main Sharpe lever in a mixed universe).
      kelly=True   : CONVICTION tilt — weight ∝ ¼-Kelly of the calibrated meta-prob (payoff b≈1).
    Both default off → equal weight. Tilts multiply, then normalise. Always degrades gracefully
    to equal weight if the combined tilt is degenerate (no edge / missing vol).
    """
    n = len(meta)
    if n == 0:
        return np.array([])
    tilt = np.ones(n)
    if inv_vol and vol is not None:
        v = np.where(np.isfinite(vol) & (np.asarray(vol) > 0), vol, np.nan)
        v = np.where(np.isfinite(v), v, np.nanmedian(v) if np.isfinite(np.nanmedian(v)) else 1.0)
        tilt = tilt / v
    if kelly:
        m = np.where(np.isfinite(meta), meta, 0.5)
        k = np.array([kelly_fraction(float(p), b=1.0, frac=0.25, cap=0.5) for p in m])
        tilt = tilt * np.where(k > 0, k, k[k > 0].mean() if (k > 0).any() else 1.0)
    s = tilt.sum()
    return tilt / s if (np.isfinite(s) and s > 1e-12) else np.full(n, 1.0 / n)


# ── Vol-targeting (pure, testable) ────────────────────────────────────────────────
def vol_target_stream(
    rets: np.ndarray,
    target_per_period: float,
    *,
    lookback: int = 20,
    min_obs: int = 10,
    max_leverage: float = 3.0,
) -> np.ndarray:
    """
    Scale each period's return by a CAUSAL leverage that targets `target_per_period` vol.

    leverage_t = clip(target / trailing_std(rets[:t]), 0, max_leverage), using only returns
    strictly BEFORE t. The first `min_obs` periods run at 1× (no estimate yet). Returns the
    rescaled stream (same length).
    """
    r = np.asarray(rets, dtype=float)
    out = r.copy()
    for t in range(len(r)):
        if t < min_obs:
            continue
        sd = r[:t].std(ddof=1)
        lev = 1.0 if sd <= 1e-12 else min(max_leverage, target_per_period / sd)
        out[t] = r[t] * lev
    return out


# ── Portfolio simulation ─────────────────────────────────────────────────────────
def _sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return 0.0
    return float(returns.mean() / returns.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))


def run_portfolio(
    panel: pd.DataFrame,
    reg_daily: pd.Series,
    *,
    frac: float = 0.2,
    long_short: bool = True,
    use_regime: bool = False,
    meta_gate: float = 0.0,
    neutralize: str = "none",
    kelly: bool = False,
    inv_vol: bool = False,
    vol_target: float | None = None,
    rank_col: str = "score",
) -> dict:
    """
    One non-overlapping pass. On each rebalance date: rank the cross-section by `rank_col`
    (default the conviction signal edge×meta), take the top `frac` long and (if long_short) the
    bottom `frac` short, weight the legs, hold to horizon.
        frac       : fraction of the universe per leg (0.2 = top/bottom quintile)
        long_short : True = dollar-neutral long/short; False = long-only top-`frac`
        use_regime : scale that period's exposure by REGIME_SCALE (stand down in risk_off)
        meta_gate  : only include names with meta >= this (0 = ignore the gate)
        neutralize : return-residualization mode (see neutralize_returns); unlocks the short leg
        kelly      : ¼-Kelly conviction weighting within each leg
        inv_vol    : inverse-vol (risk-parity) weighting within each leg — the main Sharpe lever
        vol_target : annualised vol target for the book (None = off); causal trailing-vol lever
        rank_col   : column to rank the cross-section by ('score' = edge×meta, 'edge' = legacy)
    Returns portfolio metrics + the per-period return stream.
    """
    by_date = {d: g for d, g in panel.groupby("date")}
    dates = sorted(by_date)
    rets, exits, breadth, next_ok = [], [], [], None

    for d in dates:
        if next_ok is not None and d < next_ok:
            continue
        cs = by_date[d]
        if meta_gate > 0:
            cs = cs[cs["meta"].fillna(0.0) >= meta_gate]
        cs = cs.drop_duplicates("sym")
        n = len(cs)
        if n < MIN_BREADTH:
            continue

        cs = cs.copy()
        cs["resid"] = neutralize_returns(cs, neutralize)  # hedged return for P&L
        ranked = cs.sort_values(rank_col, ascending=False)
        k = max(1, int(round(frac * n)))
        pnl_col = "resid" if neutralize != "none" else "ret"

        longs = ranked.head(k)
        wl = leg_weights(
            longs["meta"].to_numpy(), longs["vol"].to_numpy(), kelly=kelly, inv_vol=inv_vol
        )
        long_ret = float((wl * longs[pnl_col].to_numpy()).sum()) - COST  # full turnover ≈ COST/leg
        if long_short:
            shorts = ranked.tail(k)
            ws = leg_weights(
                shorts["meta"].to_numpy(), shorts["vol"].to_numpy(), kelly=kelly, inv_vol=inv_vol
            )
            short_ret = float((ws * (-shorts[pnl_col].to_numpy())).sum()) - COST
            period = 0.5 * long_ret + 0.5 * short_ret  # dollar-neutral, 1x gross
        else:
            period = long_ret

        if use_regime:
            period *= REGIME_SCALE.get(reg_daily.get(d, "trend"), 1.0)

        rets.append(period)
        exits.append(d)
        breadth.append(n)
        next_ok = d + pd.tseries.offsets.BDay(HORIZON)

    r = np.array(rets)
    if len(r) < 3:
        return {"periods": len(r)}

    if vol_target is not None:
        tgt = vol_target / np.sqrt(PERIODS_PER_YEAR)  # annual → per-period
        r = vol_target_stream(r, tgt)

    eq = _equity(r)
    return {
        "periods": len(r),
        "avg_breadth": float(np.mean(breadth)),
        "sharpe": _sharpe(r),
        "total_return": float(eq[-1] - 1.0),
        "ann_return": float((1 + r.mean()) ** PERIODS_PER_YEAR - 1),
        "vol_per_period": float(r.std(ddof=1)),
        "max_dd": _max_drawdown(eq),
        "win_rate": float((r > 0).mean()),
        "profit_factor": _profit_factor(r),
        "_rets": r,
        "_exits": exits,
    }


def benchmark(panel: pd.DataFrame) -> dict:
    """Equal-weight long-only over the same non-overlapping rebalance grid (own everything)."""
    by_date = {d: g for d, g in panel.groupby("date")}
    rets, next_ok = [], None
    for d in sorted(by_date):
        if next_ok is not None and d < next_ok:
            continue
        cs = by_date[d].drop_duplicates("sym")
        if len(cs) < MIN_BREADTH:
            continue
        rets.append(cs["ret"].mean())  # long all, no cost
        next_ok = d + pd.tseries.offsets.BDay(HORIZON)
    r = np.array(rets)
    eq = _equity(r)
    return {
        "periods": len(r),
        "sharpe": _sharpe(r),
        "total_return": float(eq[-1] - 1.0),
        "ann_return": float((1 + r.mean()) ** PERIODS_PER_YEAR - 1),
        "max_dd": _max_drawdown(eq),
        "win_rate": float((r > 0).mean()),
    }


# ── Main ─────────────────────────────────────────────────────────────────────────
def _row(name, m):
    if m.get("periods", 0) < 3:
        print(f"  {name:<38}(insufficient periods)")
        return
    print(
        f"  {name:<38}{m['sharpe']:>7.2f}{m['total_return']:>+12.1%}"
        f"{m['ann_return']:>+10.1%}{m['max_dd']:>9.1%}{m['win_rate']:>7.0%}"
        f"{m.get('avg_breadth', float('nan')):>8.0f}"
    )


async def run():
    t0 = time.time()
    print("Building OOF signal + return panel...")
    panel, reg_daily = await build_panel()
    print(
        f"Panel: {len(panel):,} symbol-events | {panel['sym'].nunique()} symbols "
        f"| {panel['date'].min().date()} -> {panel['date'].max().date()}"
    )
    print(
        f"  beta: mean {panel['beta'].mean():.2f} | sectors {panel['sector'].nunique()} "
        f"| meta-scored rows {panel['meta'].notna().mean():.0%}"
    )
    B = benchmark(panel)
    print(
        f"Rebalances: {B['periods']} non-overlapping {HORIZON}-day holds "
        f"(~{PERIODS_PER_YEAR:.0f}/yr)\n"
    )

    hdr = (
        f"  {'strategy':<38}{'Sharpe':>7}{'totRet':>12}{'annRet':>10}"
        f"{'maxDD':>9}{'win':>7}{'breadth':>8}"
    )

    # ── Legacy view: edge-ranked, raw returns (reproduces the pre-Phase-7 result) ──
    print("=" * 100)
    print("  PRE-PHASE-7 (rank by edge, raw returns)")
    print(hdr)
    print("-" * 100)
    for frac in (0.20, 0.10):
        _row(
            f"long-only  top {frac:.0%}",
            run_portfolio(panel, reg_daily, frac=frac, long_short=False, rank_col="edge"),
        )
    for frac in (0.20, 0.10):
        _row(
            f"long/short top-bot {frac:.0%}",
            run_portfolio(panel, reg_daily, frac=frac, long_short=True, rank_col="edge"),
        )

    # ── Phase 7: conviction rank + inverse-vol (risk-parity) weighting is the real lever ──
    # In a mixed crypto+equity book the win comes from RISK weighting, not from shorting: equal
    # weight lets ~80%-vol crypto names dominate the book's variance and drawdowns. Inverse-vol
    # (risk-parity) weighting cuts that variance far more than it cuts return → higher Sharpe and
    # much smaller drawdown. Conviction (edge×meta) ranking + ¼-Kelly tilt refine on top.
    print(
        "\n  PHASE 7 — LONG-ONLY (edge rank base; vol-target; risk-parity; regime drawdown control)"
    )
    print(hdr)
    print("-" * 100)
    # The locked champion: top-decile edge-ranked, equal weight, raw returns.
    LO_eq = run_portfolio(panel, reg_daily, frac=0.10, long_short=False, rank_col="edge")
    # Vol-targeting the PLAIN champion — crypto crashes cluster, so causal deleveraging should
    # lift Sharpe AND cut the -63% drawdown without distorting the (working) edge ranking.
    LO_vt = run_portfolio(
        panel, reg_daily, frac=0.10, long_short=False, rank_col="edge", vol_target=0.30
    )
    LO_vtg = run_portfolio(
        panel,
        reg_daily,
        frac=0.10,
        long_short=False,
        rank_col="edge",
        vol_target=0.30,
        use_regime=True,
    )
    LO_g = run_portfolio(
        panel, reg_daily, frac=0.10, long_short=False, rank_col="edge", use_regime=True
    )
    # Broader breadth where risk-parity can actually diversify (top quintile, ~3-4 names).
    LO_iv20 = run_portfolio(
        panel,
        reg_daily,
        frac=0.20,
        long_short=False,
        rank_col="edge",
        inv_vol=True,
        vol_target=0.30,
    )
    _row("long-only top10% edge eq-weight (base)", LO_eq)
    _row("long-only top10% edge + vol-target", LO_vt)
    _row("long-only top10% edge + volTgt+regime", LO_vtg)
    _row("long-only top10% edge + regime", LO_g)
    _row("long-only top20% edge inv-vol+volTgt", LO_iv20)

    # ── Phase 7: NEUTRALIZED LONG/SHORT — does hedging the market unlock the short leg? ──
    print("\n  PHASE 7 — NEUTRALIZED LONG/SHORT (residualize returns; inverse-vol; ¼-Kelly)")
    print(hdr)
    print("-" * 100)
    cfgs = {}
    for mode in ("none", "market", "beta", "sector_beta"):
        m = run_portfolio(
            panel, reg_daily, frac=0.20, long_short=True, neutralize=mode, inv_vol=True, kelly=True
        )
        cfgs[mode] = m
        _row(f"L/S 20% {mode:<11} inv-vol+¼K", m)
    LSbest_key = max(cfgs, key=lambda k: cfgs[k].get("sharpe", -9))
    LSbest = cfgs[LSbest_key]

    print("-" * 100)
    print(
        f"  {'BENCHMARK equal-weight long':<38}{B['sharpe']:>7.2f}{B['total_return']:>+12.1%}"
        f"{B['ann_return']:>+10.1%}{B['max_dd']:>9.1%}{B['win_rate']:>7.0%}"
    )
    print("=" * 100)

    # ── GATE-7 verdict ────────────────────────────────────────────────────────────
    # Current deployable bar (locked baseline): long-only top-decile edge-ranked, Sharpe 0.83,
    # equal-weight drawdown -63.2% (regime-gated -45.4%). GATE-7: a Phase-7 config with net-of-cost
    # Sharpe > 0.83 AND max drawdown no worse than the current deployable (-63.2% equal-weight; we
    # additionally show it beats the tighter regime-gated -45.4%).
    BAR_SHARPE, BAR_DD = 0.83, -0.632
    candidates = {
        "LO top10% edge + vol-target": LO_vt,
        "LO top10% edge + volTgt+regime": LO_vtg,
        "LO top10% edge + regime": LO_g,
        "LO top20% edge inv-vol+volTgt": LO_iv20,
        f"L/S20% {LSbest_key}-neutral inv-vol": LSbest,
    }
    print("\n  GATE-7 — net-of-cost Sharpe > 0.83 AND maxDD ≤ current deployable (-63.2%)")
    print(f"  {'config':<38}{'Sharpe':>8}{'maxDD':>9}{'ann':>9}   verdict")
    print("-" * 100)
    winners = []
    for name, m in candidates.items():
        if m.get("periods", 0) < 3:
            continue
        s_ok = m["sharpe"] > BAR_SHARPE
        d_ok = abs(m["max_dd"]) <= abs(BAR_DD) + 1e-9
        passed = s_ok and d_ok
        if passed:
            winners.append((name, m))
        print(
            f"  {name:<38}{m['sharpe']:>8.2f}{m['max_dd']:>9.1%}{m['ann_return']:>+9.1%}   "
            f"{'PASS' if passed else 'fail'} "
            f"(Sharpe {'>' if s_ok else '<='}0.83, DD {'ok' if d_ok else 'worse'})"
        )
    print("-" * 100)
    if winners:
        best = max(winners, key=lambda kv: kv[1]["sharpe"])
        bm = best[1]
        dd_vs_gate = (
            "also ≤ regime-gated -45.4%"
            if abs(bm["max_dd"]) <= 0.454 + 1e-9
            else "between -45.4% and -63.2%"
        )
        print(f"  GATE-7 PASS ✓  best: {best[0]}")
        print(
            f"    Sharpe {bm['sharpe']:.2f} (> 0.83 deployable) | maxDD {bm['max_dd']:.1%} "
            f"(≤ -63.2%, {dd_vs_gate}) | ann {bm['ann_return']:+.1%}"
        )
        print(f"    vs benchmark Sharpe {B['sharpe']:.2f} / maxDD {B['max_dd']:.1%}")
    else:
        allc = {**candidates}
        bn = max(allc, key=lambda k: allc[k].get("sharpe", -9))
        bm = allc[bn]
        print(
            f"  GATE-7 not cleared by the strict AND. Best Phase-7 Sharpe: {bn} = "
            f"{bm['sharpe']:.2f} (maxDD {bm['max_dd']:.1%})."
        )
    print(
        f"  Neutralized L/S best ({LSbest_key}) Sharpe {LSbest['sharpe']:.2f} vs raw-L/S ~0.54 — "
        f"short leg {'now tradeable' if LSbest['sharpe'] > 0.54 else 'still a drag (long-biased universe)'}."
    )

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
