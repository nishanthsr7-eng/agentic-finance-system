"""
FLUX Prediction — Train the direction classifier (Layer 2a)
===========================================================
Trains ONE pooled, cross-sectional XGBoost model on all liquid symbols. Because every
feature is stationary / ratio-based (price-vs-SMA, ATR/price, frac-diff, ...), pooling is
valid and gives the model far more data than 30 tiny per-symbol models would.

What it does, honestly:
  1. Build features + triple-barrier labels per symbol (fd_order calibrated on EARLY history
     only → causal), pool them, sort by date.
  2. Evaluate OUT-OF-FOLD with PurgedWalkForwardSplit (leak-free) → directional accuracy, AUC.
  3. Compare against three baselines (persistence, always-up, majority). The model must beat
     them out-of-sample or the features need work.
  4. Calibrate probabilities (isotonic) and report ECE (calibration error).
  5. Refit on all data and persist model + calibrator + metadata to models/.

Run:  python backend/prediction/train.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.config import settings  # noqa: E402
from backend.db import get_history  # noqa: E402
from backend.prediction.conformal import ConformalBands  # noqa: E402
from backend.prediction.cv import PurgedWalkForwardSplit, overlap_count  # noqa: E402
from backend.prediction.features import (  # noqa: E402
    build_features_from_df,
    calibrate_fd_order,
    feature_columns,
)
from backend.prediction.garch import conditional_vol  # noqa: E402
from backend.prediction.labeling import make_labels  # noqa: E402

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Symbols to pool. Exclude USDT (stablecoin → no trend), macro series, and thin symbols.
# NIFTY: removed from the product (step 8a), but old local DBs still hold its history.
EXCLUDE = {"USDT", "SPX", "VIX", "TNX", "NIFTY"}
MIN_EVENTS = 800

HORIZON, PT, SL = 5, 2.0, 2.0  # triple-barrier config (5-day, ±2·vol)
N_SPLITS, EMBARGO = 6, HORIZON + 2  # embargo ≥ horizon so no label leaks across the seam


# ── Dataset assembly ───────────────────────────────────────────────────────────
async def _all_symbols() -> list[str]:
    from backend.db import history_summary

    return [
        r["symbol"]
        for r in await history_summary()
        if r["symbol"] not in EXCLUDE and r["rows"] >= MIN_EVENTS
    ]


async def build_macro() -> pd.DataFrame:
    """Cross-asset / regime features from SPX, VIX, TNX — all causal. Indexed by date."""

    async def _series(sym):
        rows = await get_history(sym)
        s = pd.DataFrame(rows)
        s.index = pd.to_datetime(s["date"])
        return s["close"].astype(float)

    spx, vix, tnx = await _series("SPX"), await _series("VIX"), await _series("TNX")
    m = pd.DataFrame(index=spx.index)
    lp = np.log(spx)
    m["spx_ret1"] = lp.diff()
    m["spx_ret5"] = lp.diff(5)
    m["spx_ret20"] = lp.diff(20)
    m["vix_z"] = (vix - vix.rolling(252, min_periods=60).mean()) / vix.rolling(
        252, min_periods=60
    ).std()
    m["vix_chg"] = np.log(vix / vix.shift())
    m["tnx_chg"] = tnx.diff()
    m = m.fillna(0)  # Handle small seeded datasets where rolling windows would be entirely NaN

    # FRED macro block (term/credit spread, rates, CPI YoY) — SELF-GATED OFF for the cross-sectional
    # ranker. Phase-1 finding (scripts/experiments/gate1_fred_macro_sharpe_ablation.py): macro is cross-sectionally CONSTANT
    # (term_spread is identical for every symbol on a day), so it adds no within-date ranking power but
    # injects common-mode noise that HALVES the deployable portfolio Sharpe (0.906 -> 0.457) despite a
    # tiny +AUC. It passes the AUC-only GATE-1 but fails the deployable metric, so per the honesty
    # contract it stays OFF here. Macro's proper home is the regime/sizing layer (Phase 6), not this
    # per-symbol feature set. Set FLUX_FRED_RANKER=1 to re-enable for experimentation only.
    import os

    if settings.FRED_API_KEY and os.getenv("FLUX_FRED_RANKER", "0") == "1":
        try:
            from backend.prediction.fred import build_fred_features

            fred = await build_fred_features(start="1990-01-01")  # all series exist by 1990
            if not fred.empty:
                m = m.join(fred.reindex(m.index, method="ffill"))
                print(f"  + FRED macro features (FLUX_FRED_RANKER=1): {list(fred.columns)}")
        except Exception as exc:  # never let macro enrichment break training
            print(f"  (FRED macro skipped: {exc})")
    return m


async def load_dataset(symbols: list[str] | None = None):
    """Pool features + labels across symbols. Returns X, y, w, t1, meta(fd_orders)."""
    if symbols is None:
        symbols = await _all_symbols()

    macro = await build_macro()

    # Phase-2 crypto-native block (funding/OI/DVOL/on-chain/sentiment/BTC-beta) — SELF-GATED OFF for
    # the production (full-universe) model. Evidence (see crypto_features.py docstring + the gate2_crypto_features_*
    # scripts): AUC GATE-2 FAILS (best subset +0.0055 < +0.01); crypto-ONLY-book long-only Sharpe
    # HELPS (+0.10..+0.22); but the FULL 33-symbol book Sharpe HURTS (0.835 -> 0.670) because the
    # pooled model is shared and equities (~2.7x the crypto rows) get only zero/noise from these cols.
    # So it stays OFF here. It is a net positive ONLY on a crypto-only book — flip FLUX_CRYPTO_FEATURES=1
    # for that case (or revisit as a Phase-6 regime/sizing input).
    # BTC's daily log-return drives btc_lead_lag; fetched once, close-of-D return is causal for D+1.
    import os

    crypto_on = os.getenv("FLUX_CRYPTO_FEATURES", "0") == "1"
    btc_ret = None
    if crypto_on:
        from backend.prediction.crypto_features import compute_crypto_features

        brows = await get_history("BTC")
        if brows:
            b = pd.DataFrame(brows)
            b.index = pd.to_datetime(b["date"])
            btc_ret = np.log(b["adj_close"].astype(float)).diff()

    # Phase-4 equity fundamentals & events block (earnings surprise/drift, revenue revision, valuation
    # z, accruals) — point-in-time from SEC filings (datasources/sec_fundamentals.py). Like the crypto
    # block it returns finite, neutral-(0)-filled columns for EVERY symbol (all-zero for crypto / any
    # symbol without SEC fundamentals), so the dropna() below never drops a row on their account; signal
    # lives only on the equity subset. SELF-GATED OFF: GATE-4 (scripts/experiments/gate4_equity_fundamentals_eval.py) FAILS —
    # on the identical pooled panel the equity-subset OOF AUC is a within-noise tie vs Phase-3
    # (0.5107 -> 0.5099, -0.0008), no structural lift. Quarterly fundamentals are slowly-varying step
    # functions with little directional power at the 5-day horizon, so per the honesty contract this
    # stays OFF for the cross-sectional ranker. (Aside: the joint fit nudged pooled +0.0023 / crypto
    # +0.0046 via shared-tree coupling, but GATE-4 is the equity metric and it didn't lift.) The block
    # is leak-safe + tested (test_equity_features.py); flip FLUX_EQUITY_FEATURES=1 to experiment.
    equity_on = os.getenv("FLUX_EQUITY_FEATURES", "0") == "1"
    if equity_on:
        from backend.prediction.equity_features import compute_equity_features

    # Phase-5 multi-source sentiment block (GDELT tone + Alpha Vantage news sentiment, historical &
    # point-in-time). Like the crypto/equity blocks it returns finite, neutral-(0)-filled columns for
    # EVERY symbol (all-zero wherever a stream has no coverage), so the dropna() below never drops a row
    # on its account; signal lives only where news history exists. SELF-GATED OFF — flip
    # FLUX_SENTIMENT_FEATURES=1 to opt in / run GATE-5 (scripts/experiments/gate5_sentiment_eval.py). The live FinBERT/
    # CryptoBERT tilt in predict.py is independent of this and unaffected. Leak-safe + tested
    # (test_sentiment_features.py).
    sentiment_on = os.getenv("FLUX_SENTIMENT_FEATURES", "0") == "1"
    if sentiment_on:
        from backend.prediction.sentiment_features import compute_sentiment_features  # noqa: F401

    frames, fd_orders = [], {}
    for sym in symbols:
        rows = await get_history(sym)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        # Calibrate frac-diff order on the EARLIEST 50% only (causal, no future peek).
        early = df.iloc[: max(250, len(df) // 2)]
        d = calibrate_fd_order(pd.Series(early["adj_close"].values))
        fd_orders[sym] = d

        feat = build_features_from_df(df, fd_order=d)
        # Merge macro (ffill covers crypto weekends with the last market close) + relative strength.
        feat = feat.join(macro.reindex(feat.index, method="ffill"))
        feat["rs_1"] = feat["logret_1"] - feat["spx_ret1"]  # stock vs market (cross-sectional edge)
        feat["rs_5"] = feat["logret_5"] - feat["spx_ret5"]
        # Phase-2 crypto-native block (funding/OI/DVOL/on-chain/sentiment/BTC-beta). Returns finite,
        # neutral-(0)-filled columns for EVERY symbol — all-zero for equities — so the dropna() below
        # never drops a row on their account. Signal lives only on the crypto subset.
        if crypto_on:
            feat = feat.join(compute_crypto_features(sym, feat, btc_ret=btc_ret))
        # Phase-4 equity fundamentals block (point-in-time SEC). Neutral-(0) for non-equity, so the
        # dropna() below never drops a crypto row on its account. Signal lives on the equity subset.
        if equity_on:
            feat = feat.join(compute_equity_features(sym, feat))
        # Phase-5 multi-source sentiment block (GDELT/AV). Neutral-(0) wherever there is no news
        # coverage, so the dropna() below never drops a row on its account.
        if sentiment_on:
            feat = feat.join(compute_sentiment_features(sym, feat))
        feat = feat.dropna()
        ev = make_labels(feat["close"], horizon=HORIZON, pt=PT, sl=SL)
        ev = ev[ev["label"] != 0]  # binary UP/DOWN (tiny FLAT dropped)
        if len(ev) < MIN_EVENTS:
            continue

        cols = feature_columns(feat)
        part = feat.loc[ev.index, cols].copy()
        part["_y"] = (ev["label"] > 0).astype(int)  # 1 = UP, 0 = DOWN
        part["_w"] = ev["weight"].values
        part["_t1"] = pd.to_datetime(ev["t1"].values)
        part["_ret"] = ev["ret"].values  # realized triple-barrier return
        part["_sym"] = sym
        # GARCH conditional vol scaled to the horizon → per-event band width for conformal.py.
        # Causal σ_t (depends on returns < t); shapes the band, conformal guarantees coverage.
        sigma = conditional_vol(feat["close"]).reindex(ev.index)
        part["_scale"] = (sigma * np.sqrt(HORIZON)).values
        part["_date"] = pd.to_datetime(part.index)
        frames.append(part)
        print(f"  {sym:<6} {len(part):>6} events  (fd_order={d})")

    data = pd.concat(frames).sort_values("_date").reset_index(drop=True)
    feat_cols = [c for c in data.columns if not c.startswith("_")]
    X = data[feat_cols]
    X.index = data["_date"]
    return X, data["_y"].values, data["_w"].values, data["_t1"], feat_cols, fd_orders, data


def generate_oof_signals(X, y, w, t1, feat_cols):
    """
    Leak-free OOF signals for the backtest: calibrated primary prob + meta prob, each as a
    full-length array aligned to X (NaN where a row was never in an OOF test fold).
    Returns (primary_cal, meta_full, iso, meta_report).
    """
    r = evaluate_oof(X, y, w, t1, feat_cols)
    iso = fit_calibrator(r["_oof_p"], r["_oof_y"])  # serving calibrator (all OOF rows)
    # Signals use calibrators fit on strictly earlier folds only (NaN for the first fold),
    # so the backtest never sees a calibration fitted on its own outcomes.
    primary_cal = calibrate_oos(r["_oof_p_full"], y, r["_oof_fold_full"])

    oof_meta, mask, mrep = evaluate_meta(X, y, w, t1, feat_cols, primary_cal)
    meta_full = np.full(len(primary_cal), np.nan)
    meta_full[mask] = oof_meta
    return primary_cal, meta_full, iso, mrep


# ── Model factory ──────────────────────────────────────────────────────────────
def _make_model():
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=2.0,
        reg_alpha=0.0,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )


def _make_regressor():
    """Point regressor for the h-day return (Layer 2b magnitude head feeding the conformal band)."""
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=2.0,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )


def evaluate_conformal(X, ret, scale, t1, w, feat_cols, alphas=(0.2, 0.1)):
    """
    Leak-free conformal band calibration (Layer 2b):
      • OOF h-day-return regressor via the SAME purged splitter (no overlap leakage),
      • normalise residuals by the per-event GARCH scale,
      • fit conformal half-widths and report realised OOF coverage (must ≈ 1-alpha).
    Returns (bands, report). The point regressor's MAE is compared to a predict-zero baseline
    so we can see whether the magnitude head adds anything over "no move".
    """
    cv = PurgedWalkForwardSplit(n_splits=N_SPLITS, embargo=EMBARGO)
    oof = np.full(len(ret), np.nan)
    for tr, te in cv.split(X, t1):
        m = _make_regressor()
        m.fit(X.iloc[tr][feat_cols], ret[tr], sample_weight=w[tr])
        oof[te] = m.predict(X.iloc[te][feat_cols])

    mask = ~np.isnan(oof) & np.isfinite(scale) & (scale > 0)
    resid = ret[mask] - oof[mask]
    bands = ConformalBands.fit(resid, scale[mask], horizon=HORIZON, alphas=alphas)
    report = {
        "n_calib": int(mask.sum()),
        "mae": float(np.abs(resid).mean()),
        "mae_predict_zero": float(np.abs(ret[mask]).mean()),
        "half_width_sigma": {float(a): bands.q[float(a)] for a in alphas},
        "coverage": {float(a): bands.coverage[float(a)] for a in alphas},
    }
    return bands, report


# ── Metrics ────────────────────────────────────────────────────────────────────
def _ece(y_true, p, bins=10) -> float:
    """Expected Calibration Error."""
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        e += abs(p[m].mean() - y_true[m].mean()) * m.sum() / len(p)
    return e


def evaluate_oof(X, y, w, t1, feat_cols):
    """Leak-free out-of-fold evaluation + baseline comparison."""
    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

    cv = PurgedWalkForwardSplit(n_splits=N_SPLITS, embargo=EMBARGO)

    oof_p = np.full(len(y), np.nan)
    fold = np.full(len(y), -1)  # which walk-forward test fold
    leak = 0
    for k, (tr, te) in enumerate(cv.split(X, t1)):
        leak += overlap_count(X, t1, tr, te)
        m = _make_model()
        m.fit(X.iloc[tr][feat_cols], y[tr], sample_weight=w[tr])
        oof_p[te] = m.predict_proba(X.iloc[te][feat_cols])[:, 1]
        fold[te] = k

    mask = ~np.isnan(oof_p)
    yt, p = y[mask], oof_p[mask]
    pred = (p >= 0.5).astype(int)

    # Baselines on the same OOF rows
    logret1 = X["logret_1"].values[mask]
    persist = (logret1 > 0).astype(int)  # tomorrow's dir = today's return sign
    always_up = np.ones_like(yt)
    majority = np.full_like(yt, int(round(y.mean())))

    cm = confusion_matrix(yt, pred, labels=[0, 1])  # rows = actual DOWN/UP, cols = pred DOWN/UP
    return {
        "leak_overlaps": int(leak),
        "n_oof": int(mask.sum()),
        "model_acc": float((pred == yt).mean()),
        "model_auc": float(roc_auc_score(yt, p)),
        "model_f1": float(f1_score(yt, pred)),
        "confusion": cm.tolist(),
        "persistence_acc": float((persist == yt).mean()),
        "always_up_acc": float((always_up == yt).mean()),
        "majority_acc": float((majority == yt).mean()),
        "ece_raw": float(_ece(yt, p)),
        "_oof_p": p,
        "_oof_y": yt,
        "_oof_fold": fold[mask],
        "_oof_p_full": oof_p,
        "_oof_fold_full": fold,
        "_mask": mask,
    }


def evaluate_meta(X, y, w, t1, feat_cols, p_cal_full):
    """
    Meta-labeling (López de Prado): a 2nd model predicts whether the PRIMARY call should be
    acted on. Target = 1 if the primary direction was correct. Feature set = price/macro
    features + the CALIBRATED primary probability. Evaluated leak-free with purged CV.
    `p_cal_full` is the out-of-sample calibrated primary probability (calibrate_oos), NaN where
    no earlier fold exists; only those rows are used.
    Returns OOF meta probabilities aligned to the masked rows + a precision/coverage report.
    """
    from sklearn.metrics import roc_auc_score

    mask = ~np.isnan(p_cal_full)
    Xm = X[mask].copy()
    p_primary = p_cal_full[mask]  # calibrated → matches serving
    Xm["primary_p"] = p_primary
    ym = ((p_primary >= 0.5).astype(int) == y[mask]).astype(int)  # 1 = primary was right
    t1m = t1[mask]
    wm = w[mask]
    meta_cols = feat_cols + ["primary_p"]

    cv = PurgedWalkForwardSplit(n_splits=N_SPLITS, embargo=EMBARGO)
    oof_meta = np.full(len(ym), np.nan)
    for tr, te in cv.split(Xm, t1m):
        m = _make_model()
        m.fit(Xm.iloc[tr][meta_cols], ym[tr], sample_weight=wm[tr])
        oof_meta[te] = m.predict_proba(Xm.iloc[te][meta_cols])[:, 1]

    mm = ~np.isnan(oof_meta)
    primary_pred = (p_primary[mm] >= 0.5).astype(int)
    correct = (primary_pred == y[mask][mm]).astype(int)
    meta_p = oof_meta[mm]

    # Precision of the primary call on the subset the meta-model says "act" (meta_p > thr),
    # vs trading everything. This is the meta-labeling payoff.
    report = {
        "meta_auc": float(roc_auc_score(correct, meta_p)) if correct.std() else float("nan"),
        "act_all_precision": float(correct.mean()),
        "buckets": [],
    }
    for thr in (0.50, 0.55, 0.60):
        act = meta_p >= thr
        cov = float(act.mean())
        prec = float(correct[act].mean()) if act.any() else float("nan")
        report["buckets"].append({"thr": thr, "coverage": cov, "precision": prec})
    return oof_meta, mask, report


def calibrate_oos(p_full, y, fold_full, min_rows: int = 100) -> np.ndarray:
    """
    Out-of-sample calibrated probabilities: rows of walk-forward fold k get the isotonic
    calibrator fit on folds < k. NaN for rows never in a test fold, for fold 0 (nothing earlier)
    and for folds with fewer than `min_rows` earlier rows.
    """
    p_full = np.asarray(p_full, dtype=float)
    out = np.full(len(p_full), np.nan)
    have = ~np.isnan(p_full) & (fold_full >= 0)
    for k in sorted(np.unique(fold_full[have])):
        train, test = have & (fold_full < k), have & (fold_full == k)
        if train.sum() < min_rows:
            continue
        out[test] = fit_calibrator(p_full[train], y[train]).predict(p_full[test])
    return out


def ece_calibrated_oos(p, y, fold) -> tuple[float, int]:
    """
    Out-of-sample ECE of the isotonic calibrator. For each walk-forward test fold k >= 1 the
    calibrator is fit on the OOF predictions of the strictly earlier folds and scored on fold k.
    Scoring it on the predictions it was fit on is in-sample: isotonic regression reproduces
    its own training data almost exactly, which is why that number came out ~1e-16.
    Each fold is scored on its own and the per-fold ECEs are averaged (row-weighted). Pooling
    the folds first would let one period's over-estimate cancel another's under-estimate inside
    the same bin; with almost every calibrated probability in the 0.5-0.6 bin, that turned
    0.01-0.035 per fold into a pooled ~0.0001.
    Returns (row-weighted mean per-fold ECE, rows scored).
    """
    ks = sorted(k for k in np.unique(fold) if k >= 0)
    eces, ns = [], []
    for k in ks[1:]:
        train, test = (fold >= 0) & (fold < k), fold == k
        if train.sum() < 100 or not test.any():
            continue
        pc = fit_calibrator(p[train], y[train]).predict(p[test])
        eces.append(_ece(y[test], pc))
        ns.append(int(test.sum()))
    if not ns:
        return float("nan"), 0
    return float(np.average(eces, weights=ns)), int(sum(ns))


def library_versions() -> dict:
    """Versions that wrote the artifacts; backend/requirements-slim.lock must match."""
    import sklearn
    import xgboost

    return {
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def fit_calibrator(p, y):
    """Isotonic mapping raw prob → calibrated prob, fit on OOF predictions."""
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p, y)
    return iso


async def arima_baseline(symbols, horizon=HORIZON, max_points=40, window=500, start="2008-01-01"):
    """
    Classical ARIMA(1,0,0) directional baseline. For a sample of dates per symbol it fits on a
    trailing window of log-returns, forecasts `horizon` steps, and predicts the sign of the
    cumulative move. If the ML model can't beat this, the ML is overfit (López de Prado).
    Sampled + bounded so it stays cheap.
    """
    import warnings

    import numpy as _np
    from statsmodels.tsa.arima.model import ARIMA

    correct = total = ups = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sym in symbols:
            rows = await get_history(sym, start)
            if len(rows) < window + horizon + 50:
                continue
            df = pd.DataFrame(rows)
            close = df["adj_close"].astype(float).values
            logret = _np.diff(_np.log(close))
            lo, hi = window, len(close) - horizon - 1
            if hi <= lo:
                continue
            step = max(horizon, (hi - lo) // max_points)
            for t in range(lo, hi, step):
                try:
                    fit = ARIMA(logret[t - window : t], order=(1, 0, 0)).fit()
                    fc = fit.forecast(steps=horizon).sum()
                except Exception:
                    continue
                actual = close[t + horizon] / close[t] - 1.0
                if (fc > 0) == (actual > 0):
                    correct += 1
                ups += int(actual > 0)
                total += 1
    if not total:
        return float("nan"), float("nan"), 0
    # Also return the always-up accuracy ON THE SAME SAMPLE so ARIMA's real edge is visible.
    return float(correct / total), float(ups / total), total


# ── Main ───────────────────────────────────────────────────────────────────────
async def main():
    import joblib

    t0 = time.time()
    print("Loading + labeling pooled dataset...")
    X, y, w, t1, feat_cols, fd_orders, _data = await load_dataset()
    print(
        f"\nPooled: {len(X):,} events x {len(feat_cols)} features "
        f"| UP={y.mean():.1%} | {X.index.min().date()} -> {X.index.max().date()}\n"
    )

    print("Out-of-fold evaluation (purged walk-forward)...")
    r = evaluate_oof(X, y, w, t1, feat_cols)
    print(f"  purge leak overlaps : {r['leak_overlaps']}  (must be 0)")
    print(f"  OOF samples         : {r['n_oof']:,}")
    print("  ---- directional accuracy ----")
    print(
        f"  MODEL               : {r['model_acc']:.4f}   (AUC {r['model_auc']:.4f}, F1 {r['model_f1']:.4f})"
    )
    print(f"  baseline persistence: {r['persistence_acc']:.4f}")
    print(f"  baseline always-up  : {r['always_up_acc']:.4f}")
    print(f"  baseline majority   : {r['majority_acc']:.4f}")
    print("  baseline ARIMA(1,0,0): computing (sampled)...", end="", flush=True)
    arima_acc, arima_drift, arima_n = await arima_baseline(list(fd_orders.keys()))
    arima_edge = arima_acc - arima_drift
    print(
        f"\r  baseline ARIMA(1,0,0): {arima_acc:.4f}   "
        f"(vs drift {arima_drift:.4f} on same sample -> edge {arima_edge:+.4f}; n={arima_n})"
    )
    # ARIMA is on a different sample, so compare on EDGE-over-drift, not raw accuracy.
    best_base = max(r["persistence_acc"], r["always_up_acc"], r["majority_acc"])
    edge = r["model_acc"] - best_base
    print(
        f"  EDGE vs best same-sample baseline: {edge:+.4f}  -> "
        f"{'PASS' if edge > 0 else 'FAIL (iterate features)'}"
    )
    print(
        f"  (model AUC {r['model_auc']:.4f} > 0.5 confirms ranking skill; "
        f"ARIMA edge-over-drift {arima_edge:+.4f})"
    )

    # Confusion matrix (rows = actual, cols = predicted)
    cm = r["confusion"]
    print("  ---- confusion matrix (OOF) ----")
    print("               pred_DOWN  pred_UP")
    print(f"   actual_DOWN  {cm[0][0]:>8}  {cm[0][1]:>7}")
    print(f"   actual_UP    {cm[1][0]:>8}  {cm[1][1]:>7}")

    # Calibration
    iso = fit_calibrator(r["_oof_p"], r["_oof_y"])  # serving calibrator: all OOF rows
    ece_cal_insample = _ece(r["_oof_y"], iso.predict(r["_oof_p"]))
    ece_cal, ece_cal_n = ece_calibrated_oos(r["_oof_p"], r["_oof_y"], r["_oof_fold"])
    print(f"\n  calibration ECE raw            : {r['ece_raw']:.4f}")
    print(
        f"  calibration ECE cal (OOS)      : {ece_cal:.4f}  on {ece_cal_n:,} later-fold rows (lower is better)"
    )
    print(
        f"  calibration ECE cal (in-sample): {ece_cal_insample:.2e}  (fit = scored rows; not a real measure)"
    )

    # Selective prediction: does accuracy rise when we only act on confident calls?
    # This is where the real, tradeable edge lives (meta-labeling / sizing).
    print("\n  Selective prediction (act only on most-confident calls):")
    p, yt = r["_oof_p"], r["_oof_y"]
    conf = np.abs(p - 0.5)
    base = max(yt.mean(), 1 - yt.mean())  # always-pick-majority accuracy
    print(f"    {'coverage':>10}{'directional_acc':>18}{'vs always-up':>15}")
    for frac in (1.0, 0.5, 0.25, 0.10, 0.05):
        k = max(50, int(len(p) * frac))
        idx = np.argsort(conf)[-k:]  # top-k most confident
        acc = ((p[idx] >= 0.5).astype(int) == yt[idx]).mean()
        print(f"    {frac:>9.0%}{acc:>18.4f}{acc - base:>+15.4f}")

    # ── Meta-labeling model (precision booster on the traded subset) ───────────
    print("\n  Meta-labeling (2nd model: 'should we act on the primary call?'):")
    p_cal_oos = calibrate_oos(r["_oof_p_full"], y, r["_oof_fold_full"])
    _oof_meta, mmask, mrep = evaluate_meta(X, y, w, t1, feat_cols, p_cal_oos)
    mrep["primary_calibration"] = "out-of-sample (isotonic fit on earlier folds only)"
    print(f"    meta-model AUC      : {mrep['meta_auc']:.4f}")
    print(f"    act-on-everything   : precision {mrep['act_all_precision']:.4f}")
    print(f"    {'meta_thr':>10}{'coverage':>11}{'precision':>11}")
    for b in mrep["buckets"]:
        print(f"    {b['thr']:>10.2f}{b['coverage']:>11.1%}{b['precision']:>11.4f}")

    # ── Conformal prediction bands (Layer 2b: guaranteed-coverage return interval) ──
    print("\n  Conformal bands (GARCH-shaped, purged-OOF calibrated):")
    ret = _data["_ret"].values.astype(float)
    scale = _data["_scale"].values.astype(float)
    bands, crep = evaluate_conformal(X, ret, scale, t1, w, feat_cols)
    print(f"    calib events        : {crep['n_calib']:,}")
    print(
        f"    regressor MAE       : {crep['mae']:.4f}  (predict-zero {crep['mae_predict_zero']:.4f})"
    )
    print(f"    {'target':>10}{'half_width':>13}{'realised_cov':>15}")
    for a in (0.2, 0.1):
        cov, q = crep["coverage"][a], crep["half_width_sigma"][a]
        ok = "PASS" if abs(cov - (1 - a)) <= 0.03 else "WIDE/NARROW"
        print(f"    {1 - a:>9.0%}{q:>11.2f}s{cov:>14.1%}  {ok}")

    # Refit final models on ALL data and persist
    print("\nRefitting final models on all data + saving artifacts...")
    final = _make_model()
    final.fit(X[feat_cols], y, sample_weight=w)
    final.save_model(MODELS_DIR / "xgb_primary.json")
    joblib.dump(iso, MODELS_DIR / "calibrator.pkl")

    # Final meta-model: features + calibrated primary prob → P(primary correct)
    Xmeta = X[feat_cols].copy()
    Xmeta["primary_p"] = iso.predict(final.predict_proba(X[feat_cols])[:, 1])
    ymeta = ((Xmeta["primary_p"].values >= 0.5).astype(int) == y).astype(int)
    meta_model = _make_model()
    meta_model.fit(Xmeta, ymeta, sample_weight=w)
    meta_model.save_model(MODELS_DIR / "xgb_meta.json")

    # Final return regressor (point estimate for the band) + persisted conformal half-widths.
    final_reg = _make_regressor()
    final_reg.fit(X[feat_cols], ret, sample_weight=w)
    final_reg.save_model(MODELS_DIR / "xgb_return.json")
    bands.save(MODELS_DIR / "conformal.pkl")

    meta = {
        "feature_columns": feat_cols,
        "meta_columns": feat_cols + ["primary_p"],
        "fd_orders": fd_orders,
        "horizon": HORIZON,
        "pt": PT,
        "sl": SL,
        "n_splits": N_SPLITS,
        "embargo": EMBARGO,
        "metrics": {k: v for k, v in r.items() if not k.startswith("_")}
        | {
            "ece_cal": ece_cal,
            "ece_cal_n": ece_cal_n,
            "ece_cal_insample": ece_cal_insample,
            "arima_acc": arima_acc,
            "arima_drift": arima_drift,
            "arima_edge": arima_edge,
            "arima_n": arima_n,
            "arima_note": (
                "sampled ARIMA(1,0,0): sign of the raw close-to-close "
                f"{HORIZON}-day return on its own sample since 2008, not "
                "the triple-barrier label or rows the model is scored on; "
                "compare arima_edge (vs drift on the same sample), not "
                "arima_acc vs model_acc"
            ),
        },
        "meta_report": mrep,
        "conformal_report": crep,
        "conformal_alphas": [0.2, 0.1],
        "trained_at": int(time.time() * 1000),
        "n_events": int(len(X)),
        "data_end": str(X.index.max().date()),
        "versions": library_versions(),
    }
    # Keep the earlier gate decision records (history of what was tried and why).
    old_meta = MODELS_DIR / "model_meta.json"
    if old_meta.exists():
        prev = json.loads(old_meta.read_text())
        meta |= {k: v for k, v in prev.items() if k.startswith("gate")}
    (MODELS_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(
        "  saved: xgb_primary.json, xgb_meta.json, xgb_return.json, "
        "calibrator.pkl, conformal.pkl, model_meta.json"
    )
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
