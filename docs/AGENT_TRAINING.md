# How the Prediction Agent Is Trained

This document describes the design, algorithms, and end-to-end training pipeline
of **FLUX-X**, the prediction agent in `backend/prediction/`. It separates what
is **served today** from what was **built, tested and switched off**: most of the
research components failed their out-of-sample gate, and the doc says so.

> **Honesty contract.** Markets are near-efficient; no model reliably predicts
> exact prices. The agent is engineered to be *measurably better out-of-sample*
> than a typical tutorial pipeline - not by a bigger neural net, but by refusing
> to leak the future, labelling tradeable moves, calibrating its confidence, and
> backtesting net of costs. Realistic daily directional accuracy has a hard
> ceiling (~0.55–0.58). The edge comes from **selectivity + orthogonal
> information + cross-sectional construction**, and every component must beat its
> baseline on purged, embargoed, out-of-fold (OOF) data or it stays gated OFF.

---

## 1. Architecture

### What is served today

```
LLM VERIFIER (Groq in production, Ollama locally)
  Reads fresh news for the day's top calls. May VETO or DOWNGRADE confidence;
  can NEVER raise it above the calibrated number.
        ▲
NEWS TILT
  Live headline sentiment (LLM in production, FinBERT locally) nudges confidence
  when it agrees or disagrees with the model's direction.
        ▲
SIGNAL
  One pooled XGBoost direction model over 29 crypto + US-stock symbols
  → meta-model (P the call is correct) → isotonic calibration
  → GARCH-scaled conformal 80% / 90% return bands.
        ▲
FEATURES (42, all causal)
  Price and technicals (SMA, MACD, RSI, ADX, Bollinger, ATR, volume),
  frac-diff log price, returns, calendar terms, market context
  (S&P 500 returns, VIX, 10-year yield change) and relative strength.

FLYWHEEL: log → resolve at t+5 → live calibration → weekly drift retrain.
```

### Built and switched off

Each of these was implemented and tested on purged out-of-fold data, then left
off because it did not clear its gate (see §6):

- per-asset-class expert models (crypto vs equity)
- FRED macro, crypto-native (funding, open interest, DVOL, on-chain) and SEC
  fundamental feature blocks
- sentiment as a model feature (it is used only as the live tilt above)
- the regime-conditional stacking ensemble (HMM + ElasticNet second learner)

---

## 2. Algorithms and Techniques

| Technique | Purpose | Implementation |
|---|---|---|
| **Triple-barrier labeling** | Label *tradeable* moves (profit-take / stop / time-out) scaled by volatility, not arbitrary fixed-% bins | `labeling.py` |
| **Meta-labeling** | A second model predicts whether to *act* on the primary direction call; decouples side from precision/size | `labeling.py`, `train.py` |
| **Fractional differentiation** | Stationary features that *retain memory*; the minimum order `d` that passes an ADF test | `features.py` |
| **Purged + embargoed walk-forward CV** | Removes overlapping-label leakage - the #1 cause of fake accuracy | `cv.py` (`PurgedWalkForwardSplit`) |
| **Sample-uniqueness weighting** | Down-weights overlapping/crowded periods so the model does not over-count them | `labeling.py`, `train.py` |
| **Gradient-boosted trees (XGBoost)** | Direction classifier; tabular GBMs beat deep nets on noisy daily data | `train.py` |
| **ElasticNet base learner** | A decorrelated linear learner that diversifies the ensemble | `ensemble.py` |
| **Isotonic probability calibration** | Makes "62%" mean ~62% historically (ECE-checked) | `train.py` |
| **Conformal prediction bands** | Distribution-free, guaranteed-coverage intervals on a *purged* splitter | `conformal.py` |
| **GARCH(1,1) volatility** | Shapes the conformal band (widens in high-vol clusters) | `garch.py` |
| **Gaussian HMM regime detection** | Trend / chop / risk-off, shown on the Advisor and used as a size scale | `regime.py` |
| **Regime-conditional stacking** | A mixture-of-experts blend whose weights differ per regime; self-gated, **off** (§6) | `ensemble.py` |
| **Cross-sectional construction** | Rank, vol-target; market/sector-neutral long-short tested and rejected (Sharpe 0.39 vs 0.83 long-only) | `portfolio.py` |
| **Fractional-Kelly sizing** | Cost-aware position sizing from the *calibrated* win-probability (¼-Kelly) | `sizing.py` |
| **LLM verifier** | One-way veto/downgrade against fresh narrative | `agent.py` |
| **Honesty baselines** | Persistence, always-up and ARIMA every run; Chronos zero-shot locally | `train.py`, `baselines.py` |

---

## 3. Feature Engineering (Layer 1)

All features are **causal** - row `t` uses only data ≤ `t`. Built per symbol and
pooled cross-sectionally.

- **Served block** (`features.py`, the 42 columns in `models/model_meta.json`):
  moving averages, MACD, RSI, ADX, stochastics, Williams %R, Bollinger width and
  %B, ATR, realised volatility, OBV slope, MFI, 1/5/10/20-day log returns, gap
  and intraday range, calendar terms, fractional-differentiated log price,
  S&P 500 returns, VIX level and change, 10-year yield change, relative strength.

Built but gated off (code kept, not in the served model):

- **Crypto block** (`crypto_features.py`): funding level / z-score / sign-flips,
  open-interest change and OI/volume, DVOL level and term, on-chain NVT and
  address growth, Fear & Greed, BTC lead-lag.
- **Equity block** (`equity_features.py`, `fred.py`, `options.py`): FRED
  term/credit spreads and rates, options IV/skew, fundamentals (earnings
  surprise, revision, valuation z-score) aligned to **filing date** (not period
  date) to avoid leakage.
- **Sentiment block** (`sentiment.py`, `sentiment_features.py`): FinBERT for
  equity news, CryptoBERT for crypto, GDELT tone - all time-aligned ≤ `t`.

Data-source loaders in `backend/prediction/datasources/` convert each staged
dataset (see [docs/DATASETS.md](DATASETS.md)) into tidy daily frames keyed by
`(symbol, date)`.

---

## 4. Labeling and Cross-Validation

### Triple-barrier labels (`labeling.py`)

From each event `t`, set a profit-take and stop-loss barrier scaled by recent
(EWMA) volatility, plus a vertical time-out barrier at horizon `h`. The label is
*which barrier is hit first*. Default configuration in `train.py`:

```
HORIZON = 5 days,  PT = 2.0·vol,  SL = 2.0·vol
```

### Meta-labels

The primary model predicts side (±1). A binary `act` target -
`1 if the primary side matched the realised triple-barrier outcome else 0` -
trains the meta-model. At inference: `side = primary`, `size ∝ meta_prob`, and a
trade is taken only when `meta_prob` clears the act-gate (e.g. ≥ 0.60).

### Purged walk-forward CV (`cv.py`)

Triple-barrier labels overlap in time, so standard CV leaks. `PurgedWalkForwardSplit`:

- Orders folds in time (walk-forward; never random K-fold).
- **Purges** training samples whose label window overlaps the test window.
- **Embargoes** a tail buffer after each test fold (`EMBARGO = HORIZON + 2`, so
  no label leaks across the seam).

All reported metrics come **only** from these out-of-sample folds.

---

## 5. The Training Run (`backend/prediction/train.py`)

The default pooled trainer trains ONE cross-sectional XGBoost model over all
liquid symbols. Because every feature is stationary / ratio-based, pooling is
valid and gives far more data than per-symbol models.

What it does, in order:

1. **Assemble the dataset** - for each symbol with ≥ `MIN_EVENTS` (800) rows,
   build features (with the frac-diff order calibrated on **early history only**
   → causal) and triple-barrier labels, pool them, and sort by date. Stablecoins
   (USDT) and macro series (SPX, VIX, TNX) are excluded from the trainable set.
2. **Evaluate out-of-fold** with `PurgedWalkForwardSplit` (6 splits) → directional
   accuracy and AUC, leak-free.
3. **Compare against baselines** - persistence, always-up, and majority. The
   model must beat them out-of-sample or the features need work.
4. **Calibrate** probabilities (isotonic) and report **Expected Calibration
   Error (ECE)** out of sample: each walk-forward fold is scored by a calibrator
   fit only on earlier folds (`ece_calibrated_oos`), and the per-fold ECEs are
   averaged. Scoring the calibrator on the rows it was fit on gives ~0 and means
   nothing. The meta-model report and the backtest/ensemble signals use the same
   earlier-folds-only calibration (`calibrate_oos`); only the serving calibrator
   is fit on all out-of-fold rows.
5. **Fit conformal bands** on the purged splitter, shaped by GARCH conditional
   volatility, and report empirical coverage.
6. **Refit** on all data and **persist** the model, calibrator, conformal
   artifacts, and metadata to `backend/prediction/models/`.

Run it:

```bash
python backend/prediction/train.py
```

Persisted artifacts (`backend/prediction/models/`, tracked in git so the API image ships them) include the XGBoost primary and
meta models, scaler, frac-diff parameters, isotonic calibrator, conformal
object, HMM, and `model_meta.json` (the recorded baseline metrics).

---

## 6. Phased Build and Gates (FLUX-X)

The agent is built incrementally; each phase ends with a **GATE** - a measurable
check on purged OOF. If a component fails its gate it stays self-gated OFF. The
gate scripts live in `scripts/experiments/` (`gate0_*` … `gate9_*`).

| Phase | Goal | Gate | Outcome |
|---|---|---|---|
| 0 | Baseline lock + data backfill | Baseline metrics reproduced and logged | Done: price-only baseline locked |
| 1 | Activate FRED macro features | OOF AUC ≥ baseline; ECE not worse | AUC passed (+0.005), but it halved the ranking Sharpe (0.91 → 0.46) → **off** |
| 2 | Crypto-native features (funding/OI/DVOL/on-chain) | Crypto-subset AUC ≥ baseline + 0.01 | Best +0.0055 → **off** |
| 3 | Asset-class specialisation (crypto vs equity experts) | Specialised ≥ pooled per class | Tie or worse → **pooled model kept** |
| 4 | Equity fundamentals + events | Equity-subset AUC ≥ Phase 3 | Tie (−0.0008) → **off** |
| 5 | Multi-source sentiment (CryptoBERT, GDELT) | Sentiment-augmented AUC ≥ prior | Too little history to test → **off as a feature**; used as a live tilt |
| 6 | Regime-conditional stacking ensemble | Stack AUC > best single base learner | 0.536 vs 0.537 → **off** |
| 7 | Cross-sectional portfolio | Net-of-cost Sharpe > baseline; drawdown ≤ current | Long-only top 10% + vol target: Sharpe 0.84 vs 0.83, max DD −46.5% vs −63.2% → pass (thin) |
| 8 | LLM verifier + flywheel hardening | Verifier never raises confidence; drift-retrain fires | Pass |
| 9 | Honesty baselines (continuous) | FLUX-X beats naive + foundation baselines on OOF | Not on raw accuracy (0.528 vs always-up 0.531); see §8 |

---

## 7. Inference Algorithm (per symbol, per day, using only data ≤ t)

```
1. FEATURES   x = the 42 served features                     (all causal)
2. REGIME     r = HMM.filter(market_obs ≤ t)  → {trend, chop, risk_off}  (shown, sizes)
3. PRIMARY    p  = XGBoost.predict_proba(x)
4. META       m  = MetaXGBoost.predict_proba([x, p])   # P(call is correct)
   (STACK     regime-conditional blend: loaded only if it beat the primary OOS; off today)
5. CALIBRATE  p_cal = Isotonic(p) ;  band = conformal(center, GARCH σ_h)
6. NEWS TILT  p_cal nudged by live headline sentiment (agree +, disagree −)
7. EDGE/GATE  edge = p_cal − 0.5 ;  act = (m ≥ 0.60)
              size = ¼-Kelly(p_cal) · regime scale · earnings gate
8. RANK       daily cycle ranks symbols by edge·m
9. VERIFY     LLM checks the top calls against fresh news → may veto/downgrade only
10. LOG       store → resolve at t+5 → refresh calibration buckets → weekly drift check
```

Serving is handled by `predict.py` / `serve.py`; the verifier by `agent.py`; the
flywheel (logging, resolution, drift-triggered retraining) by `serve.py` and
`drift.py`. `paper.py` can mirror signals to an Alpaca **paper** account when
`ALPACA_API_KEY` / `ALPACA_SECRET_KEY` are set; the deployment does not set them.

---

## 8. Evaluation Metrics

| Layer | Metric | Target |
|---|---|---|
| Direction (meta-labeled) | Accuracy / Precision@threshold / F1 | DA 56–63%; precision on taken trades > baseline |
| Magnitude regression | MAE / RMSE on % return, directional accuracy | DA > 55%; RMSE < persistence and < Chronos |
| Confidence calibration | ECE / Brier / reliability curve | ECE < 5%; predicted ≈ realised hit-rate |
| Conformal bands | Empirical coverage vs stated | 80% band covers 80% ± 3% out-of-sample |
| Strategy backtest | Sharpe/Sortino net of 5 bps, max DD, profit factor, per regime | Sharpe > baseline after costs |

The model is compared against persistence, ARIMA(1,0,0) and "always up" in
every training run. `baselines.py` also has a Chronos zero-shot baseline; it
needs PyTorch, so it runs only in a local research environment.

### Current results (retrain of 2026-10-04, data to 2026-10-03)

Out of fold, 133,753 predictions over 29 symbols (`models/model_meta.json`):

| Metric | Result | Target met? |
|---|---|---|
| Direction accuracy / AUC | 0.528 / 0.522 (always-up 0.531) | No: no edge on raw accuracy |
| Top 10% / 5% most confident | 0.551 / 0.560 | Small ranking edge |
| Meta "act" filter (thr 0.60) | 54.4% precision at 12.8% coverage (all calls 53.5%) | Small |
| ECE raw / calibrated (OOS) | 0.031 / 0.022 | Yes (< 5%) |
| Conformal 80% / 90% coverage | 80.0% / 90.0% | Yes |
| Point return MAE vs no change | 0.0541 vs 0.0535 | No: shown greyed out in the UI |
| Regime stack (GATE-6) | AUC 0.536 vs best base 0.537 | No: self-gated off |

The live counterpart is the Advisor's "Track record" line ("Right N% of M past
calls", from `/predict/{symbol}/history`); `/predict/calibration` returns the
realised hit rate per confidence bucket, and matured predictions record whether
the price landed inside the range (`in_band`).

---

## 9. The Self-Correcting Flywheel

1. Every prediction is logged to SQLite with its calibrated confidence, regime,
   conformal band, and feature snapshot.
2. After the horizon elapses, outcomes are **resolved** (which barrier was hit,
   direction correct, P&L net of costs) into `prediction_outcomes`.
3. `calibration_buckets` is refreshed from realised per-decile hit-rates, so the
   displayed confidence stays honest over time.
4. **Drift-triggered retraining** (`drift.py`) re-runs training when recent
   accuracy or ECE drifts past a threshold.
5. Optionally, signals can be forward-tested on an **Alpaca paper account**
   (keys not set in the deployment). There is no live-trading code path.

---

## 10. Pitfalls the Pipeline Explicitly Avoids

1. Look-ahead leakage - every feature's time alignment is audited; a
   shift-forward test must *drop* accuracy.
2. Overlapping-label leakage - purged + embargoed CV, never plain
   `TimeSeriesSplit`.
3. Random K-fold on time series - never; walk-forward only.
4. Predicting raw price - predict returns; frac-diff the inputs.
5. Uncalibrated confidence - isotonic calibration with ECE checks.
6. Adjustment/survivorship bias - adjusted close, cross-checked against Stooq.
7. Overfitting one regime - tested across drawdown periods; regime gate applied.
8. Ignoring costs - 5 bps round-trip baked into every backtest.
9. Full Kelly - ¼-Kelly off calibrated probabilities only.
10. Auto-trading too early - paper forward-test only; no live execution.

---

## 11. LLM Layer Boundary

The LLM does **not** predict price - the machine-learning models do. It explains
the numbers, cross-checks them against news and RAG context, and may lower
confidence or veto a trade when the narrative contradicts the model. It can never
raise confidence above the calibrated value. This separation keeps the system
both accurate and auditable.

---

## Appendix - Reference Reading

- López de Prado, *Advances in Financial Machine Learning* - triple-barrier,
  meta-labeling, purged CV, sample uniqueness.
- Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction* -
  guaranteed-coverage intervals.
- Oreshkin et al., *N-BEATS* - interpretable deep forecasting (magnitude head).
- Documentation for `xgboost`, `scikit-learn`, `hmmlearn`, `arch`, and
  `chronos-forecasting`.
