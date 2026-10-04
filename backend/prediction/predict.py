"""
FLUX Prediction — Inference (Layer 2a serving)
==============================================
Loads the trained pooled model + isotonic calibrator + metadata, builds features for a
symbol's latest history (using that symbol's persisted frac-diff order), and returns a
single calibrated direction prediction.

    pred = await predict("AAPL")
    # {'symbol','direction','prob_up','confidence','horizon_days','target_date', ...}

The confidence is the CALIBRATED probability mapped to 0-100, so "62" really means ~62%.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("flux.prediction.predict")

MODELS_DIR = Path(__file__).parent / "models"

_model = None
_meta_model = None
_calibrator = None
_return_model = None
_bands = None  # ConformalBands | None (None until a model is trained with it)
_stack = None  # RegimeStacker | None (self-gated: only set if it beats best base OOS)
_base2 = None  # dict{model,feature_columns} | None — 2nd base learner for the stack
_magnitude = None  # MagnitudeLSTM | None (self-gated: only set if it beats predict-zero)
_meta: dict | None = None
_macro: pd.DataFrame | None = None

# meta-model P(correct) above which we recommend acting. Set to the high-selectivity operating
# point: OOF meta report shows thr 0.60 → precision 0.559 at 18% coverage (vs 0.55 → 0.550 at 44%,
# 0.50 → 0.539 at 75%). We deliberately trade fewer, higher-precision signals — the edge here is
# selectivity, not coverage. Kelly sizing (sizing.kelly_fraction) still scales by meta_prob.
ACT_THRESHOLD = 0.60
BAND_ALPHA = 0.2  # default conformal miscoverage → 80% interval
WIDE_ALPHA = 0.1  # second, wider range served alongside it → 90% interval
DRIFT_K = 0.5  # how far the risk band leans toward the called side per σ of edge;
# small by design so the symmetric conformal coverage stays valid


def _load():
    global \
        _model, \
        _meta_model, \
        _calibrator, \
        _return_model, \
        _bands, \
        _meta, \
        _stack, \
        _base2, \
        _magnitude
    if _model is not None:
        return
    import joblib
    from xgboost import XGBClassifier, XGBRegressor

    _meta = json.loads((MODELS_DIR / "model_meta.json").read_text())
    _model = XGBClassifier()
    _model.load_model(MODELS_DIR / "xgb_primary.json")
    _meta_model = XGBClassifier()
    _meta_model.load_model(MODELS_DIR / "xgb_meta.json")
    _calibrator = joblib.load(MODELS_DIR / "calibrator.pkl")
    # Layer-2b magnitude head + conformal bands (optional: only if train.py produced them).
    reg_path, band_path = MODELS_DIR / "xgb_return.json", MODELS_DIR / "conformal.pkl"
    if reg_path.exists() and band_path.exists():
        from .conformal import ConformalBands

        _return_model = XGBRegressor()
        _return_model.load_model(reg_path)
        _bands = ConformalBands.load(band_path)
        log.info("Loaded conformal bands (n_calib=%d, alphas=%s)", _bands.n_calib, list(_bands.q))

    # Regime-conditional stacking ensemble (Phase 6) — loaded but SELF-GATED: only used at serving
    # if its persisted OOS report shows the stack beats the BEST single base learner (GATE-6).
    # Needs its companion base-2 learner (base2.pkl) to produce the p2_base input live; if either is
    # missing or the gate didn't clear, the stack stays OFF and serving falls back to the primary.
    _stack = None
    _base2 = None
    stack_path, base2_path = MODELS_DIR / "regime_stack.pkl", MODELS_DIR / "base2.pkl"
    if stack_path.exists() and base2_path.exists():
        try:
            from .ensemble import GATE_MARGIN, RegimeStacker

            s = RegimeStacker.load(stack_path)
            rep = s.report or {}
            gate_ok = (
                rep.get("gate6_pass")
                or rep.get("stack_auc", 0.0) > rep.get("best_base_auc", 1.0) + GATE_MARGIN
            )
            if gate_ok:
                _base2 = joblib.load(base2_path)
                _stack = s
                log.info(
                    "Regime stack ENABLED (AUC %.4f > best base %.4f)",
                    rep.get("stack_auc", float("nan")),
                    rep.get("best_base_auc", float("nan")),
                )
            else:
                log.info("Regime stack present but gated OFF (no OOS lift over best base learner)")
        except Exception as exc:
            log.warning("regime stack load failed: %s", exc)

    # LSTM magnitude head (Phase 5) — loaded but SELF-GATED: only used to center the band if it
    # beat predict-zero out-of-sample. It currently doesn't, so the XGB return point is kept.
    _magnitude = None
    if (MODELS_DIR / "magnitude.pt").exists():
        try:
            from .magnitude import MagnitudeLSTM

            m = MagnitudeLSTM.load(MODELS_DIR)
            if (m.report or {}).get("beats_zero"):
                _magnitude = m
                log.info("LSTM magnitude head ENABLED (beats predict-zero)")
            else:
                log.info("LSTM magnitude head present but gated OFF (does not beat zero)")
        except Exception as exc:
            log.warning("magnitude load failed: %s", exc)

    log.info("Loaded primary+meta models (%d features)", len(_meta["feature_columns"]))


async def _macro_frame() -> pd.DataFrame:
    global _macro
    if _macro is None:
        from .train import build_macro

        _macro = await build_macro()
    return _macro


async def predict(
    symbol: str,
    with_sentiment: bool = True,
    with_regime: bool = True,
    with_earnings: bool = True,
    with_options: bool = True,
) -> dict | None:
    """Produce one calibrated direction prediction for `symbol`."""
    _load()
    from ..db import get_history
    from .features import build_features_from_df

    symbol = symbol.upper()
    rows = await get_history(symbol)
    if not rows:
        log.warning("No history for %s", symbol)
        return None

    d = _meta["fd_orders"].get(symbol, 0.4)  # symbol's calibrated frac-diff order
    df = pd.DataFrame(rows)
    feat = build_features_from_df(df, fd_order=d)

    macro = await _macro_frame()
    feat.index = pd.to_datetime(feat.index)  # align string dates with Timestamp macro index
    feat = feat.join(macro.reindex(feat.index, method="ffill"))
    feat["rs_1"] = feat["logret_1"] - feat["spx_ret1"]
    feat["rs_5"] = feat["logret_5"] - feat["spx_ret5"]
    feat = feat.dropna()
    if feat.empty:
        return None

    cols = _meta["feature_columns"]
    x = feat[cols].iloc[[-1]]  # most recent fully-formed row
    as_of = feat.index[-1]
    last_close = float(feat["close"].iloc[-1])

    raw_p = float(_model.predict_proba(x)[:, 1][0])
    prob_up = float(_calibrator.predict([raw_p])[0])

    # Meta-model: P(the primary direction call is correct) → confidence + act gate.
    x_meta = x.copy()
    x_meta["primary_p"] = prob_up
    meta_prob = float(_meta_model.predict_proba(x_meta[_meta["meta_columns"]])[:, 1][0])

    # ── Regime-conditional stacking blend (Phase 6) — self-gated: only runs if the stack beat the
    #    best single base learner out-of-sample (GATE-6). Mixes the XGB primary with a decorrelated
    #    ElasticNet base learner, the meta conviction and the magnitude point, with per-regime
    #    expert weights softly mixed by the live HMM posterior. ──
    if _stack is not None and _base2 is not None:
        try:
            p2 = float(_base2["model"].predict_proba(x)[:, 1][0])
            mag_pt = float(_return_model.predict(x)[0]) if _return_model is not None else 0.0
            regime_probs = {"trend": 1.0, "chop": 0.0, "risk_off": 0.0}
            if with_regime:
                from .regime import current_regime

                regime_probs = (await current_regime())["probs"]
            prob_up = _stack.predict_proba(
                {"primary_cal": prob_up, "p2_base": p2, "meta_prob": meta_prob, "mag_oof": mag_pt},
                regime_probs,
            )
        except Exception as exc:
            log.warning("stack blend failed for %s: %s", symbol, exc)

    direction = "UP" if prob_up >= 0.5 else "DOWN"
    base_conf = meta_prob * 100  # P(correct) → 0..100 confidence
    horizon = int(_meta["horizon"])
    target_date = (pd.Timestamp(as_of) + pd.tseries.offsets.BDay(horizon)).date()

    # ── Layer-2b: risk band (guaranteed coverage) around an HONEST center ───────────
    # The h-day return MAGNITUDE is genuinely noise here: neither the XGB return regressor nor the
    # LSTM head beats predict-zero out-of-sample. So we do NOT dress up that ~0 point as a forecast.
    # Instead:
    #   • if the LSTM magnitude head ever passes its self-gate (beats zero OOS), use it as the center;
    #   • otherwise center the band on a transparent DIRECTION-IMPLIED DRIFT — it leans the way of the
    #     called side, scaled by the calibrated edge (2·prob_up−1) and the GARCH vol. It is small by
    #     construction (≪ the band half-width), so the conformal coverage — calibrated as a symmetric
    #     band with ŷ≈0 — is preserved. This is a pure risk band, presented honestly as one.
    pred_return = pred_price = conf_low = conf_high = None
    conf_low_90 = conf_high_90 = None
    band_cov = None
    band_kind = None
    if _bands is not None:
        from .garch import forecast_h_vol

        scale = forecast_h_vol(feat["close"], horizon)  # GARCH h-day σ (causal, today-anchored)
        if _magnitude is not None:
            try:
                pred_return = float(_magnitude.predict_seq(feat[_magnitude.columns]))
                band_kind = "magnitude_head"
            except Exception as exc:
                log.warning("magnitude predict failed for %s: %s", symbol, exc)
        if pred_return is None:  # honest default: direction-implied drift
            pred_return = (2.0 * prob_up - 1.0) * float(scale) * DRIFT_K
            band_kind = "risk_band"  # center is a lean, not a point forecast
        ret_lo, ret_hi = _bands.interval(pred_return, scale, alpha=BAND_ALPHA)
        pred_price = round(last_close * (1 + pred_return), 4)
        conf_low = round(last_close * (1 + ret_lo), 4)
        conf_high = round(last_close * (1 + ret_hi), 4)
        if WIDE_ALPHA in _bands.q:  # older conformal.pkl may lack it
            w_lo, w_hi = _bands.interval(pred_return, scale, alpha=WIDE_ALPHA)
            conf_low_90 = round(last_close * (1 + w_lo), 4)
            conf_high_90 = round(last_close * (1 + w_hi), 4)
        band_cov = round(_bands.coverage.get(BAND_ALPHA, float("nan")), 4)
        pred_return = round(pred_return, 6)

    # ── Layer-2c fusion: tilt confidence by live news sentiment ────────────────
    sent = {"score": 0.0, "label": "neutral", "n": 0}
    if with_sentiment:
        try:
            from .sentiment import symbol_sentiment

            sent = await symbol_sentiment(symbol)
        except Exception as exc:
            log.warning("sentiment unavailable for %s: %s", symbol, exc)

    # Agreement: +1 if sentiment sign matches model direction, -1 if it opposes.
    dir_sign = 1 if direction == "UP" else -1
    agree = dir_sign * sent["score"]  # >0 agree, <0 disagree
    # Scale confidence: strong agreement up to +20%, strong disagreement down to -40%.
    factor = 1.0 + (0.20 * agree if agree >= 0 else 0.40 * agree)
    factor = max(0.5, min(1.2, factor))
    confidence = int(round(max(0.0, min(100.0, base_conf * factor))))

    # ── Regime gate: market state scales the position size ─────────────────────
    regime, regime_probs = "trend", {}
    if with_regime:
        try:
            from .regime import REGIME_SCALE, current_regime
            from .sizing import kelly_fraction

            rg = await current_regime()
            regime, regime_probs = rg["regime"], rg["probs"]
            kelly = kelly_fraction(meta_prob) * REGIME_SCALE.get(regime, 1.0)
        except Exception as exc:
            log.warning("regime unavailable for %s: %s", symbol, exc)
            from .sizing import kelly_fraction

            kelly = kelly_fraction(meta_prob)
    else:
        from .sizing import kelly_fraction

        kelly = kelly_fraction(meta_prob)

    result = {
        "symbol": symbol,
        "model": "xgb_primary+meta+sentiment",
        "direction": direction,
        "prob_up": round(prob_up, 4),
        "meta_prob": round(meta_prob, 4),
        "act": bool(meta_prob >= ACT_THRESHOLD),
        "confidence": confidence,
        "base_confidence": int(round(base_conf)),
        "pred_return": round(pred_return, 4) if pred_return is not None else None,
        "pred_price": pred_price,
        "conf_low": conf_low,
        "conf_high": conf_high,
        "conf_low_90": conf_low_90,
        "conf_high_90": conf_high_90,
        "band_pct": int(round((1 - BAND_ALPHA) * 100)),
        "band_coverage": band_cov,
        "band_kind": band_kind,  # 'magnitude_head' | 'risk_band' | None
        "sentiment": sent["score"],
        "sentiment_label": sent["label"],
        "sentiment_n": sent["n"],
        "regime": regime,
        "iv_atm": None,
        "iv_skew": None,
        "iv_term": None,
        "iv_as_of": None,
        "kelly_frac": round(kelly, 4),
        "horizon_days": horizon,
        "as_of": str(pd.Timestamp(as_of).date()),
        "target_date": str(target_date),
        "last_close": last_close,
        "fd_order": d,
        "earnings_soon": False,
        "days_to_earnings": None,
    }

    # ── Earnings-window gate (Phase 3.1): halve confidence/size if earnings within the horizon ──
    if with_earnings:
        try:
            from .earnings import apply_gate, earnings_gate

            gate = await earnings_gate(symbol, horizon=horizon, as_of=result["as_of"])
            result = apply_gate(result, gate)
        except Exception as exc:
            log.warning("earnings gate unavailable for %s: %s", symbol, exc)

    # ── Options IV/skew (orthogonal context) ───────────────────────────────────
    # Latest snapshot from the options_iv flywheel. Equity-only; crypto/no-data → stays None.
    # Surfaced for auditing / the LLM layer and logged with the prediction, but INERT w.r.t. the
    # confidence number until it has validated live history (the self-gating contract).
    if with_options:
        try:
            from .options import symbol_iv

            iv = await symbol_iv(symbol)
            if iv["available"]:
                result.update(
                    {
                        "iv_atm": iv["atm_iv"],
                        "iv_skew": iv["skew"],
                        "iv_term": iv["term_slope"],
                        "iv_as_of": iv["as_of"],
                    }
                )
        except Exception as exc:
            log.warning("options IV unavailable for %s: %s", symbol, exc)

    return result


async def predict_all(symbols: list[str] | None = None) -> list[dict]:
    """Predict for many symbols, ranked by confidence (leaderboard-ready)."""
    _load()
    if symbols is None:
        from ..db import history_summary
        from .train import EXCLUDE

        symbols = [r["symbol"] for r in await history_summary() if r["symbol"] not in EXCLUDE]
    out = []
    for s in symbols:
        try:
            p = await predict(s)
            if p:
                out.append(p)
        except Exception as exc:
            log.warning("predict %s failed: %s", s, exc)
    return sorted(out, key=lambda p: p["confidence"], reverse=True)


if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path as _P

    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))

    async def _demo():
        for sym in ("AAPL", "NVDA", "BTC", "TSLA"):
            print(await predict(sym))

    asyncio.run(_demo())
