"""
Leakage & sanity tests for the feature/label pipeline.

Run directly:    python backend/prediction/test_features.py
Or with pytest:  pytest backend/prediction/test_features.py -q

The decisive test is `test_causality_truncation_invariance`: a feature at day t MUST NOT
change when future bars are added or removed. If it does, the feature peeks at the future.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.prediction.features import build_features_from_df, feature_columns
from backend.prediction.labeling import make_labels


async def _load_df(symbol: str = "AAPL") -> pd.DataFrame:
    from backend.db import get_history, init_db

    await init_db()
    return pd.DataFrame(await get_history(symbol))


# ── 1. No feature may correlate suspiciously with the FUTURE return ───────────
# NOTE: these `_check_*` helpers take the dataframe explicitly. The pytest entry points are the
# zero-arg `test_*` wrappers near the bottom (so pytest doesn't mistake `df` for a fixture).
def _check_no_target_leakage(df: pd.DataFrame) -> None:
    feat = build_features_from_df(df)
    cols = feature_columns(feat)
    ev = make_labels(feat["close"])
    y = ev["ret"].reindex(feat.index)
    corr = feat[cols].corrwith(y).abs()
    worst = corr.max()
    assert worst < 0.95, f"Suspicious feature↔future-return corr {worst:.3f} (possible leak)"
    print(f"  [1] no-target-leakage      OK   (max |corr| = {worst:.4f})")


# ── 2. Causality: features at day t are invariant to future data ──────────────
def _check_causality_truncation_invariance(df: pd.DataFrame) -> None:
    full = build_features_from_df(df)
    cut = int(len(df) * 0.7)
    truncated = build_features_from_df(df.iloc[:cut])  # hide everything after `cut`
    common = full.index.intersection(truncated.index)
    common = common[-200:]  # check the last 200 shared rows
    cols = feature_columns(full)
    a = full.loc[common, cols].to_numpy()
    b = truncated.loc[common, cols].to_numpy()
    max_diff = np.nanmax(np.abs(a - b))
    assert max_diff < 1e-9, (
        f"Feature values changed when future data removed (d={max_diff:.2e}) -> LEAK"
    )
    print(f"  [2] causality-invariance   OK   (max delta = {max_diff:.2e})")


# ── 3. No NaN / Inf leaks into the model matrix ───────────────────────────────
def _check_no_nan_inf(df: pd.DataFrame) -> None:
    feat = build_features_from_df(df)
    cols = feature_columns(feat)
    m = feat[cols].to_numpy()
    assert not np.isnan(m).any(), "NaN in feature matrix"
    assert not np.isinf(m).any(), "Inf in feature matrix"
    print(f"  [3] no-nan-inf             OK   ({feat.shape[0]} rows x {len(cols)} cols)")


# ── 4. Labels resolve in the future (t1 >= event date) ────────────────────────
def _check_label_t1_ordering(df: pd.DataFrame) -> None:
    feat = build_features_from_df(df)
    ev = make_labels(feat["close"])
    t0 = pd.to_datetime(ev.index)
    t1 = pd.to_datetime(ev["t1"])
    assert (t1 >= t0).all(), "A label resolved before its event date (impossible) → bug"
    assert ev["weight"].between(0, 1e6).all(), "Bad sample weights"
    print(f"  [4] label-t1-ordering      OK   ({len(ev)} events, all t1 >= t0)")


def _run() -> None:
    df = asyncio.run(_load_df("AAPL"))
    print("Running leakage & sanity tests on AAPL:")
    _check_no_target_leakage(df)
    _check_causality_truncation_invariance(df)
    _check_no_nan_inf(df)
    _check_label_t1_ordering(df)
    print("All tests passed.")


# pytest entry points (auto-load the dataframe once)
def _df():
    df = asyncio.run(_load_df("AAPL"))
    if df.empty:
        pytest.skip(
            "no seeded market data for AAPL (flux_market.db not populated in this environment)"
        )
    return df


def test_leakage():
    _check_no_target_leakage(_df())


def test_causality():
    _check_causality_truncation_invariance(_df())


def test_clean():
    _check_no_nan_inf(_df())


def test_labels():
    _check_label_t1_ordering(_df())


if __name__ == "__main__":
    _run()
