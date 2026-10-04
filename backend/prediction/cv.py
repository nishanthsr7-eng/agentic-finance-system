"""
FLUX Prediction — Purged & Embargoed Walk-Forward Cross-Validation
==================================================================
Triple-barrier labels overlap in time: the label at day t looks forward up to `horizon`
days, so it shares bars with the labels at t+1, t+2, ...  Ordinary CV (even sklearn's
TimeSeriesSplit) then leaks — a test label shares bars with training labels — which inflates
out-of-sample accuracy by 10–20 points and produces models that die in production.

This splitter fixes that (López de Prado, *Advances in Financial ML*, ch. 7):
  • WALK-FORWARD : train only on the PAST, test on the next contiguous block (expanding window).
  • PURGE        : drop any training label whose window [t0, t1] overlaps the test window.
  • EMBARGO      : drop a small buffer of training labels right before the test block, to stop
                   serial-correlation bleed across the boundary.

Usage:
    cv = PurgedWalkForwardSplit(n_splits=6, embargo=10)
    for train_idx, test_idx in cv.split(X, t1):     # t1 = label resolve dates, aligned to X
        ...
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PurgedWalkForwardSplit:
    def __init__(self, n_splits: int = 6, embargo: int = 10):
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, X: pd.DataFrame, t1: pd.Series):
        """
        Yield (train_pos, test_pos) integer-position arrays.
        X   : feature frame indexed by event start date (sorted ascending).
        t1  : Series (same index/order as X) of each label's resolve date.
        """
        if not X.index.is_monotonic_increasing:
            order = np.argsort(X.index.values)
            X = X.iloc[order]
            t1 = t1.iloc[order]

        times = pd.to_datetime(X.index.values)
        t1_arr = pd.to_datetime(t1.values)
        n = len(X)
        positions = np.arange(n)
        test_size = n // (self.n_splits + 1)
        if test_size == 0:
            raise ValueError("Not enough samples for the requested n_splits.")

        for i in range(self.n_splits):
            test_start = (i + 1) * test_size
            test_end = n if i == self.n_splits - 1 else test_start + test_size
            test_pos = positions[test_start:test_end]
            if len(test_pos) == 0:
                continue

            emb_pos = max(0, test_start - self.embargo)
            emb_date = times[emb_pos]

            # Train = events that START before the test block AND whose label RESOLVES
            # before the embargo cutoff (so no label window reaches into the test window).
            train_mask = (positions < test_start) & (t1_arr < emb_date)
            train_pos = positions[train_mask]
            if len(train_pos) == 0:
                continue
            yield train_pos, test_pos

    def get_n_splits(self, *_args) -> int:
        return self.n_splits


def overlap_count(X: pd.DataFrame, t1: pd.Series, train_pos, test_pos) -> int:
    """
    Diagnostic: number of training labels whose window [t0, t1] overlaps the test window.
    A correct purged split must return 0.
    """
    times = pd.to_datetime(X.index.values)
    t1_arr = pd.to_datetime(t1.values)
    test_lo = times[test_pos].min()
    test_hi = times[test_pos].max()
    tr_t0 = times[train_pos]
    tr_t1 = t1_arr[train_pos]
    # overlap if train label's [t0, t1] intersects [test_lo, test_hi]
    overlap = (tr_t0 <= test_hi) & (tr_t1 >= test_lo)
    return int(overlap.sum())
