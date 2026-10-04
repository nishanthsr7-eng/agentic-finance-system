"""
FLUX Prediction — Magnitude Head (LSTM, Layer 2b, Phase 5)
==========================================================
The conformal band needs a POINT estimate of the h-day return to center on. The XGBoost return
regressor barely beats predicting zero (daily-return magnitude is near-unpredictable on free
data). This is a torch LSTM alternative that reads a *sequence* of the last L days of features
(so it can use path/shape, which a flat GBM row cannot) and predicts the h-day return.

It is trained and reported HONESTLY against the only baseline that matters here — predicting
zero (MAE_0). If the LSTM can't beat MAE_0 out-of-sample, the honest conclusion is that the
magnitude is noise and the band should center on 0 (or the direction-implied drift), not on a
fake point forecast. We report that verdict rather than hide it.

Leak-safety: sequences end at day t and the target is the FORWARD return t→t+h. Train/val is a
time-ordered split (early train, late val), and feature scaling is fit on TRAIN ONLY.

Public API:
    MagnitudeLSTM.load(dir).predict_seq(window) -> h-day return point estimate
    await fit_and_save(...)   # offline training + honest MAE report
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("flux.prediction.magnitude")

MODELS_DIR = Path(__file__).parent / "models"
SEQ_LEN = 30  # days of context fed to the LSTM
HIDDEN = 32
MAX_SEQUENCES = 60_000  # cap pooled sequences so CPU/GPU training stays bounded
EPOCHS = 8
BATCH = 512


def _torch():
    import torch

    return torch


class _LSTMReg:
    """Tiny 1-layer LSTM regressor (built lazily so torch isn't imported at module load)."""

    @staticmethod
    def build(n_features: int):
        _torch()
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self, n_in, hidden):
                super().__init__()
                self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
                self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        return Net(n_features, HIDDEN)


class MagnitudeLSTM:
    """Serving wrapper: holds the trained net, feature scaler, column order, and MAE report."""

    def __init__(self, state_dict, mu, sd, columns, horizon, report):
        self.mu = np.asarray(mu, dtype=float)
        self.sd = np.asarray(sd, dtype=float)
        self.columns = list(columns)
        self.horizon = int(horizon)
        self.report = report
        self._net = None
        self._state = state_dict

    def _ensure_net(self):
        if self._net is None:
            _torch()
            self._net = _LSTMReg.build(len(self.columns))
            self._net.load_state_dict(self._state)
            self._net.eval()

    def predict_seq(self, window: pd.DataFrame) -> float:
        """window: last >=SEQ_LEN rows with self.columns. Returns the h-day return point."""
        torch = _torch()
        self._ensure_net()
        x = window[self.columns].values[-SEQ_LEN:]
        if len(x) < SEQ_LEN:  # left-pad short windows
            x = np.vstack([np.repeat(x[:1], SEQ_LEN - len(x), axis=0), x])
        xs = (x - self.mu) / self.sd
        with torch.no_grad():
            t = torch.tensor(xs[None, :, :], dtype=torch.float32)
            return float(self._net(t).item())

    def save(self, d: str | Path) -> None:
        import joblib

        torch = _torch()
        d = Path(d)
        torch.save(self._state, d / "magnitude.pt")
        joblib.dump(
            {
                "mu": self.mu,
                "sd": self.sd,
                "columns": self.columns,
                "horizon": self.horizon,
                "report": self.report,
            },
            d / "magnitude_meta.pkl",
        )

    @classmethod
    def load(cls, d: str | Path) -> MagnitudeLSTM:
        import joblib

        torch = _torch()
        d = Path(d)
        meta = joblib.load(d / "magnitude_meta.pkl")
        state = torch.load(d / "magnitude.pt", map_location="cpu")
        return cls(state, meta["mu"], meta["sd"], meta["columns"], meta["horizon"], meta["report"])


# ── Sequence dataset assembly ────────────────────────────────────────────────────
async def _build_sequences(horizon: int):
    """Pool (window, forward-return) sequences across symbols. Returns Xseq, y, columns."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend.db import get_history, history_summary
    from backend.prediction.features import (
        build_features_from_df,
        calibrate_fd_order,
        feature_columns,
    )
    from backend.prediction.train import EXCLUDE, MIN_EVENTS

    syms = [
        r["symbol"]
        for r in await history_summary()
        if r["symbol"] not in EXCLUDE and r["rows"] >= MIN_EVENTS
    ]
    Xs, ys, columns = [], [], None
    for sym in syms:
        rows = await get_history(sym)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        early = df.iloc[: max(250, len(df) // 2)]
        d = calibrate_fd_order(pd.Series(early["adj_close"].values))
        feat = build_features_from_df(df, fd_order=d)
        if columns is None:
            columns = feature_columns(feat)
        F = feat[columns].values.astype(float)
        close = feat["close"].values.astype(float)
        n = len(feat)
        for i in range(SEQ_LEN - 1, n - horizon):
            Xs.append(F[i - SEQ_LEN + 1 : i + 1])
            ys.append(close[i + horizon] / close[i] - 1.0)
    Xs = np.asarray(Xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    # Cap (evenly subsample) to keep training bounded.
    if len(Xs) > MAX_SEQUENCES:
        idx = np.linspace(0, len(Xs) - 1, MAX_SEQUENCES).astype(int)
        Xs, ys = Xs[idx], ys[idx]
    return Xs, ys, columns


async def fit_and_save(horizon: int = 5) -> dict:
    """Train the LSTM magnitude head and persist it with an honest MAE-vs-zero report."""
    torch = _torch()
    import torch.nn as nn

    t0 = time.time()
    print(f"Building sequences (SEQ_LEN={SEQ_LEN}, horizon={horizon})...")
    Xs, ys, columns = await _build_sequences(horizon)
    print(f"  sequences: {len(Xs):,} x {SEQ_LEN} x {len(columns)} features")

    cut = int(len(Xs) * 0.8)  # time-ordered split
    mu = Xs[:cut].reshape(-1, Xs.shape[2]).mean(0)
    sd = Xs[:cut].reshape(-1, Xs.shape[2]).std(0) + 1e-8
    Xn = (Xs - mu) / sd

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = _LSTMReg.build(len(columns)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.SmoothL1Loss()
    Xtr = torch.tensor(Xn[:cut])
    ytr = torch.tensor(ys[:cut])
    Xva = torch.tensor(Xn[cut:]).to(dev)
    yva = ys[cut:]

    net.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(cut)
        tot = 0.0
        for b in range(0, cut, BATCH):
            j = perm[b : b + BATCH]
            xb = Xtr[j].to(dev)
            yb = ytr[j].to(dev)
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(j)
        print(f"  epoch {ep + 1}/{EPOCHS}  train_loss={tot / cut:.6f}")

    net.eval()
    with torch.no_grad():
        pred = net(Xva).cpu().numpy()
    mae = float(np.abs(pred - yva).mean())
    mae0 = float(np.abs(yva).mean())  # predict-zero baseline
    beats = mae < mae0
    report = {
        "n_val": int(len(yva)),
        "mae": mae,
        "mae_predict_zero": mae0,
        "beats_zero": bool(beats),
        "seq_len": SEQ_LEN,
        "horizon": horizon,
    }

    m = MagnitudeLSTM(net.cpu().state_dict(), mu, sd, columns, horizon, report)
    m.save(MODELS_DIR)
    print("\nMagnitude LSTM (time-split val):")
    print(f"  MAE          : {mae:.5f}")
    print(f"  MAE predict-0: {mae0:.5f}")
    print(
        f"  VERDICT      : {'beats zero (use as band center)' if beats else 'does NOT beat zero -> center band on 0/drift'}"
    )
    print(f"  saved: magnitude.pt + magnitude_meta.pkl  ({time.time() - t0:.1f}s on {dev})")
    return report


if __name__ == "__main__":
    import asyncio

    asyncio.run(fit_and_save())
