"""
GATE-0 baseline reproduction.

Re-runs the LOCKED Phase-0 baseline: train.py (OOF AUC/ECE/precision) + backtest.py (Sharpe),
with FRED macro forced OFF so the result matches the price-only 42-feature baseline that FLUX-X
must beat. FRED activation is Phase 1, deliberately excluded here.

    python scripts/experiments/repro_baseline.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Force the Phase-0 (price-only) baseline regardless of whether FRED_API_KEY is in .env.
from backend.config import settings

settings.FRED_API_KEY = ""

from backend.prediction import backtest, train  # noqa: E402


async def _main():
    print("=" * 70)
    print("GATE-0 — reproducing the price-only baseline (FRED forced OFF)")
    print("=" * 70)
    await train.main()
    print("\n" + "=" * 70)
    print("GATE-0 — cost-aware walk-forward backtest")
    print("=" * 70)
    await backtest.run()


if __name__ == "__main__":
    asyncio.run(_main())
