import os
import sys
from pathlib import Path

# Add the project root to path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))
# Add ai_engine to path
sys.path.insert(0, str(root / "ai_engine"))

try:
    from api.insights import run_bulk_insight_generation
    from config import CRYPTO_ASSETS, STOCK_ASSETS

    if __name__ == "__main__":
        symbols = [a["symbol"] for a in CRYPTO_ASSETS[:3]] + [a["symbol"] for a in STOCK_ASSETS[:2]]
        run_bulk_insight_generation(symbols)
except ImportError as e:
    print(f"Import failed: {e}")
    print(f"Path: {sys.path}")
