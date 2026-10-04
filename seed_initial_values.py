import asyncio
import os
import sys
from datetime import datetime

import httpx
from dotenv import load_dotenv

# Add backend to path to import mappings
sys.path.insert(0, os.path.abspath("ai_engine/api"))
try:
    from market import CATEGORY_MAP
except ImportError:
    print("Error: Could not import CATEGORY_MAP from ai_engine.api.market")
    sys.exit(1)

load_dotenv()
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")

if not FINNHUB_KEY:
    print("Error: FINNHUB_API_KEY not found in environment.")
    sys.exit(1)


async def seed_slowly():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting slow, safe market sweep...")
    print(
        "This will fetch every value sequentially with a 1.2s delay to prevent 429 Rate Limits.\n"
    )

    total_assets = sum(len(assets) for assets in CATEGORY_MAP.values())
    count = 0

    async with httpx.AsyncClient() as client:
        for cat, assets in CATEGORY_MAP.items():
            print(f"--- Sweeping {cat.upper()} ({len(assets)} assets) ---")
            for asset in assets:
                symbol = asset["symbol"]
                count += 1
                try:
                    resp = await client.get(
                        f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}",
                        timeout=5.0,
                    )

                    if resp.status_code == 429:
                        print(
                            f"[{count:02d}/{total_assets}] {symbol:<20} [X] RATE LIMITED (429)! Backing off..."
                        )
                        await asyncio.sleep(10)
                        continue

                    data = resp.json()
                    price = data.get("c", 0)
                    change = data.get("dp", 0)

                    if price > 0:
                        print(
                            f"[{count:02d}/{total_assets}] {symbol:<20} [OK] {price:>10.4f} ({change:+.2f}%)"
                        )
                    else:
                        print(
                            f"[{count:02d}/{total_assets}] {symbol:<20} [WARN] NO DATA (Price is 0)"
                        )

                except Exception as e:
                    print(f"[{count:02d}/{total_assets}] {symbol:<20} [X] FAILED: {str(e)[:40]}")

                # CRITICAL: Sleep 1.2s to enforce max 50 requests per minute
                await asyncio.sleep(1.2)
            print()  # newline after category

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] [OK] INITIAL FULL MARKET CYCLE COMPLETED SUCCESSFULLY."
    )


if __name__ == "__main__":
    asyncio.run(seed_slowly())
