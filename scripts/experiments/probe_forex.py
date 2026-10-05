import asyncio
import os

import httpx
from dotenv import load_dotenv

load_dotenv(".env")
key = os.getenv("FINNHUB_API_KEY")


async def test():
    symbols = ["EURUSD", "OANDA:EUR_USD", "FX:EURUSD", "EUR_USD"]
    async with httpx.AsyncClient() as client:
        for s in symbols:
            url = f"https://finnhub.io/api/v1/quote?symbol={s}&token={key}"
            r = await client.get(url)
            print(f"{s}: {r.status_code} - {r.text[:100]}")


if __name__ == "__main__":
    asyncio.run(test())
