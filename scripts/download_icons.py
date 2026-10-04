import asyncio
import os
import re
import sys
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.abspath("."))

# Dictionaries mapped out from market.py
from ai_engine.api.market import CATEGORY_MAP

ASSETS_DIR = "assets/market_icons"
MARKET_FILE = "ai_engine/api/market.py"

# Domain overrides for stocks/etfs where yfinance might fail or be slow
DOMAIN_MAP = {
    "NVDA": "nvidia.com",
    "AAPL": "apple.com",
    "MSFT": "microsoft.com",
    "GOOGL": "abc.xyz",
    "META": "meta.com",
    "TSLA": "tesla.com",
    "AMZN": "amazon.com",
    "JPM": "jpmorganchase.com",
    "UNH": "uhg.com",
    "XOM": "exxonmobil.com",
    "WMT": "walmart.com",
    "LLY": "lilly.com",
    "BABA": "alibaba.com",
    "TSM": "tsmc.com",
    "ASML": "asml.com",
    "V": "visa.com",
    "MA": "mastercard.com",
    "COST": "costco.com",
    "AMD": "amd.com",
    "INTC": "intel.com",
    "NFLX": "netflix.com",
    "DIS": "thewaltdisneycompany.com",
    "CRM": "salesforce.com",
    "PYPL": "paypal.com",
    "UBER": "uber.com",
    "SHOP": "shopify.com",
    "NOW": "servicenow.com",
    "PLTR": "palantir.com",
    "COIN": "coinbase.com",
    "RBLX": "roblox.com",
    "GLD": "ssga.com",
    "SLV": "ishares.com",
    "PPLT": "abrdn.com",
    "PALL": "abrdn.com",
    "USO": "uscfinvestments.com",
    "BNO": "uscfinvestments.com",
    "UNG": "uscfinvestments.com",
    "CPER": "uscfinvestments.com",
    "LIT": "globalxetfs.com",
    "WEAT": "teucrium.com",
    "CORN": "teucrium.com",
    "JO": "ipathetn.com",
    "SGG": "ipathetn.com",
    "NIB": "ipathetn.com",
    "SPY": "ssga.com",
    "QQQ": "invesco.com",
    "DIA": "ssga.com",
    "IWM": "ishares.com",
    "VIXY": "proshares.com",
    "INDY": "ishares.com",
    "INDA": "ishares.com",
    "EWG": "ishares.com",
    "EWU": "ishares.com",
    "EWJ": "ishares.com",
    "EWH": "ishares.com",
    "EWZ": "ishares.com",
}

# Forex to Country code mapping for flags
FOREX_MAP = {
    "EUR": "eu",
    "USD": "us",
    "GBP": "gb",
    "JPY": "jp",
    "CHF": "ch",
    "AUD": "au",
    "CAD": "ca",
    "NZD": "nz",
    "INR": "in",
    "MXN": "mx",
    "TRY": "tr",
}


async def fetch_logo(client, url, filepath):
    try:
        resp = await client.get(url, timeout=20.0, follow_redirects=True)
        if resp.status_code == 200:
            with open(filepath, "wb") as f:
                f.write(resp.content)
            return True
        else:
            print(f"Failed to fetch {url}: Status {resp.status_code}")
    except Exception as e:
        print(f"Failed to fetch {url}: Exception {type(e).__name__} - {e}")
    return False


async def get_crypto_icon(client, symbol, short):
    # Try github first
    url = f"https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/128/color/{short.lower()}.png"
    filepath = os.path.join(ASSETS_DIR, f"{short.lower()}.png")
    if await fetch_logo(client, url, filepath):
        return f"/assets/market_icons/{short.lower()}.png"

    # Fallback to predefined Google favicons for missing newer tokens
    fallback_domains = {
        "ARB": "arbitrum.io",
        "APT": "aptosfoundation.org",
        "RNDR": "rendernetwork.com",
        "FET": "fetch.ai",
        "NEAR": "near.org",
    }
    if short.upper() in fallback_domains:
        d = fallback_domains[short.upper()]
        f_url = f"https://www.google.com/s2/favicons?domain={d}&sz=128"
        if await fetch_logo(client, f_url, filepath):
            return f"/assets/market_icons/{short.lower()}.png"

    return None


async def get_stock_icon(client, symbol):
    domain = DOMAIN_MAP.get(symbol)
    filepath = os.path.join(ASSETS_DIR, f"{symbol.lower()}.png")

    if domain:
        url = f"https://www.google.com/s2/favicons?domain={domain}&sz=128"
        if await fetch_logo(client, url, filepath):
            return f"/assets/market_icons/{symbol.lower()}.png"

    # Fallback using yahoo finance logo url heuristic
    try:
        y_url = f"https://query2.finance.yahoo.com/v1/finance/quoteSummary/{symbol}?modules=summaryProfile"
        resp = await client.get(y_url, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            data = resp.json()
            website = (
                data.get("quoteSummary", {})
                .get("result", [{}])[0]
                .get("summaryProfile", {})
                .get("website")
            )
            if website:
                domain = urlparse(website).netloc.replace("www.", "")
                url = f"https://www.google.com/s2/favicons?domain={domain}&sz=128"
                if await fetch_logo(client, url, filepath):
                    return f"/assets/market_icons/{symbol.lower()}.png"
    except Exception:
        pass

    return None


async def get_forex_icon(client, symbol):
    # e.g., OANDA:EUR_USD -> EUR and USD
    parts = symbol.replace("OANDA:", "").split("_")
    if len(parts) == 2:
        base = parts[0]
        cc = FOREX_MAP.get(base, "us")
        url = f"https://flagcdn.com/w80/{cc}.png"
        filepath = os.path.join(ASSETS_DIR, f"{base.lower()}.png")
        if await fetch_logo(client, url, filepath):
            return f"/assets/market_icons/{base.lower()}.png"
    return None


async def rewrite_market_py(asset_icons_map):
    with open(MARKET_FILE) as f:
        content = f.read()

    # Simple regex block replacement to add the icon field without breaking the syntax
    # Since market.py is nicely formatted, we can replace `"symbol": "..."` with `"symbol": "...", "icon": "..."`

    for symbol, icon_path in asset_icons_map.items():
        if not icon_path:
            continue

        # Looking for {"symbol": "SYMBOL"
        # and replacing it with {"symbol": "SYMBOL", "icon": "ICON_PATH"
        # Be careful if icon is already there
        pattern = r'(\{"symbol":\s*"' + re.escape(symbol) + r'")((?!,\s*"icon":).)*?'

        # To avoid complex regex, let's just do standard string manipulation on lines
        new_lines = []
        for line in content.split("\n"):
            if f'"symbol": "{symbol}"' in line:
                if '"icon":' not in line:
                    # insert icon after symbol
                    line = line.replace(
                        f'"symbol": "{symbol}"', f'"symbol": "{symbol}", "icon": "{icon_path}"'
                    )
            new_lines.append(line)
        content = "\n".join(new_lines)

    with open(MARKET_FILE, "w") as f:
        f.write(content)


async def main():
    if not os.path.exists(ASSETS_DIR):
        os.makedirs(ASSETS_DIR)

    asset_icons_map = {}

    async with httpx.AsyncClient() as client:
        # Crypto
        print("Fetching Crypto...")
        for cp in CATEGORY_MAP["crypto"]:
            icon = await get_crypto_icon(
                client,
                cp["symbol"],
                cp.get("short", cp["symbol"].split(":")[1].replace("USDT", "")),
            )
            asset_icons_map[cp["symbol"]] = icon

        print("Fetching Stocks...")
        for sp in CATEGORY_MAP["stocks"]:
            icon = await get_stock_icon(client, sp["symbol"])
            asset_icons_map[sp["symbol"]] = icon

        print("Fetching Forex...")
        for fp in CATEGORY_MAP["forex"]:
            icon = await get_forex_icon(client, fp["symbol"])
            asset_icons_map[fp["symbol"]] = icon

        print("Fetching Indices...")
        for ip in CATEGORY_MAP["indices"]:
            icon = await get_stock_icon(client, ip["symbol"])
            asset_icons_map[ip["symbol"]] = icon

    # Check un-fetched logos and provide placeholders
    print(f"Total processed: {len(asset_icons_map)}")

    missing = [k for k, v in asset_icons_map.items() if not v]
    if missing:
        print(f"Failed to fetch {len(missing)} icons: {missing}")

    await rewrite_market_py(asset_icons_map)
    print("market.py rewritten.")


if __name__ == "__main__":
    asyncio.run(main())
