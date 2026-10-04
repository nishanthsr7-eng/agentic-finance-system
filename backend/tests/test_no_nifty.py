"""NIFTY 50 is gone; the candle route stays and refuses unknown assets cleanly.

Uses TestClient without the lifespan, so no scheduler, DB or model loads.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend import ingestion, main

client = TestClient(main.app)


def test_old_nifty_candles_url_is_404_not_500():
    # A cached old Analysis page may still request this URL.
    r = client.get("/market/candles/nifty?tf=1D")
    assert r.status_code == 404


def test_candle_assets():
    assert set(main._CANDLE_SYMBOL_MAP) == {"btc", "eth", "spy"}


def test_ingestion_no_longer_fetches_nifty():
    assert "^NSEI" not in ingestion.OHLCV_SYMBOLS
    assert "^NSEI" not in ingestion.HISTORY_UNIVERSE


def test_holding_quote_symbols():
    assert main._holding_yf_symbol({"symbol": "BTC", "asset_type": "crypto"}) == "BTC-USD"
    assert (
        main._holding_yf_symbol({"symbol": "SPY", "asset_type": "etf", "currency": "USD"}) == "SPY"
    )
    assert (
        main._holding_yf_symbol({"symbol": "TCS", "asset_type": "equity", "currency": "INR"})
        == "TCS.NS"
    )


def test_usd_holdings_convert_through_fx():
    assert main._holding_needs_fx({"asset_type": "crypto", "currency": "INR"})
    assert main._holding_needs_fx({"asset_type": "etf", "currency": "USD"})
    assert not main._holding_needs_fx({"asset_type": "equity", "currency": "INR"})


def test_shipped_model_has_no_nifty():
    import json
    from pathlib import Path

    meta = json.loads(
        (Path(__file__).parents[1] / "prediction" / "models" / "model_meta.json").read_text()
    )
    assert not any("NIFTY" in s or "NSEI" in s for s in meta["fd_orders"])
