"""API keys passed as query parameters never reach the logs."""

from __future__ import annotations

import logging

from backend import log_redact

URL = "https://finnhub.io/api/v1/calendar/earnings?from=2026-10-05&symbol=BTC&token=abc123SECRET"


def test_redact_masks_every_key_param():
    for name in ("token", "apiKey", "api_key", "apikey", "x_cg_demo_api_key"):
        out = log_redact.redact(f"GET https://x.test/a?q=1&{name}=SECRETVALUE&b=2 failed")
        assert "SECRETVALUE" not in out
        assert f"{name}=***" in out
        assert "q=1" in out and "b=2" in out


def test_redact_leaves_plain_text_alone():
    text = "Ingested 20 news articles; token count 5"
    assert log_redact.redact(text) == text


def test_installed_factory_masks_formatted_args(caplog):
    log_redact.install()
    log_redact.install()  # idempotent
    exc = RuntimeError(f"Client error '401 Unauthorized' for url '{URL}'")
    with caplog.at_level(logging.WARNING):
        logging.getLogger("flux.test").warning("earnings fetch for %s failed: %s", "BTC", exc)
    assert "abc123SECRET" not in caplog.text
    assert "token=***" in caplog.text
    assert "earnings fetch for BTC failed" in caplog.text
