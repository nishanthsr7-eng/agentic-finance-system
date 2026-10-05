"""Keep API keys out of the logs.

Finnhub, NewsAPI, FRED and Alpha Vantage take the key as a URL query
parameter, and an httpx error message quotes the full URL, so a plain
`log.warning("... failed: %s", exc)` writes the key into the Render logs.
`install()` swaps in a log-record factory that masks those values in every
record, whichever logger or handler it goes through.
"""

from __future__ import annotations

import logging
import re

_SECRET_PARAM = re.compile(
    r"(?i)\b(token|apikey|api_key|x_cg_demo_api_key|x_cg_pro_api_key)=[^&\s'\"<>]+"
)


def redact(text: str) -> str:
    """Replace the value of any key-bearing query parameter with ***."""
    return _SECRET_PARAM.sub(r"\1=***", text)


_installed = False


def install() -> None:
    """Mask secrets in every log record created from now on. Idempotent."""
    global _installed
    if _installed:
        return
    base = logging.getLogRecordFactory()

    def factory(*args, **kwargs):
        record = base(*args, **kwargs)
        try:
            msg = record.getMessage()
        except Exception:  # malformed args: leave the record to fail as usual
            return record
        clean = redact(msg)
        if clean != msg:
            record.msg, record.args = clean, None
        return record

    logging.setLogRecordFactory(factory)
    _installed = True
