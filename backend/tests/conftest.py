import pytest

from backend import ratelimit


@pytest.fixture(autouse=True)
def _fresh_rate_limits():
    """Every test starts with an empty allowance."""
    ratelimit.reset()
    yield
    ratelimit.reset()
