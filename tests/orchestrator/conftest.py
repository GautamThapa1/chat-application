import pytest

from backend.orchestrator.rate_limit import reset_limits


@pytest.fixture(autouse=True)
def clean_rate_limits():
    """Every test starts with empty rate limit counters."""
    reset_limits()