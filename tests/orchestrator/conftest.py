import pytest

from backend.orchestrator.rate_limit import reset_limits
from backend.orchestrator.websocket import hub


@pytest.fixture(autouse=True)
def clean_rate_limits():
    """Every test starts with empty rate limit counters."""
    reset_limits()


@pytest.fixture(autouse=True)
def clean_generation_hub():
    """Generations are process-wide, so every test starts with an empty hub."""
    hub.reset()
    yield
    hub.reset()
