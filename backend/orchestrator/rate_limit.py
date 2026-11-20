import math
import time

from limits import parse
from limits.storage import MemoryStorage
from limits.strategies import FixedWindowRateLimiter

from slowapi import Limiter
from slowapi.util import get_remote_address  # identifies the client using the address

limiter = Limiter(
    key_func=get_remote_address,
)

storage = MemoryStorage()
strategy = FixedWindowRateLimiter(storage)
DEFAULT_LIMIT = parse("60/minute")


def default_limit_retry_after(client_id: str) -> int | None:
    """None if allowed, else seconds until the client may retry."""
    if strategy.hit(DEFAULT_LIMIT, "default", client_id):
        return None
    stats = strategy.get_window_stats(DEFAULT_LIMIT, "default", client_id)
    return max(1, math.ceil(stats.reset_time - time.time()))


def reset_limits() -> None:
    """Clear all counters. Used by tests."""
    limiter.reset()
    storage.reset()