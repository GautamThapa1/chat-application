import math
import time

from limits import parse
from limits.storage import MemoryStorage
from limits.strategies import FixedWindowRateLimiter, MovingWindowRateLimiter
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

moving_strategy = MovingWindowRateLimiter(storage)
MESSAGE_LIMIT = parse("10/minute")


def message_limit_retry_after(user_id: int) -> int | None:
    """None if this user may send a chat message, else seconds to wait."""
    key = str(user_id)
    if moving_strategy.hit(MESSAGE_LIMIT, "ws-message", key):
        return None
    stats = moving_strategy.get_window_stats(MESSAGE_LIMIT, "ws-message", key)
    return max(1, math.ceil(stats.reset_time - time.time()))

# Per account, so it holds even if the client address is forged. The cost: someone
# who keeps guessing can also lock the real owner out until the window passes.
LOGIN_LIMIT = parse("10/15 minutes")


def login_limit_retry_after(username: str) -> int | None:
    """Count one login attempt; None if allowed, else seconds to wait."""
    key = username.lower()  # usernames ignore case
    if moving_strategy.hit(LOGIN_LIMIT, "login-user", key):
        return None
    stats = moving_strategy.get_window_stats(LOGIN_LIMIT, "login-user", key)
    return max(1, math.ceil(stats.reset_time - time.time()))


def clear_login_limit(username: str) -> None:
    """A correct password starts the count again."""
    moving_strategy.clear(LOGIN_LIMIT, "login-user", username.lower())


def reset_limits() -> None:
    """Clear all counters. Used by tests."""
    limiter.reset()
    storage.reset()