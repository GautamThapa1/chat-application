from slowapi import Limiter
from slowapi.util import get_remote_address  # identifies the client using the address

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["60 per minute"],
)