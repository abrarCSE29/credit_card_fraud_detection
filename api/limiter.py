from slowapi import Limiter
from slowapi.util import get_remote_address
from config import REDIS_URL, RATE_LIMIT_STRING

# Initialize limiter with Redis backend
# This limiter instance is shared across main.py and route decorators
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=REDIS_URL,
    default_limits=[RATE_LIMIT_STRING],
)
