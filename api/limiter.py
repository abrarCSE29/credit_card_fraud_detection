import os
from slowapi import Limiter
from slowapi.util import get_remote_address
from config import REDIS_URL, RATE_LIMIT_STRING

# Determine storage backend based on environment
# In test environment, use in-memory storage if Redis is not available
is_test = "pytest" in " ".join(os.sys.argv)
storage_uri = REDIS_URL

if is_test:
    # For tests, try Redis first, fall back to memory if not available
    import redis

    try:
        test_redis = redis.from_url(REDIS_URL)
        test_redis.ping()
        test_redis.close()
        storage_uri = REDIS_URL
    except redis.ConnectionError:
        # Use in-memory storage for tests when Redis is not available
        storage_uri = "memory://"

# Initialize limiter with appropriate backend
# This limiter instance is shared across main.py and route decorators
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=storage_uri,
    default_limits=[RATE_LIMIT_STRING] if storage_uri != "memory://" else [],
)
