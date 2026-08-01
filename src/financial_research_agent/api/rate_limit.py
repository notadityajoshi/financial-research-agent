"""Redis-backed fixed-window rate limiting."""

import time
from typing import Any, Protocol

from fastapi import HTTPException, Request
from redis.asyncio import Redis

from financial_research_agent.config import get_settings
from financial_research_agent.logging_config import get_logger

log = get_logger(__name__)

_WINDOW_SECONDS = 60


class RedisLike(Protocol):
    """Minimal Redis surface the limiter needs."""

    def pipeline(self, transaction: bool = ...) -> Any: ...


class RateLimiter:
    """Fixed-window request counter per identity."""

    def __init__(self, redis: RedisLike, *, limit: int) -> None:
        self._redis = redis
        self._limit = limit

    async def allow(self, identity: str) -> bool:
        """True if this identity is within its per-minute budget.

        Fails open on Redis errors: a broken limiter must not take
        the API down.
        """
        window = int(time.time() // _WINDOW_SECONDS)
        key = f"ratelimit:{identity}:{window}"
        try:
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.incr(key)
                pipe.expire(key, _WINDOW_SECONDS * 2)
                count, _ = await pipe.execute()
        except Exception as exc:
            log.warning("rate_limit_unavailable", error=str(exc))
            return True
        allowed = int(count) <= self._limit
        if not allowed:
            log.warning("rate_limited", identity=identity, count=int(count))
        return allowed


def _identity(request: Request) -> str:
    """Prefer the API key; fall back to client IP."""
    key = request.headers.get("X-API-Key")
    if key:
        return f"key:{key}"
    host = request.client.host if request.client else "unknown"
    return f"ip:{host}"


async def enforce_rate_limit(request: Request) -> None:
    """FastAPI dependency: 429 when over budget."""
    limiter: RateLimiter | None = getattr(request.app.state, "limiter", None)
    if limiter is None:
        return  # not configured (tests without limiter)
    if not await limiter.allow(_identity(request)):
        raise HTTPException(status_code=429, detail="rate limit exceeded")


def build_limiter() -> RateLimiter:
    """Construct the production limiter from settings."""
    settings = get_settings()
    return RateLimiter(
        Redis.from_url(settings.redis_url),
        limit=settings.rate_limit_per_minute,
    )
