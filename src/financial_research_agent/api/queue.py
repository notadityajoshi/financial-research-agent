"""Job queue abstraction: API enqueues, worker executes."""

import uuid
from typing import Protocol

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from financial_research_agent.config import get_settings
from financial_research_agent.logging_config import get_logger

log = get_logger(__name__)


class JobQueue(Protocol):
    """Anything that can enqueue a research run."""

    async def enqueue(self, run_id: uuid.UUID, ticker: str) -> None: ...


class ArqQueue:
    """arq-backed queue over Redis."""

    def __init__(self, redis: ArqRedis) -> None:
        self._redis = redis

    async def enqueue(self, run_id: uuid.UUID, ticker: str) -> None:
        """Enqueue one research run for the worker."""
        await self._redis.enqueue_job("execute_research_run", str(run_id), ticker)
        log.info("job_enqueued", run_id=str(run_id), ticker=ticker)

    async def close(self) -> None:
        """Release the Redis connection pool."""
        await self._redis.aclose()


async def build_queue() -> ArqQueue:
    """Construct the production queue from settings."""
    settings = get_settings()
    return ArqQueue(await create_pool(RedisSettings.from_dsn(settings.redis_url)))