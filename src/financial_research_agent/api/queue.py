"""Job queue abstraction: arq in production, inline fallback for tests/dev."""

import asyncio
import uuid
from typing import Any, Protocol

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from financial_research_agent.logging_config import get_logger

log = get_logger(__name__)


class JobQueue(Protocol):
    """Anything that can schedule a research run for execution."""

    async def enqueue(self, run_id: uuid.UUID, ticker: str) -> None: ...


class ArqJobQueue:
    """Production queue: enqueues jobs into Redis for the arq worker."""

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._pool: ArqRedis | None = None
        self._lock = asyncio.Lock()

    async def _get_pool(self) -> ArqRedis:
        if self._pool is None:
            async with self._lock:
                if self._pool is None:
                    self._pool = await create_pool(
                        RedisSettings.from_dsn(self._redis_url)
                    )
        return self._pool

    async def enqueue(self, run_id: uuid.UUID, ticker: str) -> None:
        """Enqueue one run; _job_id makes re-enqueueing the same run a no-op."""
        pool = await self._get_pool()
        await pool.enqueue_job("run_research", str(run_id), ticker, _job_id=str(run_id))
        log.info("job_enqueued", run_id=str(run_id), ticker=ticker)


class InlineJobQueue:
    """Executes immediately in-process. Test/dev fallback only — blocks the request."""

    def __init__(self, service: Any) -> None:
        self._service = service

    async def enqueue(self, run_id: uuid.UUID, ticker: str) -> None:
        await self._service.execute(run_id, ticker)
