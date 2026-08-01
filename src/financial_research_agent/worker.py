"""arq worker: consumes research-run jobs from Redis."""

import uuid

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from financial_research_agent.config import get_settings
from financial_research_agent.logging_config import configure_logging, get_logger
from financial_research_agent.telemetry import setup_telemetry

log = get_logger(__name__)

_service = None  # built lazily inside the worker process


def _get_service():
    global _service
    if _service is None:
        from financial_research_agent.api.main import _build_default_service

        _service = _build_default_service()
    return _service


async def run_research(ctx: dict, run_id: str, ticker: str) -> None:
    """arq job: execute one research run by id."""
    log.info("job_started", run_id=run_id, ticker=ticker)
    await _get_service().execute(uuid.UUID(run_id), ticker)
    log.info("job_finished", run_id=run_id)


async def _startup(ctx: dict) -> None:
    configure_logging()
    setup_telemetry("research-worker")
    log.info("worker_started")


def _redis_settings() -> RedisSettings:
    return RedisSettings.from_dsn(get_settings().redis_url)


class WorkerSettings:
    """arq worker configuration (referenced by the CLI)."""

    functions = [run_research]
    on_startup = _startup
    redis_settings = _redis_settings()


async def get_arq_pool() -> ArqRedis:
    """Connection pool for enqueuing jobs from the API process."""
    return await create_pool(_redis_settings())