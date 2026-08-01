"""arq worker: consumes research-run jobs from Redis."""

import uuid
from typing import ClassVar

from arq.connections import RedisSettings

from financial_research_agent.config import get_settings
from financial_research_agent.logging_config import configure_logging, get_logger
from financial_research_agent.telemetry import setup_telemetry

log = get_logger(__name__)

_service = None  # built lazily inside the worker process


def _get_service():
    """Lazily build the production service inside the worker process."""
    global _service
    if _service is None:
        from financial_research_agent.api.main import _build_default_service

        _service = _build_default_service()
    return _service


async def run_research(ctx: dict, run_id: str, ticker: str) -> None:
    """arq job: execute one research run by id."""
    log.info("job_started", run_id=run_id, ticker=ticker)
    service = ctx.get("service") or _get_service()
    await service.execute(uuid.UUID(run_id), ticker)
    log.info("job_finished", run_id=run_id)


async def _startup(ctx: dict) -> None:
    """Worker process startup hook."""
    configure_logging()
    setup_telemetry("research-worker")
    log.info("worker_started")


def _redis_settings() -> RedisSettings:
    """Redis connection settings from application config."""
    return RedisSettings.from_dsn(get_settings().redis_url)


class WorkerSettings:
    """arq worker configuration (referenced by the CLI)."""

    functions: ClassVar[list] = [run_research]
    on_startup = _startup
    redis_settings = _redis_settings()
