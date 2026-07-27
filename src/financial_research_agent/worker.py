"""arq worker: executes research runs consumed from the Redis queue.

Run with:
    uv run arq financial_research_agent.worker.WorkerSettings
"""

import uuid
from typing import Any

from arq.connections import RedisSettings

from financial_research_agent.api.main import build_default_service
from financial_research_agent.config import get_settings
from financial_research_agent.logging_config import configure_logging, get_logger

log = get_logger(__name__)


async def startup(ctx: dict[str, Any]) -> None:
    """Build the expensive dependency graph once per worker process."""
    configure_logging()
    ctx["service"] = build_default_service()
    log.info("worker_started")


async def run_research(ctx: dict[str, Any], run_id: str, ticker: str) -> None:
    """Execute one research run end to end."""
    await ctx["service"].execute(uuid.UUID(run_id), ticker)


class WorkerSettings:
    """arq worker configuration."""

    functions = [run_research]
    on_startup = startup
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_jobs = 2  # each job is LLM+embedding heavy; keep the Mac responsive
    job_timeout = 1800