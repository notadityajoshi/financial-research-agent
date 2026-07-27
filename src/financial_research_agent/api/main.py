"""FastAPI application factory."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from financial_research_agent.api.queue import ArqQueue, JobQueue, build_queue
from financial_research_agent.api.rate_limit import RateLimiter, build_limiter
from financial_research_agent.api.routes import router
from financial_research_agent.api.service import ResearchService
from financial_research_agent.logging_config import configure_logging


def create_app(
    service: ResearchService | None = None,
    limiter: RateLimiter | None = None,
    queue: JobQueue | None = None,
    *,
    enable_rate_limit: bool = True,
) -> FastAPI:
    """Build the app; inject collaborators in tests, wire real ones otherwise."""
    configure_logging()
    production = service is None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if production and getattr(app.state, "queue", None) is None:
            app.state.queue = await build_queue()
        yield
        q = getattr(app.state, "queue", None)
        if isinstance(q, ArqQueue):
            await q.close()

    app = FastAPI(title="Financial Research Agent", version="0.1.0", lifespan=lifespan)

    if production:
        from financial_research_agent.api.wiring import build_default_service

        app.state.service = build_default_service()
    else:
        app.state.service = service

    if queue is not None:
        app.state.queue = queue
    if limiter is not None:
        app.state.limiter = limiter
    elif enable_rate_limit and production:
        app.state.limiter = build_limiter()

    app.include_router(router)
    return app