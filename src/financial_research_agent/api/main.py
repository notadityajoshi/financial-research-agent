"""FastAPI application factory."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

from fastapi import FastAPI

from financial_research_agent.api.rate_limit import RateLimiter, build_limiter
from financial_research_agent.api.routes import router
from financial_research_agent.api.service import GraphLike, ResearchService
from financial_research_agent.logging_config import configure_logging, get_logger
from financial_research_agent.telemetry import instrument_fastapi, setup_telemetry

log = get_logger(__name__)


def _build_default_service() -> ResearchService:
    """Wire the real production dependency graph."""
    from financial_research_agent.agents.graph import build_research_graph
    from financial_research_agent.config import get_settings
    from financial_research_agent.integrations.financial_data import (
        FinancialDataClient,
    )
    from financial_research_agent.integrations.news import NewsClient
    from financial_research_agent.integrations.sec_edgar import SECEdgarClient
    from financial_research_agent.llm.embeddings import create_embedding_client
    from financial_research_agent.llm.factory import create_llm_client
    from financial_research_agent.retrieval.client import create_qdrant_client
    from financial_research_agent.retrieval.reranker import Reranker
    from financial_research_agent.retrieval.vector_store import VectorStore

    settings = get_settings()
    store = VectorStore(
        create_qdrant_client(),
        create_embedding_client(),
        dim=settings.embedding_dim,
    )
    graph = build_research_graph(
        SECEdgarClient(),
        FinancialDataClient(),
        NewsClient(),
        create_llm_client(),
        store,
        Reranker(),
    )
    return ResearchService(cast(GraphLike, graph), Path(settings.reports_dir))


def create_app(
    service: ResearchService | None = None,
    limiter: RateLimiter | None = None,
    *,
    enable_rate_limit: bool = True,
) -> FastAPI:
    """Build the app; inject collaborators in tests, wire real ones otherwise."""
    configure_logging()
    setup_telemetry("research-api")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is None:
            from financial_research_agent.api.queue import ArqJobQueue
            from financial_research_agent.config import get_settings

            app.state.queue = ArqJobQueue(get_settings().redis_url)
            log.info("arq_queue_attached")
        yield

    app = FastAPI(
        title="Financial Research Agent", version="0.1.0", lifespan=lifespan
    )
    instrument_fastapi(app)
    app.state.service = service if service is not None else _build_default_service()
    if limiter is not None:
        app.state.limiter = limiter
    elif enable_rate_limit and service is None:
        app.state.limiter = build_limiter()
    app.include_router(router)
    return app