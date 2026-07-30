"""FastAPI application factory."""

from pathlib import Path

from fastapi import FastAPI
from financial_research_agent.telemetry import instrument_fastapi, setup_telemetry
from financial_research_agent.api.queue import ArqJobQueue, InlineJobQueue, JobQueue
from financial_research_agent.api.rate_limit import RateLimiter, build_limiter
from financial_research_agent.api.routes import router
from financial_research_agent.api.service import ResearchService
from financial_research_agent.logging_config import configure_logging


def build_default_service() -> ResearchService:
    """Wire the real production dependency graph. Used by the arq worker only —
    the API process never touches Qdrant (local-mode storage is single-process)."""
    from qdrant_client import AsyncQdrantClient

    from financial_research_agent.agents.graph import build_research_graph
    from financial_research_agent.config import get_settings
    from financial_research_agent.integrations.financial_data import (
        FinancialDataClient,
    )
    from financial_research_agent.integrations.news import NewsClient
    from financial_research_agent.integrations.sec_edgar import SECEdgarClient
    from financial_research_agent.llm.embeddings import create_embedding_client
    from financial_research_agent.llm.factory import create_llm_client
    from financial_research_agent.retrieval.reranker import Reranker
    from financial_research_agent.retrieval.vector_store import VectorStore

    settings = get_settings()
    store = VectorStore(
        AsyncQdrantClient(path=settings.qdrant_path),
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
    return ResearchService(graph, Path(settings.reports_dir))


def _build_api_service() -> ResearchService:
    """Lightweight service for the API process: DB reads/writes and report
    paths only. No graph, no Qdrant — the API never executes runs itself."""
    from financial_research_agent.config import get_settings

    class _NoGraph:
        async def ainvoke(self, state):  # pragma: no cover - never called
            msg = "API process cannot execute runs; use the arq worker"
            raise RuntimeError(msg)

    settings = get_settings()
    return ResearchService(_NoGraph(), Path(settings.reports_dir))


def create_app(
    service: ResearchService | None = None,
    limiter: RateLimiter | None = None,
    queue: JobQueue | None = None,
    *,
    enable_rate_limit: bool = True,
) -> FastAPI:
    """Build the app; inject collaborators in tests, wire real ones otherwise."""
    configure_logging()
    setup_telemetry("research-api")
    app = FastAPI(title="Financial Research Agent", version="0.1.0")
    instrument_fastapi(app)
    injected = service is not None
    app.state.service = service if injected else _build_api_service()

    if queue is not None:
        app.state.queue = queue
    elif injected:
        app.state.queue = InlineJobQueue(app.state.service)  # test/dev fallback
    else:
        from financial_research_agent.config import get_settings

        app.state.queue = ArqJobQueue(get_settings().redis_url)

    if limiter is not None:
        app.state.limiter = limiter
    elif enable_rate_limit and not injected:
        app.state.limiter = build_limiter()

    app.include_router(router)
    return app