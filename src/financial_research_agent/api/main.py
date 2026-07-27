"""FastAPI application factory."""

from pathlib import Path

from fastapi import FastAPI
from financial_research_agent.api.rate_limit import RateLimiter, build_limiter
from financial_research_agent.api.routes import router
from financial_research_agent.api.service import ResearchService
from financial_research_agent.logging_config import configure_logging


def _build_default_service() -> ResearchService:
    """Wire the real production dependency graph."""
    from qdrant_client import AsyncQdrantClient
    from typing import cast
    from financial_research_agent.api.service import GraphLike, ResearchService
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
    return ResearchService(cast(GraphLike, graph), Path(settings.reports_dir))




def create_app(
    service: ResearchService | None = None,
    limiter: RateLimiter | None = None,
    *,
    enable_rate_limit: bool = True,
) -> FastAPI:
    """Build the app; inject collaborators in tests, wire real ones otherwise."""
    configure_logging()
    app = FastAPI(title="Financial Research Agent", version="0.1.0")
    app.state.service = service if service is not None else _build_default_service()
    if limiter is not None:
        app.state.limiter = limiter
    elif enable_rate_limit and service is None:
        app.state.limiter = build_limiter()
    app.include_router(router)
    return app