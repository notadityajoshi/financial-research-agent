"""Production dependency wiring shared by the API and the worker."""

from pathlib import Path

from qdrant_client import AsyncQdrantClient

from financial_research_agent.agents.graph import build_research_graph
from financial_research_agent.api.service import ResearchService
from financial_research_agent.config import get_settings
from financial_research_agent.integrations.financial_data import FinancialDataClient
from financial_research_agent.integrations.news import NewsClient
from financial_research_agent.integrations.sec_edgar import SECEdgarClient
from financial_research_agent.llm.embeddings import create_embedding_client
from financial_research_agent.llm.factory import create_llm_client
from financial_research_agent.retrieval.reranker import Reranker
from financial_research_agent.retrieval.vector_store import VectorStore


def build_default_service() -> ResearchService:
    """Wire the real production dependency graph."""
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
