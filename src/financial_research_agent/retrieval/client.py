"""Qdrant client factory: server mode when a URL is set, else local file mode."""

from qdrant_client import AsyncQdrantClient

from financial_research_agent.config import get_settings


def create_qdrant_client() -> AsyncQdrantClient:
    """URL (server) takes precedence over path (embedded local mode)."""
    settings = get_settings()
    if settings.qdrant_url:
        return AsyncQdrantClient(url=settings.qdrant_url)
    return AsyncQdrantClient(path=settings.qdrant_path)