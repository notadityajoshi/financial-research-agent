"""Provider-agnostic embedding interface (Ollama dev, OpenAI prod)."""

from typing import Protocol

from openai import AsyncOpenAI

from financial_research_agent.config import get_settings
from financial_research_agent.logging_config import get_logger

log = get_logger(__name__)


class EmbeddingClient(Protocol):
    """Interface every embedding provider must satisfy."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in order."""
        ...


class OpenAICompatibleEmbeddings:
    """Embeddings via any OpenAI-compatible endpoint (Ollama, OpenAI)."""

    def __init__(self, model: str, api_key: str, base_url: str | None = None) -> None:
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts; order preserved."""
        response = await self._client.embeddings.create(model=self._model, input=texts)
        vectors = [item.embedding for item in response.data]
        log.info("embeddings_created", count=len(vectors), model=self._model)
        return vectors


def create_embedding_client() -> EmbeddingClient:
    """Return the embedding client selected by Settings."""
    settings = get_settings()
    if settings.embedding_provider == "ollama":
        return OpenAICompatibleEmbeddings(
            model=settings.embedding_model,
            api_key="ollama",
            base_url=settings.ollama_base_url,
        )
    if not settings.openai_api_key:
        msg = "OPENAI_API_KEY must be set when EMBEDDING_PROVIDER=openai"
        raise ValueError(msg)
    return OpenAICompatibleEmbeddings(
        model=settings.embedding_model, api_key=settings.openai_api_key
    )
