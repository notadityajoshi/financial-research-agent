"""Builds the configured LLM client from application settings."""

from financial_research_agent.config import get_settings
from financial_research_agent.llm.base import LLMClient
from financial_research_agent.llm.openai_compatible import OpenAICompatibleClient


def create_llm_client() -> LLMClient:
    """Return the LLM client selected by Settings (fail fast if misconfigured)."""
    settings = get_settings()

    if settings.llm_provider == "ollama":
        return OpenAICompatibleClient(
            model=settings.llm_model,
            api_key="ollama",  # Ollama ignores the key but the SDK requires one
            base_url=settings.ollama_base_url,
        )

    if not settings.openai_api_key:
        msg = "OPENAI_API_KEY must be set when LLM_PROVIDER=openai"
        raise ValueError(msg)
    return OpenAICompatibleClient(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
    )