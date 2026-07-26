"""Application configuration loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central, typed application settings.

    Values are read from environment variables, then the .env file.
    Invalid values raise an error at startup (fail fast).
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "financial-research-agent"
    environment: Literal["dev", "test", "prod"] = "dev"
    debug: bool = True
    log_level: str = "INFO"

    # LLM
    llm_provider: Literal["ollama", "openai"] = "ollama"
    llm_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://localhost:11434/v1"
    openai_api_key: str = ""
    # SEC EDGAR
    sec_user_agent: str = ""
    # Embeddings
    embedding_provider: Literal["ollama", "openai"] = "ollama"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768

    # Vector store
    qdrant_path: str = "data/qdrant"
@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (loaded once per process)."""
    return Settings()