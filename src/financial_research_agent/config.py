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
    qdrant_url: str = ""
    # Reranking
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Database
    database_url: str = "postgresql+asyncpg://localhost/finagent"
    # RAG
    max_index_chunks: int = 120
    # Reports
    reports_dir: str = "data/reports"
    # API auth (comma-separated keys; empty disables auth for local dev)
    api_keys: str = ""
    # Rate limiting
    redis_url: str = "redis://localhost:6379/0"
    rate_limit_per_minute: int = 5
    # Frontend
    api_base_url: str = "http://localhost:8000"
    # Observability (Langfuse — empty disables tracing)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    # Tracing (OpenTelemetry)
    otel_enabled: bool = False
    otel_exporter: Literal["console", "otlp"] = "console"
    otel_endpoint: str = "http://localhost:4317"
    @property
    def tracing_enabled(self) -> bool:
        """True only when both Langfuse keys are configured."""
        return bool(self.langfuse_public_key and self.langfuse_secret_key)
    @property
    def api_key_set(self) -> frozenset[str]:
        """Parsed set of valid API keys."""
        return frozenset(k.strip() for k in self.api_keys.split(",") if k.strip())
@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (loaded once per process)."""
    return Settings()