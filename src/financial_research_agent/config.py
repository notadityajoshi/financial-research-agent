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


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (loaded once per process)."""
    return Settings()