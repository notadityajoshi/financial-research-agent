"""Structured logging configuration for the whole application."""

import logging
import sys

import structlog

from financial_research_agent.config import get_settings


def configure_logging() -> None:
    """Configure structlog once at application startup.

    Dev/test: pretty, coloured console output for humans.
    Prod: JSON lines for log aggregation tools.
    """
    settings = get_settings()
    level: int = logging.getLevelNamesMapping()[settings.log_level.upper()]

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]

    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if settings.environment == "prod"
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[*processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    """Return a named structured logger."""
    return structlog.get_logger(name)
