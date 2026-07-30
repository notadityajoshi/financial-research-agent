"""Offline telemetry tests: disabled by default, span helper works when on."""

import financial_research_agent.telemetry as telemetry
from financial_research_agent import config
from financial_research_agent.telemetry import get_tracer, setup_telemetry


def test_setup_is_noop_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("OTEL_ENABLED", "false")
    config.get_settings.cache_clear()
    telemetry._configured = False
    setup_telemetry("test-service")  # must not raise
    config.get_settings.cache_clear()


def test_tracer_span_usable_without_setup() -> None:
    """The no-op tracer supports the span API even when telemetry is off."""
    tracer = get_tracer("test")
    with tracer.start_as_current_span("unit") as span:
        span.set_attribute("k", "v")  # no-op span accepts attributes silently


def test_setup_enabled_console(monkeypatch) -> None:
    monkeypatch.setenv("OTEL_ENABLED", "true")
    monkeypatch.setenv("OTEL_EXPORTER", "console")
    config.get_settings.cache_clear()
    telemetry._configured = False
    setup_telemetry("test-service")
    assert telemetry._configured is True
    telemetry._configured = False  # reset for other tests
    config.get_settings.cache_clear()