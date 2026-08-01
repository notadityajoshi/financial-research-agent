"""OpenTelemetry setup: distributed tracing across API and worker."""

from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SpanExporter,
)

from financial_research_agent.config import get_settings
from financial_research_agent.logging_config import get_logger

if TYPE_CHECKING:
    from fastapi import FastAPI
log = get_logger(__name__)

_configured = False


def _build_exporter() -> SpanExporter:
    """Pick the span exporter from settings (console by default)."""
    settings = get_settings()
    if settings.otel_exporter == "otlp":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        return OTLPSpanExporter(endpoint=settings.otel_endpoint, insecure=True)
    return ConsoleSpanExporter()


def setup_telemetry(service_name: str) -> None:
    """Configure the global tracer provider once per process.

    No-op when OTEL_ENABLED is false, so dev and tests stay silent.
    """
    global _configured
    settings = get_settings()
    if not settings.otel_enabled or _configured:
        return

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(_build_exporter()))
    trace.set_tracer_provider(provider)
    _configured = True
    log.info("otel_configured", service=service_name, exporter=settings.otel_exporter)


def instrument_fastapi(app: "FastAPI") -> None:
    """Auto-instrument a FastAPI app if telemetry is enabled."""
    if not get_settings().otel_enabled:
        return
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()


def get_tracer(name: str) -> trace.Tracer:
    """Return a tracer; safe no-op tracer when telemetry is disabled."""
    return trace.get_tracer(name)
