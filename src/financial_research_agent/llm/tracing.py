"""Optional Langfuse tracing wrapper for LLM clients (decorator pattern)."""

from typing import Protocol

from financial_research_agent.config import get_settings
from financial_research_agent.llm.base import ChatMessage, LLMClient, LLMResponse
from financial_research_agent.logging_config import get_logger

log = get_logger(__name__)


class _Observation(Protocol):
    def update(self, **kwargs: object) -> None: ...
    def end(self) -> None: ...


class _Tracer(Protocol):
    def start_observation(self, **kwargs: object) -> _Observation: ...


class TracingLLMClient:
    """Wraps an LLMClient; records one Langfuse generation per call.

    Same interface as the wrapped client — callers are unaware.
    """

    def __init__(self, inner: LLMClient, langfuse: _Tracer) -> None:
        self._inner = inner
        self._langfuse = langfuse

    async def complete(
        self, messages: list[ChatMessage], *, temperature: float = 0.0
    ) -> LLMResponse:
        """Delegate to the inner client, wrapping the call in a trace span."""
        generation = self._langfuse.start_observation(
            as_type="generation",
            name="llm_completion",
            input=[{"role": m.role.value, "content": m.content} for m in messages],
            model_parameters={"temperature": temperature},
        )
        try:
            response = await self._inner.complete(messages, temperature=temperature)
            generation.update(
                output=response.content,
                model=response.model,
                usage_details={
                    "input": response.input_tokens,
                    "output": response.output_tokens,
                },
            )
            return response
        except Exception as exc:
            generation.update(level="ERROR", status_message=str(exc))
            raise
        finally:
            generation.end()


def maybe_wrap_with_tracing(client: LLMClient) -> LLMClient:
    """Return a tracing-wrapped client if Langfuse is configured, else the client.

    Any import/auth failure degrades to the untraced client — observability
    must never break the application.
    """
    settings = get_settings()
    if not settings.tracing_enabled:
        return client
    try:
        from langfuse import Langfuse

        langfuse = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        log.info("tracing_enabled", host=settings.langfuse_host)
        return TracingLLMClient(client, langfuse)  # type: ignore[arg-type]
    except Exception as exc:
        log.warning("tracing_setup_failed", error=str(exc))
        return client
