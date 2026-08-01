"""Offline tracing tests: fake Langfuse, no network."""

from financial_research_agent.llm.base import ChatMessage, LLMResponse, Role
from financial_research_agent.llm.tracing import TracingLLMClient


class FakeGeneration:
    def __init__(self) -> None:
        self.updated: dict = {}
        self.ended = False

    def update(self, **kwargs) -> None:
        self.updated.update(kwargs)

    def end(self) -> None:
        self.ended = True


class FakeLangfuse:
    def __init__(self) -> None:
        self.generation = FakeGeneration()
        self.started_with: dict = {}

    def start_observation(self, **kwargs):
        self.started_with = kwargs
        return self.generation


class FakeLLM:
    async def complete(self, messages, *, temperature: float = 0.0) -> LLMResponse:
        return LLMResponse(content="hi", model="fake", input_tokens=3, output_tokens=1)


class BoomLLM:
    async def complete(self, messages, *, temperature: float = 0.0) -> LLMResponse:
        raise RuntimeError("llm down")


MSG = [ChatMessage(role=Role.USER, content="hello")]


async def test_traces_successful_call() -> None:
    lf = FakeLangfuse()
    client = TracingLLMClient(FakeLLM(), lf)
    result = await client.complete(MSG)
    assert result.content == "hi"
    assert lf.generation.updated["output"] == "hi"
    assert lf.generation.updated["usage_details"] == {"input": 3, "output": 1}
    assert lf.generation.ended is True


async def test_records_error_and_reraises() -> None:
    lf = FakeLangfuse()
    client = TracingLLMClient(BoomLLM(), lf)
    try:
        await client.complete(MSG)
        raise AssertionError("should have raised")
    except RuntimeError:
        pass
    assert lf.generation.updated["level"] == "ERROR"
    assert lf.generation.ended is True  # span closed even on failure
