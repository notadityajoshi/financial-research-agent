"""Offline cost tests: pure calculation + accumulating decorator."""

from financial_research_agent.core.cost import CostBreakdown, compute_cost
from financial_research_agent.llm.base import ChatMessage, LLMResponse, Role
from financial_research_agent.llm.cost_tracking import CostTrackingLLMClient


def test_known_model_priced() -> None:
    cost = compute_cost("gpt-4o-mini", 1_000_000, 1_000_000)
    assert cost.usd == round(0.15 + 0.60, 6)


def test_local_model_free() -> None:
    assert compute_cost("llama3.2:3b", 5000, 5000).usd == 0.0


def test_unknown_model_free() -> None:
    assert compute_cost("mystery", 1000, 1000).usd == 0.0


def test_breakdown_addition() -> None:
    total = compute_cost("gpt-4o-mini", 1000, 0) + compute_cost("gpt-4o-mini", 0, 1000)
    assert total.input_tokens == 1000
    assert total.output_tokens == 1000


class SpyLLM:
    async def complete(self, messages, *, temperature: float = 0.0) -> LLMResponse:
        return LLMResponse(
            content="ok", model="gpt-4o-mini", input_tokens=1000, output_tokens=500
        )


async def test_decorator_accumulates() -> None:
    client = CostTrackingLLMClient(SpyLLM())
    msg = [ChatMessage(role=Role.USER, content="hi")]
    await client.complete(msg)
    await client.complete(msg)
    assert client.total.input_tokens == 2000
    assert client.total.output_tokens == 1000
    assert client.total.usd > 0