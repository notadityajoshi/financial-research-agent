"""LLM decorator that accumulates token cost across a run."""

import asyncio

from financial_research_agent.core.cost import CostBreakdown, compute_cost
from financial_research_agent.llm.base import ChatMessage, LLMClient, LLMResponse
from financial_research_agent.logging_config import get_logger

log = get_logger(__name__)


class CostTrackingLLMClient:
    """Wraps an LLMClient; sums the cost of every call it makes.

    One instance per run gives that run's total spend. Safe under the
    concurrent analyst/RAG branches via an async lock.
    """

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self._total = CostBreakdown()
        self._lock = asyncio.Lock()

    async def complete(
        self, messages: list[ChatMessage], *, temperature: float = 0.0
    ) -> LLMResponse:
        """Delegate, then add this call's cost to the running total."""
        response = await self._inner.complete(messages, temperature=temperature)
        breakdown = compute_cost(
            response.model, response.input_tokens, response.output_tokens
        )
        async with self._lock:
            self._total = self._total + breakdown
        return response

    @property
    def total(self) -> CostBreakdown:
        """Accumulated cost across all calls so far."""
        return self._total