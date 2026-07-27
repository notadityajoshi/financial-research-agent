"""Offline analyst-node tests: fake LLM, fake data clients."""

import json

from financial_research_agent.agents.graph import build_research_graph
from financial_research_agent.agents.state import ResearchState
from financial_research_agent.llm.base import LLMResponse
from tests.test_graph import FakeFinancial, FakeNews, FakeSEC

ITEMS = {
    "items": [
        {"title": "Concentration", "detail": "Very concentrated.", "severity": "high"}
    ],
    "headline": "h",
    "thesis": "t",
    "strengths": [],
    "concerns": [],
}


class FakeLLM:
    async def complete(self, messages, *, temperature: float = 0.0):
        return LLMResponse(
            content=json.dumps(ITEMS), model="fake", input_tokens=0, output_tokens=0
        )


async def test_graph_populates_analyses() -> None:
    graph = build_research_graph(FakeSEC(), FakeFinancial(), FakeNews(), FakeLLM())
    result = ResearchState(**await graph.ainvoke(ResearchState(ticker="NVDA")))
    assert result.risks[0].title == "Concentration"
    assert result.opportunities[0].severity == "high"
    assert result.errors == []


def test_context_contains_precomputed_numbers() -> None:
    from financial_research_agent.agents.analysts import build_context
    from financial_research_agent.core.metrics import compute_metrics

    state = ResearchState(ticker="NVDA")
    state.facts = {"revenue": []}
    state.metrics = compute_metrics({"revenue": []})
    assert "Annual metrics" in build_context(state)