"""Offline summary-node test with a fake LLM."""

import json

from financial_research_agent.agents.report import make_summary_writer
from financial_research_agent.agents.state import ResearchState
from financial_research_agent.llm.base import LLMResponse

SUMMARY = {
    "headline": "Strong but concentrated.",
    "thesis": "Growth is exceptional. Concentration is the core risk.",
    "strengths": ["Growth"],
    "concerns": ["Concentration"],
}


class FakeLLM:
    async def complete(self, messages, *, temperature: float = 0.0):
        return LLMResponse(
            content=json.dumps(SUMMARY), model="fake", input_tokens=0, output_tokens=0
        )


async def test_summary_node_populates_state() -> None:
    out = await make_summary_writer(FakeLLM())(ResearchState(ticker="NVDA"))
    assert out["summary"].headline == "Strong but concentrated."
    assert out["summary"].concerns == ["Concentration"]