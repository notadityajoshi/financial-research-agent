"""LLM analyst nodes: interpret pre-computed data, never calculate."""

from financial_research_agent.agents.nodes import Node, fault_isolated
from financial_research_agent.agents.schemas import (
    OpportunityAnalysis,
    RiskAnalysis,
)
from financial_research_agent.agents.state import ResearchState
from financial_research_agent.llm.base import ChatMessage, LLMClient, Role
from financial_research_agent.llm.structured import generate_structured


def _fmt(value: float | None, suffix: str = "") -> str:
    return f"{value:.1f}{suffix}" if value is not None else "n/a"


def build_context(state: ResearchState) -> str:
    """Compact factual context for analyst prompts (numbers pre-computed)."""
    lines = [f"Company ticker: {state.ticker}", "", "Annual metrics:"]
    if state.metrics:
        for m in state.metrics.annual:
            lines.append(
                f"- FY{m.fiscal_year}: revenue {_fmt(m.revenue and m.revenue / 1e9, 'B USD')}, "
                f"rev growth {_fmt(m.revenue_growth_pct, '%')}, "
                f"net margin {_fmt(m.net_margin_pct, '%')}, "
                f"ROE {_fmt(m.roe_pct, '%')}, "
                f"liabilities/equity {_fmt(m.liabilities_to_equity)}"
            )
        lines.append(f"Revenue CAGR: {_fmt(state.metrics.revenue_cagr_pct, '%')}")
    else:
        lines.append("- unavailable")
    lines += ["", "Recent headlines:"]
    lines += [f"- {a.title} ({a.source})" for a in state.news] or ["- none"]
    return "\n".join(lines)


def _analyst_node(
    llm: LLMClient, *, name: str, role_prompt: str, schema: type
) -> Node:
    async def analyze(state: ResearchState) -> dict:
        messages = [
            ChatMessage(role=Role.SYSTEM, content=role_prompt),
            ChatMessage(role=Role.USER, content=build_context(state)),
        ]
        analysis = await generate_structured(llm, messages, schema)
        return {name: analysis.items}

    return fault_isolated(f"analyze_{name}", analyze)


def make_risk_analyst(llm: LLMClient) -> Node:
    """Node factory: identify material risks from context."""
    return _analyst_node(
        llm,
        name="risks",
        role_prompt=(
            "You are a sceptical equity risk analyst. From ONLY the provided "
            "context, identify the most material risks to the investment case. "
            "Do not invent numbers; reference only figures given."
        ),
        schema=RiskAnalysis,
    )


def make_opportunity_analyst(llm: LLMClient) -> Node:
    """Node factory: identify credible opportunities from context."""
    return _analyst_node(
        llm,
        name="opportunities",
        role_prompt=(
            "You are a measured equity analyst. From ONLY the provided context, "
            "identify credible growth opportunities. Do not invent numbers; "
            "reference only figures given."
        ),
        schema=OpportunityAnalysis,
    )