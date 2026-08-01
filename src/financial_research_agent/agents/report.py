"""Summary node: synthesise every analysis stream into an executive view."""

from financial_research_agent.agents.analysts import build_context
from financial_research_agent.agents.nodes import Node, fault_isolated
from financial_research_agent.agents.schemas import InvestmentSummary
from financial_research_agent.agents.state import ResearchState
from financial_research_agent.llm.base import ChatMessage, LLMClient, Role
from financial_research_agent.llm.structured import generate_structured


def _bullet_block(title: str, items: list[str]) -> str:
    body = "\n".join(f"- {i}" for i in items) if items else "- none identified"
    return f"{title}:\n{body}"


def make_summary_writer(llm: LLMClient) -> Node:
    """Node factory: write the investment summary from accumulated state."""

    async def generate_summary(state: ResearchState) -> dict:
        context = "\n\n".join(
            (
                build_context(state),
                _bullet_block(
                    "Identified risks",
                    [f"[{r.severity}] {r.title}: {r.detail}" for r in state.risks],
                ),
                _bullet_block(
                    "Identified opportunities",
                    [
                        f"[{o.severity}] {o.title}: {o.detail}"
                        for o in state.opportunities
                    ],
                ),
                _bullet_block(
                    "Filing insights (cited from 10-K)",
                    [f"{i.title}: {i.detail}" for i in state.filing_insights],
                ),
            )
        )
        messages = [
            ChatMessage(
                role=Role.SYSTEM,
                content=(
                    "You are a senior equity research editor. Synthesise the "
                    "provided analysis into a balanced executive summary. Use "
                    "ONLY the provided material; invent no figures. This is "
                    "research, not investment advice."
                ),
            ),
            ChatMessage(role=Role.USER, content=context),
        ]
        summary = await generate_structured(llm, messages, InvestmentSummary)
        return {"summary": summary}

    return fault_isolated("generate_summary", generate_summary)
