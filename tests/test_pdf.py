"""Offline PDF rendering tests."""

from financial_research_agent.agents.schemas import (
    AnalysisItem,
    EvidenceRef,
    GroundedInsight,
    InvestmentSummary,
)
from financial_research_agent.agents.state import NodeError, ResearchState
from financial_research_agent.reports.pdf import render_pdf


def _full_state() -> ResearchState:
    state = ResearchState(ticker="NVDA")
    state.summary = InvestmentSummary(
        headline="Head", thesis="Thesis.", strengths=["S1"], concerns=["C1"]
    )
    state.risks = [
        AnalysisItem(title="Risk — dash", detail="Unicode ' test.", severity="high")
    ]
    state.filing_insights = [
        GroundedInsight(
            title="Insight",
            detail="Detail.",
            severity="medium",
            evidence=[
                EvidenceRef(
                    excerpt="Excerpt", form_type="10-K", filing_date="2026-01-01"
                )
            ],
        )
    ]
    state.errors = [NodeError(node="fetch_news", message="down")]
    return state


def test_renders_valid_pdf_bytes() -> None:
    data = render_pdf(_full_state())
    assert data.startswith(b"%PDF")
    assert len(data) > 1000


def test_empty_state_still_renders() -> None:
    assert render_pdf(ResearchState(ticker="X")).startswith(b"%PDF")


def test_unicode_does_not_crash() -> None:
    state = ResearchState(ticker="NVDA")
    state.risks = [
        AnalysisItem(
            title="Em—dash 'quotes' €", detail="More — unicode.", severity="low"
        )
    ]
    assert render_pdf(state).startswith(b"%PDF")
