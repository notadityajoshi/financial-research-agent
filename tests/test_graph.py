"""Offline graph tests: fake clients, no network, no LLM."""

from financial_research_agent.agents.graph import build_research_graph
from financial_research_agent.agents.state import ResearchState
from financial_research_agent.integrations.financial_data import FinancialFact
from financial_research_agent.integrations.news import NewsArticle
from financial_research_agent.integrations.sec_edgar import Filing

FACT = FinancialFact(
    metric="revenue",
    concept="Revenues",
    value=100.0,
    unit="USD",
    fiscal_year=2025,
    end_date="2025-12-31",
    form="10-K",
)
FILING = Filing(
    cik=1,
    company_name="Test Co",
    form_type="10-K",
    filing_date="2026-01-01",
    accession_number="0000000001-26-000001",
    primary_document="test.htm",
)
ARTICLE = NewsArticle(title="Test", source="Wire", url="http://x", published="")


class FakeSEC:
    async def get_recent_filings(self, ticker: str, limit: int = 3) -> list[Filing]:
        return [FILING]


class FakeFinancial:
    async def get_annual_facts(self, ticker: str) -> dict[str, list[FinancialFact]]:
        return {"revenue": [FACT]}


class FailingFinancial:
    async def get_annual_facts(self, ticker: str) -> dict[str, list[FinancialFact]]:
        raise RuntimeError("SEC is down")


class FakeNews:
    async def get_company_news(self, company: str, limit: int = 8) -> list[NewsArticle]:
        return [ARTICLE]


async def test_happy_path_populates_state() -> None:
    graph = build_research_graph(FakeSEC(), FakeFinancial(), FakeNews())
    result = ResearchState(**await graph.ainvoke(ResearchState(ticker="NVDA")))
    assert result.filings and result.news
    assert result.metrics is not None
    assert result.metrics.annual[0].revenue == 100.0
    assert result.errors == []


async def test_fault_isolation_keeps_graph_alive() -> None:
    graph = build_research_graph(FakeSEC(), FailingFinancial(), FakeNews())
    result = ResearchState(**await graph.ainvoke(ResearchState(ticker="NVDA")))
    assert result.filings and result.news  # healthy branches unaffected
    assert result.metrics is None
    failed = {e.node for e in result.errors}
    assert failed == {"fetch_facts", "compute_metrics"}