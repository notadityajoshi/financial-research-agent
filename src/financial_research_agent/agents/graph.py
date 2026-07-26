"""Assembly of the research graph: parallel collection, then computation."""

from langgraph.graph import END, START, StateGraph

from financial_research_agent.agents.nodes import (
    make_compute_metrics,
    make_fetch_facts,
    make_fetch_filings,
    make_fetch_news,
)
from financial_research_agent.agents.state import ResearchState
from financial_research_agent.integrations.financial_data import FinancialDataClient
from financial_research_agent.integrations.news import NewsClient
from financial_research_agent.integrations.sec_edgar import SECEdgarClient


def build_research_graph(
    sec: SECEdgarClient,
    financial: FinancialDataClient,
    news: NewsClient,
):
    """Compile the graph: START fans out to three fetchers, joins at metrics."""
    graph = StateGraph(ResearchState)

    graph.add_node("fetch_filings", make_fetch_filings(sec))
    graph.add_node("fetch_facts", make_fetch_facts(financial))
    graph.add_node("fetch_news", make_fetch_news(news))
    graph.add_node("compute_metrics", make_compute_metrics())

    for fetcher in ("fetch_filings", "fetch_facts", "fetch_news"):
        graph.add_edge(START, fetcher)
        graph.add_edge(fetcher, "compute_metrics")
    graph.add_edge("compute_metrics", END)

    return graph.compile()