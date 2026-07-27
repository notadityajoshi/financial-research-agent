"""Assembly of the research graph: parallel collection, metrics, analysis, summary."""

from typing import cast

from langgraph.graph import END, START, StateGraph

from financial_research_agent.agents.analysts import (
    make_opportunity_analyst,
    make_risk_analyst,
)
from financial_research_agent.agents.nodes import (
    FactsSource,
    FilingSource,
    NewsSource,
    make_compute_metrics,
    make_fetch_facts,
    make_fetch_filings,
    make_fetch_news,
)
from financial_research_agent.agents.rag import make_filing_analyst, make_index_filing
from financial_research_agent.agents.report import make_summary_writer
from financial_research_agent.agents.state import ResearchState
from financial_research_agent.llm.base import LLMClient
from financial_research_agent.retrieval.indexer import FilingDownloader
from financial_research_agent.retrieval.reranker import Reranker
from financial_research_agent.retrieval.vector_store import VectorStore


def build_research_graph(
    sec: FilingSource,
    financial: FactsSource,
    news: NewsSource,
    llm: LLMClient,
    store: VectorStore | None = None,
    reranker: Reranker | None = None,
):
    """START → parallel fetch → metrics → parallel analysts → summary → END.

    With a vector store, a RAG branch (index_filing → analyze_filing)
    runs in parallel and also feeds the summary. In that case `sec`
    must also satisfy FilingDownloader (the real SECEdgarClient does).
    """
    graph = StateGraph(ResearchState)

    graph.add_node("fetch_filings", make_fetch_filings(sec))
    graph.add_node("fetch_facts", make_fetch_facts(financial))
    graph.add_node("fetch_news", make_fetch_news(news))
    graph.add_node("compute_metrics", make_compute_metrics())
    graph.add_node("analyze_risks", make_risk_analyst(llm))
    graph.add_node("analyze_opportunities", make_opportunity_analyst(llm))
    graph.add_node("generate_summary", make_summary_writer(llm))

    for fetcher in ("fetch_filings", "fetch_facts", "fetch_news"):
        graph.add_edge(START, fetcher)
        graph.add_edge(fetcher, "compute_metrics")

    for analyst in ("analyze_risks", "analyze_opportunities"):
        graph.add_edge("compute_metrics", analyst)
        graph.add_edge(analyst, "generate_summary")

    if store is not None:
        downloader = cast(FilingDownloader, sec)
        graph.add_node("index_filing", make_index_filing(downloader, store))
        graph.add_node("analyze_filing", make_filing_analyst(store, llm, reranker))
        graph.add_edge("fetch_filings", "index_filing")
        graph.add_edge("index_filing", "analyze_filing")
        graph.add_edge("analyze_filing", "generate_summary")

    graph.add_edge("generate_summary", END)

    return graph.compile()