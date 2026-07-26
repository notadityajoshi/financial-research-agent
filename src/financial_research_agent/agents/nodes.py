"""Graph nodes: thin, fault-isolated wrappers over existing components."""

from collections.abc import Awaitable, Callable

from financial_research_agent.agents.state import NodeError, ResearchState
from financial_research_agent.core.metrics import compute_metrics
from financial_research_agent.integrations.financial_data import FinancialDataClient
from financial_research_agent.integrations.news import NewsClient
from financial_research_agent.integrations.sec_edgar import SECEdgarClient
from financial_research_agent.logging_config import get_logger

log = get_logger(__name__)

Node = Callable[[ResearchState], Awaitable[dict]]


def fault_isolated(name: str, node: Node) -> Node:
    """Wrap a node: on exception, record a NodeError and keep the graph alive."""

    async def wrapper(state: ResearchState) -> dict:
        try:
            log.info("node_start", node=name, ticker=state.ticker)
            result = await node(state)
            log.info("node_done", node=name)
            return result
        except Exception as exc:  # noqa: BLE001 — isolation boundary by design
            log.error("node_failed", node=name, error=str(exc))
            return {"errors": [NodeError(node=name, message=str(exc))]}

    return wrapper


def make_fetch_filings(sec: SECEdgarClient) -> Node:
    """Node factory: list recent SEC filings for the ticker."""

    async def fetch_filings(state: ResearchState) -> dict:
        filings = await sec.get_recent_filings(state.ticker, limit=3)
        return {"filings": filings}

    return fault_isolated("fetch_filings", fetch_filings)


def make_fetch_facts(financial: FinancialDataClient) -> Node:
    """Node factory: load annual XBRL fundamentals."""

    async def fetch_facts(state: ResearchState) -> dict:
        return {"facts": await financial.get_annual_facts(state.ticker)}

    return fault_isolated("fetch_facts", fetch_facts)


def make_fetch_news(news: NewsClient) -> Node:
    """Node factory: load recent company news."""

    async def fetch_news(state: ResearchState) -> dict:
        return {"news": await news.get_company_news(state.ticker, limit=8)}

    return fault_isolated("fetch_news", fetch_news)


def make_compute_metrics() -> Node:
    """Node factory: deterministic ratios from whatever facts arrived."""

    async def compute(state: ResearchState) -> dict:
        if not state.facts:
            return {
                "errors": [
                    NodeError(node="compute_metrics", message="no facts available")
                ]
            }
        return {"metrics": compute_metrics(state.facts)}

    return fault_isolated("compute_metrics", compute)