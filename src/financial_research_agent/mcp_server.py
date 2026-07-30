"""MCP server: exposes the research system as tools for AI clients."""

from typing import Protocol

import httpx
from mcp.server.fastmcp import FastMCP

from financial_research_agent.config import get_settings
from financial_research_agent.core.metrics import compute_metrics
from financial_research_agent.integrations.financial_data import FinancialDataClient
from financial_research_agent.logging_config import configure_logging, get_logger

log = get_logger(__name__)

mcp = FastMCP("financial-research-agent")


class RunAPI(Protocol):
    """Anything that can start and poll runs (real HTTP client or fake)."""

    async def post(self, url: str, *, json: dict) -> httpx.Response: ...
    async def get(self, url: str) -> httpx.Response: ...
    

class FactsAPI(Protocol):
    """Anything that provides annual fundamentals."""

    async def get_annual_facts(self, ticker: str) -> dict: ...


class Deps:
    """Injectable dependencies (swapped for fakes in tests)."""

    def __init__(self) -> None:
        self._http: httpx.AsyncClient | None = None
        self._facts: FactsAPI | None = None

    @property
    def http(self) -> RunAPI:
        if self._http is None:
            settings = get_settings()
            key = next(iter(settings.api_key_set), "")
            self._http = httpx.AsyncClient(
                base_url=settings.api_base_url,
                headers={"X-API-Key": key} if key else {},
                timeout=15.0,
            )
        return self._http

    @property
    def facts(self) -> FactsAPI:
        if self._facts is None:
            self._facts = FinancialDataClient()
        return self._facts


deps = Deps()


@mcp.tool()
async def start_research(ticker: str) -> dict:
    """Start a full research run (filings, metrics, cited RAG insights, PDF).

    Returns the run id; poll with get_research_status.
    """
    response = await deps.http.post("/runs", json={"ticker": ticker.upper()})
    response.raise_for_status()
    body = response.json()
    log.info("mcp_run_started", run_id=body["id"], ticker=ticker.upper())
    return {"run_id": body["id"], "status": body["status"]}


@mcp.tool()
async def get_research_status(run_id: str) -> dict:
    """Get the status of a research run (pending, running, completed, failed)."""
    response = await deps.http.get(f"/runs/{run_id}")
    response.raise_for_status()
    body = response.json()
    return {"run_id": body["id"], "status": body["status"], "error": body["error"]}


@mcp.tool()
async def get_financial_metrics(ticker: str) -> dict:
    """Deterministic multi-year financial metrics from official SEC XBRL data.

    Fast: no LLM involved. Margins, growth, ROE/ROA, leverage, CAGR.
    """
    facts = await deps.facts.get_annual_facts(ticker.upper())
    summary = compute_metrics(facts)
    return summary.model_dump()


def main() -> None:
    """Run the MCP server on stdio (how MCP hosts launch servers)."""
    configure_logging()
    log.info("mcp_server_starting")
    mcp.run()


if __name__ == "__main__":
    main()