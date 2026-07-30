"""Offline MCP tool tests: fake HTTP and facts dependencies injected."""

import httpx

from financial_research_agent.integrations.financial_data import FinancialFact
from financial_research_agent.mcp_server import (
    deps,
    get_financial_metrics,
    get_research_status,
    start_research,
)

RUN = {"id": "abc", "ticker": "NVDA", "status": "pending", "error": None}


class FakeHTTP:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict]] = []

    async def post(self, url: str, json: dict) -> httpx.Response:
        self.posts.append((url, json))
        return httpx.Response(
            202, json=RUN, request=httpx.Request("POST", f"http://t{url}")
        )

    async def get(self, url: str) -> httpx.Response:
        return httpx.Response(
            200,
            json={**RUN, "status": "completed"},
            request=httpx.Request("GET", f"http://t{url}"),
        )


class FakeFacts:
    async def get_annual_facts(self, ticker: str) -> dict:
        return {
            "revenue": [
                FinancialFact(
                    metric="revenue",
                    concept="Revenues",
                    value=100.0,
                    unit="USD",
                    fiscal_year=2025,
                    end_date="2025-12-31",
                    form="10-K",
                )
            ]
        }


async def test_start_research_uppercases_and_returns_id(monkeypatch) -> None:
    fake = FakeHTTP()
    monkeypatch.setattr(deps, "_http", fake)
    result = await start_research("nvda")
    assert fake.posts == [("/runs", {"ticker": "NVDA"})]
    assert result == {"run_id": "abc", "status": "pending"}


async def test_get_research_status(monkeypatch) -> None:
    monkeypatch.setattr(deps, "_http", FakeHTTP())
    result = await get_research_status("abc")
    assert result["status"] == "completed"


async def test_get_financial_metrics_is_deterministic(monkeypatch) -> None:
    monkeypatch.setattr(deps, "_facts", FakeFacts())
    result = await get_financial_metrics("nvda")
    assert result["annual"][0]["revenue"] == 100.0
