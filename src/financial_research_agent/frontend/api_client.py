"""Typed synchronous client for the research API (Streamlit is sync)."""

import httpx
from pydantic import BaseModel

from financial_research_agent.logging_config import get_logger

log = get_logger(__name__)


class RunInfo(BaseModel):
    """Run status as returned by the API."""

    id: str
    ticker: str
    status: str
    error: str | None = None


class ResearchAPIClient:
    """Thin client over the FastAPI service."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        client: httpx.Client | None = None,
    ) -> None:
        headers = {"X-API-Key": api_key} if api_key else {}
        self._client = client or httpx.Client(
            base_url=base_url, headers=headers, timeout=15.0
        )

    def health(self) -> bool:
        """True if the API is reachable and healthy."""
        try:
            return self._client.get("/health").status_code == 200
        except httpx.HTTPError:
            return False

    def create_run(self, ticker: str) -> RunInfo:
        """Start a research run."""
        response = self._client.post("/runs", json={"ticker": ticker})
        response.raise_for_status()
        return RunInfo.model_validate(response.json())

    def get_run(self, run_id: str) -> RunInfo:
        """Fetch current run status."""
        response = self._client.get(f"/runs/{run_id}")
        response.raise_for_status()
        return RunInfo.model_validate(response.json())

    def get_report(self, run_id: str) -> bytes:
        """Download the finished PDF."""
        response = self._client.get(f"/runs/{run_id}/report")
        response.raise_for_status()
        return response.content
