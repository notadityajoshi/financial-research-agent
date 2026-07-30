"""Offline API-client tests via httpx MockTransport."""

import json

import httpx
import pytest

from financial_research_agent.frontend.api_client import ResearchAPIClient

RUN = {"id": "abc", "ticker": "NVDA", "status": "pending", "error": None}


def _client(handler) -> ResearchAPIClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(
        transport=transport, base_url="http://test", headers={"X-API-Key": "k"}
    )
    return ResearchAPIClient("http://test", client=http)


def test_create_run_posts_ticker_and_parses() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/runs"
        assert request.headers["X-API-Key"] == "k"
        assert json.loads(request.content) == {"ticker": "NVDA"}
        return httpx.Response(202, json=RUN)

    run = _client(handler).create_run("NVDA")
    assert (run.id, run.status) == ("abc", "pending")


def test_get_report_returns_bytes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"%PDF-fake")

    assert _client(handler).get_report("abc").startswith(b"%PDF")


def test_error_status_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "nope"})

    with pytest.raises(httpx.HTTPStatusError):
        _client(handler).create_run("NVDA")


def test_health_false_when_down() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    assert _client(handler).health() is False