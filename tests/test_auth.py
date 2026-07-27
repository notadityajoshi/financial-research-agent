"""Auth dependency tests: valid, invalid, missing, disabled."""

import httpx
import pytest

from financial_research_agent import config
from financial_research_agent.api.main import create_app
from tests.test_api import FakeService


@pytest.fixture
async def raw_client(tmp_path, monkeypatch):
    """Client with keys configured but NO key header attached."""
    monkeypatch.setenv("API_KEYS", "secret-1,secret-2")
    config.get_settings.cache_clear()
    transport = httpx.ASGITransport(app=create_app(service=FakeService(tmp_path)))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    config.get_settings.cache_clear()


async def test_missing_key_401(raw_client) -> None:
    assert (await raw_client.post("/runs", json={"ticker": "NVDA"})).status_code == 401


async def test_wrong_key_401(raw_client) -> None:
    response = await raw_client.post(
        "/runs", json={"ticker": "NVDA"}, headers={"X-API-Key": "wrong"}
    )
    assert response.status_code == 401


async def test_second_valid_key_accepted(raw_client) -> None:
    response = await raw_client.post(
        "/runs", json={"ticker": "NVDA"}, headers={"X-API-Key": "secret-2"}
    )
    assert response.status_code == 202


async def test_health_open_without_key(raw_client) -> None:
    assert (await raw_client.get("/health")).status_code == 200


async def test_auth_disabled_when_no_keys(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("API_KEYS", "")
    config.get_settings.cache_clear()
    transport = httpx.ASGITransport(app=create_app(service=FakeService(tmp_path)))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        assert (await c.post("/runs", json={"ticker": "NVDA"})).status_code == 202
    config.get_settings.cache_clear()