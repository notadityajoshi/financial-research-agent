"""Offline rate-limit tests: fakeredis, injected limiter."""

import httpx
import pytest
from fakeredis import FakeAsyncRedis

from financial_research_agent import config
from financial_research_agent.api.main import create_app
from financial_research_agent.api.rate_limit import RateLimiter
from tests.test_api import FakeService


async def test_allow_within_limit() -> None:
    limiter = RateLimiter(FakeAsyncRedis(), limit=3)
    results = [await limiter.allow("key:abc") for _ in range(3)]
    assert results == [True, True, True]


async def test_deny_over_limit() -> None:
    limiter = RateLimiter(FakeAsyncRedis(), limit=2)
    await limiter.allow("key:abc")
    await limiter.allow("key:abc")
    assert await limiter.allow("key:abc") is False


async def test_identities_isolated() -> None:
    limiter = RateLimiter(FakeAsyncRedis(), limit=1)
    assert await limiter.allow("key:a") is True
    assert await limiter.allow("key:b") is True


class _BrokenRedis:
    def pipeline(self, transaction: bool = True):
        raise ConnectionError("redis down")


async def test_fails_open_when_redis_down() -> None:
    limiter = RateLimiter(_BrokenRedis(), limit=1)
    assert await limiter.allow("key:a") is True
    assert await limiter.allow("key:a") is True


@pytest.fixture
async def limited_client(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEYS", "test-key")
    config.get_settings.cache_clear()
    limiter = RateLimiter(FakeAsyncRedis(), limit=2)
    app = create_app(service=FakeService(tmp_path), limiter=limiter)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"X-API-Key": "test-key"},
    ) as c:
        yield c
    config.get_settings.cache_clear()


async def test_429_after_budget(limited_client) -> None:
    codes = [
        (await limited_client.post("/runs", json={"ticker": "NVDA"})).status_code
        for _ in range(3)
    ]
    assert codes == [202, 202, 429]


async def test_polling_not_limited(limited_client) -> None:
    for _ in range(5):
        await limited_client.post("/runs", json={"ticker": "NVDA"})
    run_id = "00000000-0000-0000-0000-000000000000"
    assert (await limited_client.get(f"/runs/{run_id}")).status_code == 404  # not 429