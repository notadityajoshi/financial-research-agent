"""API route tests: fake service, ASGI transport, fully offline."""

import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from financial_research_agent.api.main import create_app
from financial_research_agent.api.service import ResearchService
from financial_research_agent.db.models import ResearchRun, RunStatus

class FakeQueue:
    """Records enqueued jobs."""

    def __init__(self) -> None:
        self.jobs: list[str] = []

    async def enqueue(self, run_id: uuid.UUID, ticker: str) -> None:
        self.jobs.append(ticker)
class FakeService(ResearchService):
    """In-memory service; records execute calls."""

    def __init__(self, reports_dir: Path) -> None:
        self._runs: dict[uuid.UUID, ResearchRun] = {}
        self._reports_dir = reports_dir
        self.executed: list[str] = []

    async def create_run(self, ticker: str) -> ResearchRun:
        run = ResearchRun(
            id=uuid.uuid4(),
            ticker=ticker,
            status=RunStatus.PENDING,
            error=None,
            created_at=datetime.now(UTC),
        )
        self._runs[run.id] = run
        return run

    async def get_run(self, run_id: uuid.UUID) -> ResearchRun | None:
        return self._runs.get(run_id)

    async def execute(self, run_id: uuid.UUID, ticker: str) -> None:
        self.executed.append(ticker)


@pytest.fixture
def fake(tmp_path: Path) -> FakeService:
    return FakeService(tmp_path)

@pytest.fixture
def fake_queue() -> FakeQueue:
    return FakeQueue()


@pytest.fixture
async def client(fake: FakeService, fake_queue: FakeQueue, monkeypatch):
    from financial_research_agent import config

    monkeypatch.setenv("API_KEYS", "test-key")
    config.get_settings.cache_clear()
    transport = httpx.ASGITransport(app=create_app(service=fake, queue=fake_queue))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"X-API-Key": "test-key"},
    ) as c:
        yield c
    config.get_settings.cache_clear()
@pytest.fixture
async def client(fake: FakeService, monkeypatch):
    from financial_research_agent import config

    monkeypatch.setenv("API_KEYS", "test-key")
    config.get_settings.cache_clear()
    transport = httpx.ASGITransport(app=create_app(service=fake))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"X-API-Key": "test-key"},
    ) as c:
        yield c
    config.get_settings.cache_clear()


async def test_health(client) -> None:
    assert (await client.get("/health")).status_code == 200


async def test_create_run_schedules_execution(client, fake_queue: FakeQueue) -> None:
    response = await client.post("/runs", json={"ticker": "nvda"})
    assert response.status_code == 202
    body = response.json()
    assert body["ticker"] == "NVDA"  # validator uppercased
    assert body["status"] == "pending"
    assert fake_queue.jobs == ["NVDA"]  # enqueued, not executed inline

async def test_get_run(client, fake: FakeService) -> None:
    run = await fake.create_run("NVDA")
    response = await client.get(f"/runs/{run.id}")
    assert response.status_code == 200
    assert response.json()["id"] == str(run.id)


async def test_get_run_404(client) -> None:
    assert (await client.get(f"/runs/{uuid.uuid4()}")).status_code == 404


async def test_bad_ticker_422(client) -> None:
    assert (await client.post("/runs", json={"ticker": "NV DA!"})).status_code == 422


async def test_report_not_ready_409(client, fake: FakeService) -> None:
    run = await fake.create_run("NVDA")
    assert (await client.get(f"/runs/{run.id}/report")).status_code == 409


async def test_report_download(client, fake: FakeService) -> None:
    run = await fake.create_run("NVDA")
    run.status = RunStatus.COMPLETED
    fake.report_path(run.id).write_bytes(b"%PDF-fake")
    response = await client.get(f"/runs/{run.id}/report")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")