"""Worker task tests: fake service injected via ctx."""

import uuid

from financial_research_agent.api.queue import InlineJobQueue
from financial_research_agent.worker import run_research


class FakeExecService:
    def __init__(self) -> None:
        self.calls: list[tuple[uuid.UUID, str]] = []

    async def execute(self, run_id: uuid.UUID, ticker: str) -> None:
        self.calls.append((run_id, ticker))


async def test_run_research_delegates_to_service() -> None:
    service = FakeExecService()
    run_id = uuid.uuid4()
    await run_research({"service": service}, str(run_id), "NVDA")
    assert service.calls == [(run_id, "NVDA")]


async def test_inline_queue_executes_immediately() -> None:
    service = FakeExecService()
    run_id = uuid.uuid4()
    await InlineJobQueue(service).enqueue(run_id, "NVDA")
    assert service.calls == [(run_id, "NVDA")]
