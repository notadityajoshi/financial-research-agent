"""Service lifecycle tests: in-memory SQLite, fake graph, tmp reports dir."""

from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from financial_research_agent.agents.state import ResearchState
from financial_research_agent.api.service import ResearchService
from financial_research_agent.db.models import Base, RunStatus


class FakeGraph:
    async def ainvoke(self, state: ResearchState) -> dict:
        return ResearchState(ticker=state.ticker).model_dump()


class FailingGraph:
    async def ainvoke(self, state: ResearchState) -> dict:
        raise RuntimeError("graph exploded")


@pytest.fixture
async def session_scope():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def scope():
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    return scope


async def test_execute_success(session_scope, tmp_path: Path) -> None:
    service = ResearchService(FakeGraph(), tmp_path, session_scope)
    run = await service.create_run("NVDA")
    await service.execute(run.id, run.ticker)

    loaded = await service.get_run(run.id)
    assert loaded is not None
    assert loaded.status is RunStatus.COMPLETED


async def test_execute_failure_recorded(session_scope, tmp_path: Path) -> None:
    service = ResearchService(FailingGraph(), tmp_path, session_scope)
    run = await service.create_run("NVDA")
    await service.execute(run.id, run.ticker)

    loaded = await service.get_run(run.id)
    assert loaded is not None
    assert loaded.status is RunStatus.FAILED
    assert loaded.error is not None
    assert "graph exploded" in loaded.error
