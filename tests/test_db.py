"""Offline DB tests: in-memory SQLite, same ORM models as Postgres."""

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from financial_research_agent.db.models import Base, ResearchRun, RunStatus


@pytest.fixture
async def session_factory() -> async_sessionmaker:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False)


async def test_create_and_read_run(session_factory: async_sessionmaker) -> None:
    async with session_factory() as session:
        run = ResearchRun(ticker="NVDA")
        session.add(run)
        await session.commit()

    async with session_factory() as session:
        loaded = await session.get(ResearchRun, run.id)
        assert loaded is not None
        assert loaded.ticker == "NVDA"
        assert loaded.status is RunStatus.PENDING
        assert loaded.created_at is not None


async def test_status_transition(session_factory: async_sessionmaker) -> None:
    async with session_factory() as session:
        run = ResearchRun(ticker="NVDA")
        session.add(run)
        await session.commit()
        run.status = RunStatus.FAILED
        run.error = "boom"
        await session.commit()

    async with session_factory() as session:
        loaded = await session.get(ResearchRun, run.id)
        assert loaded.status is RunStatus.FAILED
        assert loaded.error == "boom"