"""Research service: owns the run lifecycle from creation to report."""

import uuid
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Protocol
from typing import Any, Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from financial_research_agent.agents.state import ResearchState
from financial_research_agent.db.engine import get_session
from financial_research_agent.db.models import ResearchRun, RunStatus
from financial_research_agent.logging_config import get_logger
from financial_research_agent.reports.pdf import render_pdf

log = get_logger(__name__)

SessionScope = Callable[[], AbstractAsyncContextManager[AsyncSession]]


class GraphLike(Protocol):
    """Anything invokable like a compiled LangGraph graph."""

    async def ainvoke(self, state: ResearchState) -> dict[str, Any]: ...


class ResearchService:
    """Creates, executes and reports on research runs."""

    def __init__(
        self,
        graph: GraphLike,
        reports_dir: Path,
        session_scope: SessionScope = get_session,
    ) -> None:
        self._graph = graph
        self._reports_dir = reports_dir
        self._session_scope = session_scope

    async def create_run(self, ticker: str) -> ResearchRun:
        """Persist a new pending run."""
        async with self._session_scope() as session:
            run = ResearchRun(ticker=ticker)
            session.add(run)
        log.info("run_created", run_id=str(run.id), ticker=ticker)
        return run

    async def get_run(self, run_id: uuid.UUID) -> ResearchRun | None:
        """Load one run by id."""
        async with self._session_scope() as session:
            return await session.get(ResearchRun, run_id)

    def report_path(self, run_id: uuid.UUID) -> Path:
        """Deterministic on-disk location of a run's PDF."""
        return self._reports_dir / f"{run_id}.pdf"

    async def _set_status(
        self, run_id: uuid.UUID, status: RunStatus, *, error: str | None = None
    ) -> None:
        async with self._session_scope() as session:
            run = await session.get(ResearchRun, run_id)
            if run is not None:
                run.status = status
                run.error = error

    async def execute(self, run_id: uuid.UUID, ticker: str) -> None:
        """Run the full pipeline for one run; never raises (records failure)."""
        await self._set_status(run_id, RunStatus.RUNNING)
        try:
            raw = await self._graph.ainvoke(ResearchState(ticker=ticker))
            state = ResearchState(**raw)
            path = self.report_path(run_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(render_pdf(state))
            await self._set_status(run_id, RunStatus.COMPLETED)
            log.info("run_completed", run_id=str(run_id))
        except Exception as exc:  # noqa: BLE001 — service boundary
            await self._set_status(run_id, RunStatus.FAILED, error=str(exc))
            log.error("run_failed", run_id=str(run_id), error=str(exc))