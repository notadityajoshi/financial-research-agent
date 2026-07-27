"""HTTP routes for the research API."""

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from financial_research_agent.api.schemas import RunCreateRequest, RunResponse
from financial_research_agent.api.service import ResearchService
from financial_research_agent.db.models import RunStatus

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from financial_research_agent.api.auth import require_api_key
router = APIRouter()
protected = APIRouter(dependencies=[Depends(require_api_key)])

def get_service(request: Request) -> ResearchService:
    """Dependency: the app-wide research service."""
    return request.app.state.service


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@protected.post("/runs", status_code=202, response_model=RunResponse)
async def create_run(
    body: RunCreateRequest,
    background_tasks: BackgroundTasks,
    service: ResearchService = Depends(get_service),
) -> RunResponse:
    """Create a run and execute it in the background."""
    run = await service.create_run(body.ticker)
    background_tasks.add_task(service.execute, run.id, run.ticker)
    return RunResponse.model_validate(run)


@protected.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: uuid.UUID, service: ResearchService = Depends(get_service)
) -> RunResponse:
    """Fetch run status."""
    run = await service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return RunResponse.model_validate(run)


@protected.get("/runs/{run_id}/report")
async def get_report(
    run_id: uuid.UUID, service: ResearchService = Depends(get_service)
) -> FileResponse:
    """Download the finished PDF report."""
    run = await service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    if run.status is not RunStatus.COMPLETED:
        raise HTTPException(
            status_code=409, detail=f"run is {run.status.value}, not completed"
        )
    path = service.report_path(run_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="report file missing")
    return FileResponse(
        path, media_type="application/pdf", filename=f"{run.ticker}_report.pdf"
    )

router.include_router(protected)