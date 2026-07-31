"""API request/response schemas."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from financial_research_agent.db.models import RunStatus


class RunCreateRequest(BaseModel):
    """Request to start a research run."""

    ticker: str = Field(
        min_length=1, max_length=10, pattern=r"^[A-Za-z][A-Za-z0-9.\-]*$"
    )

    @field_validator("ticker")
    @classmethod
    def _uppercase(cls, value: str) -> str:
        return value.upper()


class RunResponse(BaseModel):
    """A research run as exposed by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    ticker: str
    status: RunStatus
    error: str | None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    created_at: datetime