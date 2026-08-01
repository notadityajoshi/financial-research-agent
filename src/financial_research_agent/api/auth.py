"""API-key authentication dependency."""

import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from financial_research_agent.config import get_settings
from financial_research_agent.logging_config import get_logger

log = get_logger(__name__)

_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(provided: str | None = Security(_header)) -> None:
    """Reject the request unless a valid X-API-Key is presented.

    With no keys configured, auth is DISABLED (local dev only) and a
    warning is logged once per request.
    """
    valid = get_settings().api_key_set
    if not valid:
        log.warning("auth_disabled", reason="no API keys configured")
        return
    if provided is not None and any(
        secrets.compare_digest(provided, key) for key in valid
    ):
        return
    raise HTTPException(status_code=401, detail="invalid or missing API key")
