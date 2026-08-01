"""Shared asynchronous HTTP client with timeouts, retries and logging."""

import httpx
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from financial_research_agent.logging_config import get_logger

log = get_logger(__name__)

DEFAULT_TIMEOUT = httpx.Timeout(10.0, connect=5.0)

RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


def create_http_client(headers: dict[str, str] | None = None) -> httpx.AsyncClient:
    """Return a configured async HTTP client (use as `async with`)."""
    return httpx.AsyncClient(
        timeout=DEFAULT_TIMEOUT,
        headers=headers,
        follow_redirects=True,
    )


def _is_retryable(exc: BaseException) -> bool:
    """True for transient network errors and retryable HTTP statuses."""
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return False


def _log_retry(retry_state: RetryCallState) -> None:
    """Log each retry attempt with its wait time."""
    wait = retry_state.next_action.sleep if retry_state.next_action else 0.0
    log.warning("http_retry", attempt=retry_state.attempt_number, wait_seconds=wait)


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=8),
    before_sleep=_log_retry,
    reraise=True,
)
async def get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, str] | None = None,
) -> httpx.Response:
    """GET a URL; retry up to 3 times on transient failures.

    Raises httpx.HTTPStatusError on non-retryable or final failure.
    """
    response = await client.get(url, params=params)
    response.raise_for_status()
    return response
