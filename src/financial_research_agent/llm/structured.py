"""Get schema-validated Pydantic objects out of an LLM."""

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from financial_research_agent.llm.base import ChatMessage, LLMClient, Role
from financial_research_agent.logging_config import get_logger

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?|```", re.IGNORECASE)


def _extract_json(text: str) -> str:
    """Strip code fences and slice from first '{' to last '}'."""
    cleaned = _FENCE.sub("", text)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        msg = "no JSON object found in LLM reply"
        raise ValueError(msg)
    return cleaned[start : end + 1]


async def generate_structured[T: BaseModel](
    llm: LLMClient,
    messages: list[ChatMessage],
    model_cls: type[T],
    *,
    retries: int = 1,
) -> T:
    """Call the LLM and validate its reply against `model_cls`.

    On parse/validation failure, retries once with the error appended
    so the model can correct itself. Raises after final failure.
    """
    schema = json.dumps(model_cls.model_json_schema())
    attempt_messages = [
        *messages,
        ChatMessage(
            role=Role.SYSTEM,
            content=(
                "Respond with ONLY a single JSON object matching this JSON schema. "
                f"No prose, no markdown fences.\nSchema: {schema}"
            ),
        ),
    ]

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        response = await llm.complete(attempt_messages, temperature=0.0)
        try:
            payload = json.loads(_extract_json(response.content))
            result = model_cls.model_validate(payload)
            log.info(
                "structured_output_ok", model_cls=model_cls.__name__, attempt=attempt
            )
            return result
        except (ValueError, ValidationError) as exc:
            last_error = exc
            log.warning(
                "structured_output_retry",
                model_cls=model_cls.__name__,
                attempt=attempt,
                error=str(exc)[:200],
            )
            attempt_messages.append(
                ChatMessage(
                    role=Role.USER,
                    content=f"Invalid. Error: {exc}. Return ONLY the corrected JSON object.",
                )
            )
    msg = f"LLM failed to produce valid {model_cls.__name__}: {last_error}"
    raise ValueError(msg)
