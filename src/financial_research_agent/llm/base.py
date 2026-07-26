"""Provider-agnostic LLM interface and data models."""

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel


class Role(StrEnum):
    """Chat message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """One message in a chat conversation."""

    role: Role
    content: str


class LLMResponse(BaseModel):
    """Normalised LLM output, identical across providers."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int


class LLMClient(Protocol):
    """Interface every LLM provider must satisfy."""

    async def complete(
        self, messages: list[ChatMessage], *, temperature: float = 0.0
    ) -> LLMResponse:
        """Generate a completion for the given messages."""
        ...