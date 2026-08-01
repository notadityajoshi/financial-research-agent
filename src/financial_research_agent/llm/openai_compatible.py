"""LLM client for any OpenAI-compatible API (Ollama, OpenAI)."""

from typing import cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from financial_research_agent.llm.base import ChatMessage, LLMResponse
from financial_research_agent.logging_config import get_logger

log = get_logger(__name__)


class OpenAICompatibleClient:
    """Implements LLMClient against any OpenAI-compatible endpoint."""

    def __init__(self, model: str, api_key: str, base_url: str | None = None) -> None:
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(
        self, messages: list[ChatMessage], *, temperature: float = 0.0
    ) -> LLMResponse:
        """Generate a completion; normalise the provider response."""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=cast(
                list[ChatCompletionMessageParam],
                [{"role": m.role.value, "content": m.content} for m in messages],
            ),
            temperature=temperature,
        )
        usage = response.usage
        result = LLMResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )
        log.info(
            "llm_completion",
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=cast(
                list[ChatCompletionMessageParam],
                [{"role": m.role.value, "content": m.content} for m in messages],
            ),
            temperature=temperature,
        )
        return result
