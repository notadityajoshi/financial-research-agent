"""Offline tests for structured LLM output parsing and retry."""

import pytest
from pydantic import BaseModel

from financial_research_agent.llm.base import ChatMessage, Role
from financial_research_agent.llm.structured import generate_structured


class Item(BaseModel):
    name: str
    score: int


class ScriptedLLM:
    """Returns canned replies in order; counts calls."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self.calls = 0

    async def complete(self, messages, *, temperature: float = 0.0):
        from financial_research_agent.llm.base import LLMResponse

        reply = self._replies[self.calls]
        self.calls += 1
        return LLMResponse(content=reply, model="fake", input_tokens=0, output_tokens=0)


MSG = [ChatMessage(role=Role.USER, content="go")]


async def test_clean_json_parsed() -> None:
    llm = ScriptedLLM(['{"name": "a", "score": 1}'])
    item = await generate_structured(llm, MSG, Item)
    assert (item.name, item.score) == ("a", 1)


async def test_fenced_json_with_prose_parsed() -> None:
    llm = ScriptedLLM(['Sure! ```json\n{"name": "b", "score": 2}\n``` done'])
    assert (await generate_structured(llm, MSG, Item)).score == 2


async def test_retry_recovers_from_garbage() -> None:
    llm = ScriptedLLM(["not json at all", '{"name": "c", "score": 3}'])
    item = await generate_structured(llm, MSG, Item)
    assert item.score == 3
    assert llm.calls == 2


async def test_raises_after_retries() -> None:
    llm = ScriptedLLM(["junk", "more junk"])
    with pytest.raises(ValueError, match="failed to produce valid Item"):
        await generate_structured(llm, MSG, Item)
