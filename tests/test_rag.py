"""Offline RAG node tests: fake SEC, fake embedder, fake LLM, in-memory Qdrant."""

import json

import pytest
from qdrant_client import AsyncQdrantClient

from financial_research_agent.agents.rag import make_filing_analyst, make_index_filing
from financial_research_agent.agents.state import ResearchState
from financial_research_agent.ingestion.chunking import Chunk
from financial_research_agent.integrations.sec_edgar import Filing
from financial_research_agent.llm.base import LLMResponse
from financial_research_agent.retrieval.vector_store import VectorStore
from tests.test_graph import FILING
from tests.test_vector_store import FakeEmbedder


class FakeSECDownload:
    async def download_filing(self, filing: Filing) -> str:
        return (
            "<html><body><p>Risk risk competition intensifying.</p>"
            "<p>Revenue growth strategy expanding.</p></body></html>"
        )


class FakeLLM:
    async def complete(self, messages, *, temperature: float = 0.0):
        payload = {
            "items": [
                {
                    "title": "Competition",
                    "detail": "Competition is intensifying.",
                    "severity": "high",
                    "source_ids": [1, 99],
                }
            ]
        }
        return LLMResponse(
            content=json.dumps(payload), model="fake", input_tokens=0, output_tokens=0
        )


@pytest.fixture
async def store() -> VectorStore:
    return VectorStore(AsyncQdrantClient(location=":memory:"), FakeEmbedder(), dim=8)


async def test_index_filing_node(store: VectorStore) -> None:
    node = make_index_filing(FakeSECDownload(), store)
    out = await node(ResearchState(ticker="NVDA", filings=[FILING]))
    assert "errors" not in out
    assert len(await store.scroll_all()) >= 1


async def test_index_filing_requires_10k(store: VectorStore) -> None:
    node = make_index_filing(FakeSECDownload(), store)
    out = await node(ResearchState(ticker="NVDA"))
    assert out["errors"][0].node == "index_filing"


async def test_filing_analyst_resolves_citations(store: VectorStore) -> None:
    await store.index_chunks(
        [
            Chunk(
                text="Risk risk competition ahead of us.",
                index=0,
                token_count=6,
                metadata={
                    "ticker": "NVDA",
                    "form_type": "10-K",
                    "filing_date": "2026-01-01",
                },
            )
        ]
    )
    out = await make_filing_analyst(store, FakeLLM())(ResearchState(ticker="NVDA"))
    insight = out["filing_insights"][0]
    assert len(insight.evidence) == 1  # out-of-range id 99 dropped
    assert insight.evidence[0].form_type == "10-K"
    assert insight.evidence[0].excerpt.startswith("Risk risk")


async def test_filing_analyst_without_evidence_errors(store: VectorStore) -> None:
    out = await make_filing_analyst(store, FakeLLM())(ResearchState(ticker="NVDA"))
    assert out["errors"][0].node == "analyze_filing"
