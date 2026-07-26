"""Offline vector store tests: fake embedder + in-memory Qdrant. No LLM calls."""

import pytest
from qdrant_client import AsyncQdrantClient

from financial_research_agent.ingestion.chunking import Chunk
from financial_research_agent.retrieval.vector_store import VectorStore

DIM = 8


class FakeEmbedder:
    """Deterministic keyword embedder: no network, no models."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        keywords = ("revenue", "risk", "growth", "debt")
        return [
            [float(text.lower().count(k)) for k in keywords] + [0.1] * (DIM - 4)
            for text in texts
        ]


def _chunk(index: int, text: str) -> Chunk:
    return Chunk(
        text=text,
        index=index,
        token_count=len(text.split()),
        metadata={"ticker": "TEST", "form_type": "10-K", "filing_date": "2026-01-01"},
    )


@pytest.fixture
async def store() -> VectorStore:
    s = VectorStore(
        AsyncQdrantClient(location=":memory:"), FakeEmbedder(), dim=DIM
    )
    await s.index_chunks(
        [
            _chunk(0, "Revenue revenue revenue increased this year."),
            _chunk(1, "Risk risk risk factors include competition."),
            _chunk(2, "Unrelated administrative boilerplate text."),
        ]
    )
    return s


async def test_search_ranks_relevant_first(store: VectorStore) -> None:
    results = await store.search("revenue", limit=3)
    assert results[0].text.startswith("Revenue")


async def test_metadata_preserved(store: VectorStore) -> None:
    results = await store.search("risk", limit=1)
    assert results[0].metadata["ticker"] == "TEST"
    assert "text" not in results[0].metadata


async def test_idempotent_reindex(store: VectorStore) -> None:
    await store.index_chunks([_chunk(0, "Revenue revenue revenue increased this year.")])
    results = await store.search("revenue", limit=10)
    assert len(results) == 3  # overwrite, not duplicate