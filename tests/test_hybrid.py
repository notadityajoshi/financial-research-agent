"""Offline hybrid retrieval tests: fake embedder + in-memory Qdrant."""

import pytest
from qdrant_client import AsyncQdrantClient

from financial_research_agent.ingestion.chunking import Chunk
from financial_research_agent.retrieval.bm25_index import BM25Index
from financial_research_agent.retrieval.hybrid import HybridRetriever
from financial_research_agent.retrieval.vector_store import VectorStore

DIM = 8


class FakeEmbedder:
    """Keyword-count embedder; blind to terms outside its vocabulary."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        keywords = ("revenue", "risk", "growth", "debt")
        return [
            [float(t.lower().count(k)) for k in keywords] + [0.1] * (DIM - 4)
            for t in texts
        ]


TEXTS = [
    "Revenue revenue revenue increased strongly this year.",
    "Risk risk factors include growth of competition and revenue pressure.",
    "The XR9950 accelerator platform shipped in volume.",
]


def _chunk(i: int, text: str) -> Chunk:
    return Chunk(
        text=text,
        index=i,
        token_count=len(text.split()),
        metadata={"ticker": "TEST", "form_type": "10-K", "filing_date": "2026-01-01"},
    )


@pytest.fixture
async def retriever() -> HybridRetriever:
    store = VectorStore(AsyncQdrantClient(location=":memory:"), FakeEmbedder(), dim=DIM)
    await store.index_chunks([_chunk(i, t) for i, t in enumerate(TEXTS)])
    return HybridRetriever(store, BM25Index(await store.scroll_all()))


async def test_lexical_rescues_dense_blind_spot(retriever: HybridRetriever) -> None:
    results = await retriever.search("XR9950", limit=3)
    assert results and "XR9950" in results[0].text


async def test_both_lists_beats_one_list(retriever: HybridRetriever) -> None:
    results = await retriever.search("revenue risk", limit=3)
    assert "Risk risk factors" in results[0].text  # ranks high in dense AND bm25


async def test_limit_respected(retriever: HybridRetriever) -> None:
    assert len(await retriever.search("revenue", limit=2)) <= 2


def test_empty_index_returns_nothing() -> None:
    assert BM25Index([]).search("anything") == []